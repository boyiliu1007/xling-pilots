"""On-policy RL with verifiable rewards on Swahili GSM8K-MT (PREREGISTRATION 2026-11-15 ~17:00).

GRPO in its simplest on-policy form (one optimizer step per batch of rollouts, so the importance ratio is 1 and
PPO-style clipping is inactive; no KL penalty). Advantage = reward - mean reward of the question's group, without
dividing by the group's standard deviation; loss = -sum_i A_i * sum_t log p(y_it) / (number of rollouts *
MAX_TOKENS), a constant normaliser (Dr. GRPO, Liu et al. 2025). Groups whose rewards are all equal carry no
gradient and are skipped in the backward pass.

Rollouts: the student's Swahili prompt (COPSD style, thinking mode), sampled from the current policy with vLLM at
temperature 1.0, top-p 1.0, no repetition penalty (pure on-policy: earlier penalty-decoded data drifted to English).

Rewards (run variant, `REWARDS`), all 0/1:
  gold     the final answer (after </think>, finished) is equivalent to the gold answer   (ceiling; any language)
  goldsw   gold AND the whole output (thinking included) is Swahili by copsd.evaluate.language_of
  votesw   the final answer equals the model's own English majority vote AND the whole output is Swahili
           (gold-free; questions the vote did not accept are skipped)
  goldswv  goldsw + 0.5 x (answer equals the English vote AND whole output Swahili)  (gold plus English signal)
  goldswr  gold AND hack-resistant Swahili (`robust_sw`): goldsw at lr 5e-5 learned to pass language_of by repeating
           one Swahili function word ("jibu" was 51% of all test-output words), so this counts DISTINCT Swahili
           function words (>= SW_MIN_TYPES, and more than distinct English ones) and caps the most frequent word's share
"""
import json
import logging
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..answers_math import equiv, extract_pred, parse_answer
from ..copsd.collator import make_collator
from ..copsd.config import OFFICIAL
from ..paths import MODEL
from ..utils import versions, write_json, write_jsonl

log = logging.getLogger("votedistill.exp2.grpo")

REWARDS = ("gold", "goldsw", "votesw", "goldswv", "goldswr", "goldlid", "goldlidref", "votelid", "votelidref",
           "lidonly", "selfvotelid")
# Self-play transfer design (PREREGISTRATION 2026-11-16 ~10:30), SP3F-style (Sutawika et al., ACL 2026) rewards:
#   goldlid     correct (gold)         + lid_ok                         A: gold ceiling
#   goldlidref  correct (gold)         + lid_ok + ref_f1 (gold English solution)          B: SP3F-like, gold privilege
#   votelid     equals own English vote + lid_ok  (vote-accepted questions only)          C: gold-free, answer only
#   votelidref  equals own English vote + lid_ok + ref_f1 (own English voted answer text)   D: the proposal
#   lidonly     lid_ok only (vote-accepted questions, as C)                               E: control, language only
#   selfvotelid equals the group's own Swahili majority answer + lid_ok (as C)            F: control, no English
# lid_ok: >= LID_MIN of the prose (chunks with >= 3 words; math/LaTeX removed) is Swahili by lingua over ALL languages
# (a Swahili-vs-English-only detector passes the "jibu"-spam outputs; the all-language one passes 13% of them, 86% of
# base outputs). ref_f1: F1 between the sets of numbers in the rollout and in the English reference (a judge-free
# stand-in for SP3F's privileged pairwise judge; numbers are language-independent, and precision penalises spam).
LID_MIN = 0.7
SAVE_TEXT_EVERY = 10    # rollout texts kept every N steps, to inspect for reward hacking
SW_MIN_TYPES = 8        # goldswr: at least this many DISTINCT Swahili function words ...
SW_MAX_TOP_SHARE = 0.35  # ... and no single word above this share of all words (amendment 2026-11-16 ~00:30)
STEPS = 200
QUESTIONS_PER_STEP = 8
GROUP = 8
MAX_TOKENS = 2048
LR = 1e-5
MAX_GRAD_NORM = 1.0
TEMPERATURE = 1.0
SAVE_EVERY = 25
LOG_HEADER = ("step,loss,grad_norm,reward,correct,sw,correct_sw,vote_sw,finished,mean_len,"
              "groups_with_signal,questions,t_gen,t_train,lid_ok,correct_lid,vote_ok,ref_f1\n")
_LID = None


def lid_fraction(text):
    """Share of prose characters lingua (all languages) labels Swahili."""
    global _LID
    from lingua import Language, LanguageDetectorBuilder
    if _LID is None:
        _LID = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().with_preloaded_language_models().build()
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$|\\boxed\{[^}]*\}", " ", text, flags=re.S)
    chunks = [c.strip() for c in re.split(r"[\n.!?]+", t)]
    chunks = [c for c in chunks if len(re.findall(r"[^\W\d_]{2,}", c)) >= 3]
    if not chunks:
        return 0.0
    n = [len(c) for c in chunks]
    return sum(k for k, l in zip(n, _LID.detect_languages_in_parallel_of(chunks)) if l == Language.SWAHILI) / sum(n)


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers(text):
    out = set()
    for x in _NUM.findall(text or ""):
        try:
            v = float(x.replace(",", ""))
        except ValueError:
            continue
        out.add(round(v, 4))
    return out


def ref_f1(text, ref):
    a, b = numbers(text), numbers(ref)
    if not a or not b:
        return 0.0
    tp = len(a & b)
    return 0.0 if tp == 0 else 2 * tp / (len(a) + len(b))


_VARIANT = re.compile(r"(?P<kind>[a-z]+)(?:-lr(?P<lr>[0-9.]+(?:e-?[0-9]+)?))?(?:-q(?P<q>[0-9]+))?(?:-steps(?P<steps>[0-9]+))?")


def parse_variant(variant):
    """'goldsw' -> preregistered defaults; optional suffixes in this order: -lr<value> (amendment 18:45),
    -q<questions per step>, -steps<steps> (amendment 2026-11-16 ~00:00), e.g. 'goldsw-lr5e-5-q32-steps150'.
    -> dict(kind, lr, questions_per_step, steps), or None if the variant is not a GRPO variant."""
    m = _VARIANT.fullmatch(variant)
    if not m or m["kind"] not in REWARDS:
        return None
    return dict(kind=m["kind"], lr=float(m["lr"]) if m["lr"] else LR,
                questions_per_step=int(m["q"]) if m["q"] else QUESTIONS_PER_STEP,
                steps=int(m["steps"]) if m["steps"] else STEPS)


def parse_kind(variant):
    v = parse_variant(variant)
    if v is None:
        raise SystemExit(f"unknown GRPO variant {variant}; known {REWARDS} with optional -lr<v>-q<n>-steps<n>")
    return v


def _same(a, b):
    return a is not None and b is not None and equiv(a, parse_answer(a), b, parse_answer(b))


def robust_sw(text):
    """Swahili by distinct function-word types, not counts; rejects single-word spam (see goldswr)."""
    import collections
    from ..copsd.evaluate import EN_WORDS, SW_WORDS
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$|\\boxed\{[^}]*\}", " ", text, flags=re.S)
    w = re.findall(r"[a-zA-Z']+", t.lower())
    if not w:
        return False
    c = collections.Counter(w)
    sw, en = len(set(c) & SW_WORDS), len(set(c) & EN_WORDS)
    return sw >= SW_MIN_TYPES and sw > en and c.most_common(1)[0][1] / len(w) <= SW_MAX_TOP_SHARE


def score(text, finish_reason, gold, vote, ref=None):
    """Per-rollout facts the rewards are built from."""
    from ..copsd.evaluate import language_of
    pred, _ = extract_pred(text, thinking=True)
    fin = finish_reason == "stop" and pred is not None
    sw = language_of(text) == "sw"
    correct = fin and _same(str(gold), pred)
    vote_ok = fin and vote is not None and _same(str(vote), pred)
    return {"pred": pred, "finished": fin, "sw": sw, "correct": correct, "correct_sw": correct and sw,
            "vote_sw": vote_ok and sw, "correct_swr": correct and robust_sw(text), "vote_ok": vote_ok,
            "lid": (lid := lid_fraction(text)), "lid_ok": lid >= LID_MIN, "correct_lid": correct and lid >= LID_MIN,
            "ref_f1": ref_f1(text, ref) if ref else 0.0}


def group_majority(outputs):
    """TTRL-style pseudo-label: the most common final answer among the group's finished samples (None if no answer
    occurs at least twice)."""
    preds = [extract_pred(c.text, thinking=True)[0] for c in outputs if c.finish_reason == "stop"]
    clusters = []   # [representative, count], answers merged by the evaluation's equivalence check
    for x in (x for x in preds if x is not None):
        for cl in clusters:
            if _same(cl[0], x):
                cl[1] += 1
                break
        else:
            clusters.append([x, 1])
    if not clusters:
        return None
    ans, n = max(clusters, key=lambda cl: cl[1])
    return ans if n >= 2 else None


def reward(kind, s):
    if kind == "gold":
        return float(s["correct"])
    if kind == "goldsw":
        return float(s["correct_sw"])
    if kind == "votesw":
        return float(s["vote_sw"])
    if kind == "goldswr":
        return float(s["correct_swr"])
    if kind == "goldswv":
        return float(s["correct_sw"]) + 0.5 * float(s["vote_sw"])
    if kind == "lidonly":
        return float(s["lid_ok"])
    if kind == "selfvotelid":   # score() was called with vote = the group's Swahili majority answer
        return float(s["vote_ok"]) + float(s["lid_ok"])
    if kind in ("goldlid", "goldlidref", "votelid", "votelidref"):
        ok = s["correct"] if kind.startswith("gold") else s["vote_ok"]
        return float(ok) + float(s["lid_ok"]) + (s["ref_f1"] if kind.endswith("ref") else 0.0)
    raise ValueError(kind)


def question_rows(data, kind):
    from .trainsets import _gsm8k_rows, voted_references
    rows = _gsm8k_rows(data)
    if kind in ("votesw", "goldswv", "votelid", "votelidref", "lidonly", "selfvotelid"):
        refs = voted_references(data)
        rows = [{**r, "vote_answer": refs[r["qid"]][1] if r["qid"] in refs else None,
                 "vote_ref": refs[r["qid"]][0] if r["qid"] in refs else None} for r in rows]
        if kind in ("votesw", "votelid", "votelidref", "lidonly", "selfvotelid"):   # same questions as C
            rows = [r for r in rows if r["vote_answer"] is not None]
    for r in rows:   # privileged reference for the ref_f1 term
        r["ref"] = r.get("vote_ref") if kind == "votelidref" else r["solution"] if kind == "goldlidref" else None
    return rows


def schedule(n_rows, seed, steps=STEPS, per_step=QUESTIONS_PER_STEP):
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    need = steps * per_step
    order = (order * (need // n_rows + 1))[:need]
    return [order[i * per_step:(i + 1) * per_step] for i in range(steps)]


def train_grpo(name, data, seed, kind):
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from ..copsd.train import _load_rng, _save_rng, latest_ckpt, prune_old_states
    from ..gpu import wait_for_gpu
    from .runs import paths
    from .screen import _prompt_ids

    v = parse_kind(kind)
    kind, lr, per_step, n_steps = v["kind"], v["lr"], v["questions_per_step"], v["steps"]
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    start = latest_ckpt(ck) or 0
    if start >= n_steps:
        log.info("GRPO already complete")
        return
    rows = question_rows(data, kind)
    sched = schedule(len(rows), seed, n_steps, per_step)
    write_json(rd / "config.json", {
        "model": MODEL, "method": "GRPO (grpo.py)", "doc": __doc__, "reward": kind, "data": data,
        "questions": len(rows), "steps": n_steps, "questions_per_step": per_step, "group": GROUP,
        "max_tokens": MAX_TOKENS, "lr": lr, "max_grad_norm": MAX_GRAD_NORM, "temperature": TEMPERATURE, "seed": seed,
        "lora": {k: OFFICIAL[k] for k in ("lora_r", "lora_alpha", "lora_target_modules")}, "versions": versions(),
        "created": datetime.now().isoformat(timespec="seconds")})
    wait_for_gpu()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    col = make_collator(tok, OFFICIAL["max_length"], False, False)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_cpu_loras=2, max_model_len=4096, max_num_seqs=min(per_step * GROUP, 256), seed=seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                                             lora_dropout=0.0, target_modules=OFFICIAL["lora_target_modules"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    log_csv = rd / "train_log.csv"
    if start:
        d = ck / f"step-{start}"
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        _load_rng(st["rng"])
        if log_csv.exists():
            lines = log_csv.read_text().splitlines()
            log_csv.write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start]) + "\n")
        rf = rd / "rollouts.jsonl"
        if rf.exists():   # drop rollouts logged after this checkpoint by the interrupted run
            keep = [l for l in rf.read_text().splitlines() if json.loads(l)["step"] <= start]
            rf.write_text("".join(l + "\n" for l in keep))
        log.info(f"RESUMED from {d}")
    if not log_csv.exists():
        log_csv.write_text(LOG_HEADER)
    sync_dir = Path(f"/tmp/votedistill_grpo_adapter_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    samples_f = rd / "rollouts.jsonl"

    for step in range(start + 1, n_steps + 1):
        t0 = time.time()
        lora = None
        if step > 1:
            if sync_dir.exists():
                shutil.rmtree(sync_dir)
            model.save_pretrained(str(sync_dir))
            lora = LoRARequest(f"policy{step}", step, str(sync_dir))
        qs = [rows[i] for i in sched[step - 1]]
        p_ids = [_prompt_ids(col, r, "student") for r in qs]
        sps = [SamplingParams(n=GROUP, temperature=TEMPERATURE, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=MAX_TOKENS,
                              seed=seed * 1_000_003 + step * 1000 + j) for j in range(len(qs))]
        outs = llm.generate([{"prompt_token_ids": p} for p in p_ids], sps, lora_request=lora, use_tqdm=False)
        t_gen = time.time() - t0

        recs, groups = [], []
        for q, p, o in zip(qs, p_ids, outs):
            g = []
            vote = group_majority(o.outputs) if kind == "selfvotelid" else \
                None if kind == "lidonly" else q.get("vote_answer")
            for c in o.outputs:
                s = score(c.text, c.finish_reason, q["Answer"], vote, q.get("ref"))
                g.append((list(c.token_ids), reward(kind, s), s))
                recs.append({"step": step, "qid": q["qid"], "reward": reward(kind, s), "len": len(c.token_ids),
                             **{k: s[k] for k in ("pred", "finished", "sw", "correct", "vote_sw", "correct_swr",
                                                  "vote_ok", "lid", "lid_ok", "correct_lid", "ref_f1")},
                             **({"text": c.text} if step % SAVE_TEXT_EVERY == 0 else {})})
            groups.append((p, g))

        model.train()
        opt.zero_grad(set_to_none=True)
        norm = len(qs) * GROUP * MAX_TOKENS
        loss_sum, n_signal = 0.0, 0
        for p, g in groups:
            r = np.array([x[1] for x in g])
            if r.max() == r.min():
                continue
            n_signal += 1
            adv = r - r.mean()
            for (comp, _, _), a in zip(g, adv):
                if a == 0 or not comp:
                    continue
                ids = torch.tensor(p + comp, dtype=torch.long).unsqueeze(0).cuda()
                L = len(comp)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, logits_to_keep=L + 1).logits[:, :-1, :]
                lp = torch.log_softmax(logits.float() / TEMPERATURE, -1)
                tok_lp = lp.gather(-1, torch.tensor(comp, device=lp.device).view(1, -1, 1))[..., 0]
                loss = -(float(a) * tok_lp.sum()) / norm
                loss.backward()
                loss_sum += float(loss.detach())
                del logits, lp, tok_lp, loss
        gn = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)) if n_signal else 0.0
        if n_signal:
            opt.step()
        t_all = time.time() - t0
        m = lambda k: float(np.mean([x[k] for x in recs]))
        with open(log_csv, "a") as fh:
            fh.write(f"{step},{loss_sum:.6f},{gn:.5f},{m('reward'):.4f},{m('correct'):.4f},{m('sw'):.4f},"
                     f"{float(np.mean([x['correct'] and x['sw'] for x in recs])):.4f},{m('vote_sw'):.4f},{m('finished'):.4f},"
                     f"{m('len'):.0f},{n_signal},{len(qs)},{t_gen:.1f},{t_all - t_gen:.1f},"
                     f"{m('lid_ok'):.4f},{m('correct_lid'):.4f},{m('vote_ok'):.4f},{m('ref_f1'):.4f}\n")
        with open(samples_f, "a") as fh:
            for x in recs:
                fh.write(json.dumps(x) + "\n")
        log.info(f"step {step}: reward {m('reward'):.3f} correct {m('correct'):.3f} sw {m('sw'):.3f} finished "
                 f"{m('finished'):.3f} signal {n_signal}/{len(qs)} grad_norm {gn:.3f} gen {t_gen:.0f}s train {t_all - t_gen:.0f}s")
        if step % SAVE_EVERY == 0:
            d = ck / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "rng": _save_rng(), "step": step}, d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
            prune_old_states(ck, step)
            log.info(f"checkpoint saved: {d}")
