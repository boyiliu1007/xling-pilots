"""PASMR stage 2, self-feedback RL (SRL), in Experiment 2's setting.

Per step, PROMPTS_PER_STEP RL-half questions x 1 sample (the authors' n_samples_per_prompt 1, train batch 128):
  1. the policy answers the Swahili question (Experiment 2's student prompt, thinking mode, T 1.0, top-p 1.0,
     MAX_TOKENS); the answer should start with <translation_start> English question <translation_end>;
  2. the pivot English question is fed back to the current policy (the English prompt of the English vote, thinking
     mode, same sampling, EN_MAX_TOKENS) for an English answer A_En;
  3. F = 1 if the Swahili answer A_T equals A_En, 0.1 if they differ (or A_En is missing), 0 on a format error (no
     pivot, or no finished boxed Swahili answer). Answers compared like every Experiment 2 reward (_same);
  4. REINFORCE++ (as the authors' OpenRLHF run): F normalised over the batch (--normalize_reward), per-token rewards
     r_i = -KL_COEF * (log pi(a_i) - log pi_PAM(a_i)) with F added at the last token, returns G_i = sum_{k>=i}
     GAMMA^(k-i) r_k, advantages whitened over all tokens of the batch; loss = -mean over tokens of A_i log pi(a_i);
     one optimizer step per batch, so the PPO ratio is 1 and its clip is inactive.
The policy starts from the PAM checkpoint of the same data and seed; PAM is also the KL reference (its LoRA weights are
swapped in for the reference pass). Deviations from the paper (PREREGISTRATION 2026-11-22 PASMR entry): GAMMA 1.0
(paper 0.99: with 1-2k-token thinking outputs 0.99^L leaves early tokens no reward), LoRA lr 5e-6 (authors 5e-7 full
fine-tuning; same 10x mapping as Experiment 2's other runs), boxed answers instead of '###', English answers up to
4096 tokens in thinking mode, 300 steps of 128 prompts (= D's 38,400 rollouts) on the RL half.
Run name: pasmr__<data>__exact[__q<n>-steps<n>]__s<seed> (the optional variant only for smoke tests).
"""
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import PIVOT_CLOSE, PIVOT_OPEN

log = logging.getLogger("votedistill.pasmr.srl")
STEPS = 300
PROMPTS_PER_STEP = 128
MAX_TOKENS = 2048          # Swahili rollout, as D
EN_MAX_TOKENS = 4096       # pivot English answer (thinking mode)
TEMPERATURE, TOP_P = 1.0, 1.0
LR = 5e-6
KL_COEF = 0.01
GAMMA = 1.0
MAX_GRAD_NORM = 1.0
R_AGREE, R_DISAGREE, R_FORMAT = 1.0, 0.1, 0.0
SAVE_EVERY = 25
SAVE_TEXT_EVERY = 10
EN_PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."   # English vote's
LOG_HEADER = ("step,loss,grad_norm,reward,pivot_ok,sw_finished,en_finished,agree,format_error,correct,lid_ok,"
              "correct_lid,mean_len,mean_kl,t_gen,t_train\n")
_PIVOT = re.compile(re.escape(PIVOT_OPEN) + r"(.*?)" + re.escape(PIVOT_CLOSE), re.S)
_VARIANT = re.compile(r"(?:q(?P<q>\d+))?-?(?:steps(?P<steps>\d+))?")


def parse_variant(v):
    """'' -> defaults; 'q4-steps2' -> 4 prompts per step, 2 steps (smoke tests)."""
    m = _VARIANT.fullmatch(v or "")
    if m is None:
        raise SystemExit(f"bad PASMR variant {v} (expected q<n>-steps<n>)")
    return dict(prompts_per_step=int(m["q"]) if m["q"] else PROMPTS_PER_STEP,
                steps=int(m["steps"]) if m["steps"] else STEPS)


def extract_pivot(text):
    """The English question between the pivot delimiters (first occurrence), or None."""
    m = _PIVOT.search(text or "")
    if m is None:
        return None
    return m.group(1).strip() or None


def pasmr_reward(sw_text, sw_finish, en_text, en_finish):
    """-> (F, facts). Format error: no pivot, or no finished boxed Swahili answer."""
    from ..answers_math import extract_pred
    from ..exp2.grpo import _same
    pivot = extract_pivot(sw_text)
    a_t = extract_pred(sw_text, thinking=True)[0] if sw_finish == "stop" else None
    a_en = extract_pred(en_text, thinking=True)[0] if en_text is not None and en_finish == "stop" else None
    if pivot is None or a_t is None:
        f = R_FORMAT
    else:
        f = R_AGREE if a_en is not None and _same(a_t, a_en) else R_DISAGREE
    return f, {"pivot": pivot, "pred": a_t, "en_pred": a_en, "pivot_ok": pivot is not None,
               "sw_finished": a_t is not None, "en_finished": a_en is not None,
               "agree": f == R_AGREE, "format_error": f == R_FORMAT}


def token_returns(f, kl, beta=KL_COEF, gamma=GAMMA):
    """Per-token returns G_i = sum_{k>=i} gamma^(k-i) r_k with r_k = -beta * kl_k and f added at the last token."""
    r = -beta * np.asarray(kl, dtype=np.float64)
    if len(r):
        r[-1] += f
    g, run = np.zeros_like(r), 0.0
    for i in range(len(r) - 1, -1, -1):
        run = r[i] + gamma * run
        g[i] = run
    return g


def en_prompt_ids(tok, question):
    text = tok.apply_chat_template([{"role": "user", "content": EN_PROMPT.format(question=question)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def rl_rows(data):
    from ..exp2.trainsets import _gsm8k_rows
    from .data import load_split
    _, rl = load_split(data)
    keep = set(rl)
    return [r for r in _gsm8k_rows(data) if r["qid"] in keep]


def _token_logprobs(model, prompt, comp, grad):
    import torch
    ids = torch.tensor(prompt + comp, dtype=torch.long).unsqueeze(0).cuda()
    L = len(comp)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, logits_to_keep=L + 1).logits[:, :-1, :]
    lp = torch.log_softmax(logits.float() / TEMPERATURE, -1)
    return lp.gather(-1, torch.tensor(comp, device=lp.device).view(1, -1, 1))[0, :, 0]


def train_srl(name, data, seed, variant, init_name):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from ..copsd.collator import make_collator
    from ..copsd.config import OFFICIAL
    from ..copsd.train import _load_rng, _save_rng, latest_ckpt, prune_old_states
    from ..exp2.grpo import LID_MIN, _same, lid_fraction, schedule
    from ..exp2.runs import final_step, paths
    from ..exp2.screen import _prompt_ids
    from ..gpu import wait_for_gpu
    from ..paths import MODEL
    from ..utils import versions, write_json

    v = parse_variant(variant)
    per_step, n_steps = v["prompts_per_step"], v["steps"]
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    start = latest_ckpt(ck) or 0
    if start >= n_steps:
        log.info("SRL already complete")
        return
    i_rd, i_ck = paths(init_name)
    i_step = final_step(init_name)
    pam = i_ck / f"step-{i_step}" / "adapter"
    if not (pam.parent / "COMPLETE").exists():
        raise SystemExit(f"PAM checkpoint {pam.parent} missing: train {init_name} first")
    rows = rl_rows(data)
    sched = schedule(len(rows), seed, n_steps, per_step)
    write_json(rd / "config.json", {
        "model": MODEL, "method": "PASMR stage 2, self-feedback RL (pasmr/srl.py)", "doc": __doc__, "data": data,
        "questions": len(rows), "question_set": "RL half (pasmr/data.py split.json)", "steps": n_steps,
        "prompts_per_step": per_step, "samples_per_prompt": 1, "max_tokens": MAX_TOKENS, "en_max_tokens": EN_MAX_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "lr": LR, "kl_coef": KL_COEF, "gamma": GAMMA,
        "max_grad_norm": MAX_GRAD_NORM, "rewards": {"agree": R_AGREE, "disagree": R_DISAGREE, "format": R_FORMAT},
        "init_and_kl_reference": {"run": init_name, "step": i_step, "adapter": str(pam)}, "seed": seed,
        "lora": {k: OFFICIAL[k] for k in ("lora_r", "lora_alpha", "lora_target_modules")}, "versions": versions(),
        "created": datetime.now().isoformat(timespec="seconds")})
    wait_for_gpu()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    col = make_collator(tok, OFFICIAL["max_length"], False, False)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_cpu_loras=2, max_model_len=EN_MAX_TOKENS + 512, max_num_seqs=min(per_step, 256), seed=seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                                             lora_dropout=0.0, target_modules=OFFICIAL["lora_target_modules"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    pam_sd = load_file(str(pam / "adapter_model.safetensors"))
    log_csv, samples_f = rd / "train_log.csv", rd / "rollouts.jsonl"
    if start:
        d = ck / f"step-{start}"
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        _load_rng(st["rng"])
        if log_csv.exists():   # drop lines written after this checkpoint by the interrupted run
            lines = log_csv.read_text().splitlines()
            log_csv.write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start]) + "\n")
        if samples_f.exists():
            keep = [l for l in samples_f.read_text().splitlines() if json.loads(l)["step"] <= start]
            samples_f.write_text("".join(l + "\n" for l in keep))
        log.info(f"RESUMED from {d}")
    else:
        set_peft_model_state_dict(model, pam_sd)
        log.info(f"initialised from PAM {pam}")
    if not log_csv.exists():
        log_csv.write_text(LOG_HEADER)
    import os
    sync_dir = Path(f"/tmp/votedistill_pasmr_adapter_{os.environ.get('SLURM_JOB_ID', os.getpid())}")

    for step in range(start + 1, n_steps + 1):
        t0 = time.time()
        if sync_dir.exists():
            shutil.rmtree(sync_dir)
        model.save_pretrained(str(sync_dir))
        lora = LoRARequest(f"policy{step}", step, str(sync_dir))
        qs = [rows[i] for i in sched[step - 1]]
        p_ids = [_prompt_ids(col, r, "student") for r in qs]
        sp = lambda j, off, mt: SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=-1, min_p=0.0,
                                               max_tokens=mt, seed=seed * 1_000_003 + step * 1000 + off + j)
        sw = [o.outputs[0] for o in llm.generate([{"prompt_token_ids": p} for p in p_ids],
                                                 [sp(j, 0, MAX_TOKENS) for j in range(len(qs))],
                                                 lora_request=lora, use_tqdm=False)]
        pivots = [extract_pivot(c.text) for c in sw]
        todo = [j for j, pv in enumerate(pivots) if pv]
        en = {}
        if todo:
            outs = llm.generate([{"prompt_token_ids": en_prompt_ids(tok, pivots[j])} for j in todo],
                                [sp(j, 500, EN_MAX_TOKENS) for j in todo], lora_request=lora, use_tqdm=False)
            en = {j: o.outputs[0] for j, o in zip(todo, outs)}
        t_gen = time.time() - t0

        recs, items = [], []
        for j, (q, p, c) in enumerate(zip(qs, p_ids, sw)):
            e = en.get(j)
            f, s = pasmr_reward(c.text, c.finish_reason, e.text if e else None, e.finish_reason if e else None)
            lid = lid_fraction(c.text)
            correct = s["pred"] is not None and _same(str(q["Answer"]), s["pred"])   # analysis only, not in the reward
            items.append((p, list(c.token_ids), f))
            recs.append({"step": step, "qid": q["qid"], "reward": f, "len": len(c.token_ids), "correct": correct,
                         "lid": lid, "lid_ok": lid >= LID_MIN, "correct_lid": correct and lid >= LID_MIN,
                         **{k: s[k] for k in ("pred", "en_pred", "pivot_ok", "sw_finished", "en_finished", "agree",
                                              "format_error")},
                         **({"text": c.text, "pivot": s["pivot"], "en_text": e.text if e else None}
                            if step % SAVE_TEXT_EVERY == 0 else {})})

        # REINFORCE++: old-policy and PAM log-probs (no grad), KL rewards, returns, batch-whitened advantages
        model.eval()
        old = [_token_logprobs(model, p, comp, False).cpu() if comp else None for p, comp, _ in items]
        cur = {k: t.detach().clone() for k, t in get_peft_model_state_dict(model).items()}
        set_peft_model_state_dict(model, pam_sd)
        ref = [_token_logprobs(model, p, comp, False).cpu() if comp else None for p, comp, _ in items]
        set_peft_model_state_dict(model, cur)
        del cur
        fs = np.array([x[2] for x in items], dtype=np.float64)
        fn = (fs - fs.mean()) / (fs.std() + 1e-8) if fs.std() > 0 else np.zeros_like(fs)
        kls = [(o - r).numpy() if o is not None else np.zeros(0) for o, r in zip(old, ref)]
        gs = [token_returns(fn[j], kls[j]) for j in range(len(items))]
        allg = np.concatenate([g for g in gs if len(g)]) if any(len(g) for g in gs) else np.zeros(1)
        mu, sd = allg.mean(), allg.std() + 1e-8
        n_tok = max(int(sum(len(g) for g in gs)), 1)

        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for (p, comp, _), g in zip(items, gs):
            if not comp:
                continue
            adv = torch.tensor((g - mu) / sd, dtype=torch.float32, device="cuda")
            lp = _token_logprobs(model, p, comp, True)
            loss = -(adv * lp).sum() / n_tok
            loss.backward()
            loss_sum += float(loss.detach())
            del lp, loss, adv
        gn = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM))
        opt.step()
        t_all = time.time() - t0
        m = lambda k: float(np.mean([x[k] for x in recs]))
        mean_kl = float(np.mean(np.concatenate([k for k in kls if len(k)]))) if any(len(k) for k in kls) else 0.0
        with open(log_csv, "a") as fh:
            fh.write(f"{step},{loss_sum:.6f},{gn:.5f},{m('reward'):.4f},{m('pivot_ok'):.4f},{m('sw_finished'):.4f},"
                     f"{m('en_finished'):.4f},{m('agree'):.4f},{m('format_error'):.4f},{m('correct'):.4f},"
                     f"{m('lid_ok'):.4f},{m('correct_lid'):.4f},{m('len'):.0f},{mean_kl:.5f},{t_gen:.1f},{t_all - t_gen:.1f}\n")
        with open(samples_f, "a") as fh:
            for x in recs:
                fh.write(json.dumps(x) + "\n")
        log.info(f"step {step}: reward {m('reward'):.3f} pivot {m('pivot_ok'):.3f} agree {m('agree'):.3f} "
                 f"format_err {m('format_error'):.3f} correct {m('correct'):.3f} lid_ok {m('lid_ok'):.3f} "
                 f"kl {mean_kl:.4f} grad_norm {gn:.3f} gen {t_gen:.0f}s train {t_all - t_gen:.0f}s")
        if step % SAVE_EVERY == 0 or step == n_steps:
            d = ck / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "rng": _save_rng(), "step": step}, d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
            prune_old_states(ck, step)
            log.info(f"checkpoint saved: {d}")
