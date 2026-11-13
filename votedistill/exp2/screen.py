"""Pre-training screen of teacher contexts (after Zou et al. 2026, arXiv:2610.22512).

The initial student is the base model (LoRA B = 0), so every quantity below is what training would see at step 0.
  1. Sample k student rollouts per training problem with the training prompt and sampling (T 1.1, top-p 0.95,
     top-k 20, 2048 new tokens). The same rollouts are scored under every teacher context (paired comparison).
  2. For each rollout token y_t, with s = student prompt and c = teacher context:
       delta_t = log p(y_t | s, y_<t) - log p(y_t | c, y_<t)          (raw logits, T = 1)
       D       = mean_t |delta_t|  per rollout, then mean over rollouts       (Zou et al.'s D(I))
       signed  = mean_t delta_t    (sampled estimate of KL(student || teacher))
       fkl     = sum_v p_c log(p_c / p_s) per token at T 1.1                  (COPSD loss, unclipped)
       fkl_clip= the same with the official per-(token, vocab) clamp at 0.05 (the loss training actually uses)
  Every sequence is scored on its own (no padding), so positions are exact.

Outputs in results/exp2/screen/<name>/: config.json, rollouts.jsonl, tokens_<context>.npz (per-token arrays),
scores.jsonl (per rollout x context), report.txt.
"""
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from ..data import load_copsd_train
from ..gpu import wait_for_gpu
from ..paths import COPSD_COMMIT, MODEL
from ..utils import read_jsonl, versions, write_json, write_jsonl
from ..copsd.collator import make_collator, prepare_rows
from ..copsd.config import OFFICIAL, SEED
from .levels import EXP2_RESULTS, TEACHER_CONTEXTS, teacher_rows

log = logging.getLogger("votedistill.exp2.screen")
BOXED = "\\boxed"
RENAMED = {"L0+instr": "L0-same"}
CHUNK = 512   # positions per logit chunk (memory: CHUNK x vocab x fp32)


def screen_dir(name):
    return EXP2_RESULTS / "screen" / name


def select_problems(n_rows, n_problems, seed=SEED):
    idx = list(range(n_rows))
    if n_problems and n_problems < n_rows:
        idx = sorted(random.Random(seed).sample(idx, n_problems))
    return idx


def _prompt_ids(collator, row, key):
    """Unpadded token ids of one row's student or teacher prompt (a batch of one has no padding)."""
    b = collator([row])
    return b[f"{key}_prompts"][0].tolist()


def generate_rollouts(sd, rows, idx, k, tok):
    from vllm import LLM, SamplingParams
    p = sd / "rollouts.jsonl"
    if p.exists():
        log.info(f"[skip] {p} exists")
        return read_jsonl(p)
    collator = make_collator(tok, OFFICIAL["max_length"], False, False)   # student prompt does not depend on flags
    sp_ids = {i: _prompt_ids(collator, rows[i], "student") for i in idx}
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, max_model_len=8192, seed=SEED)
    jobs = [(i, j) for i in idx for j in range(k)]
    sps = [SamplingParams(n=1, temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"],
                          min_p=0.0, max_tokens=OFFICIAL["max_completion_length"], seed=SEED * 1_000_003 + i * 100 + j)
           for i, j in jobs]
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": sp_ids[i]} for i, _ in jobs], sps, use_tqdm=False)
    log.info(f"generated {len(jobs)} rollouts in {time.time() - t0:.0f}s")
    recs = [{"problem": i, "sample": j, "student_prompt_ids": sp_ids[i], "completion_ids": list(o.outputs[0].token_ids),
             "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}
            for (i, j), o in zip(jobs, outs)]
    write_jsonl(p, recs)
    import gc
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return recs


def _completion_logits(model, prompt_ids, comp_ids):
    import torch
    ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device="cuda")
    L = len(comp_ids)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=ids, logits_to_keep=L + 1).logits[0, :-1]   # [L, V]: position t predicts comp_ids[t]


def token_stats(s_logits, t_logits, comp_ids, temp=OFFICIAL["temperature"], clip=OFFICIAL["jsd_token_clip"]):
    """Per-token delta (T=1, sampled token), forward KL(teacher||student) at T=temp, unclipped and clipped."""
    import torch
    import torch.nn.functional as F
    y = torch.tensor(comp_ids, device=s_logits.device)
    delta, fkl, fklc = [], [], []
    for a in range(0, len(comp_ids), CHUNK):
        s, t, yy = s_logits[a:a + CHUNK].float(), t_logits[a:a + CHUNK].float(), y[a:a + CHUNK]
        ls, lt = F.log_softmax(s, -1), F.log_softmax(t, -1)
        delta.append((ls.gather(-1, yy[:, None]) - lt.gather(-1, yy[:, None]))[:, 0])
        lsT, ltT = F.log_softmax(s / temp, -1), F.log_softmax(t / temp, -1)
        kl = ltT.exp() * (ltT - lsT)                          # == F.kl_div(lsT, ltT, log_target=True), as in loss.py
        fkl.append(kl.sum(-1))
        fklc.append(kl.clamp(max=clip).sum(-1))
    return [torch.cat(x).cpu().numpy() for x in (delta, fkl, fklc)]


def screen(name, contexts, n_problems=500, k=2, lang="swa"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sd = screen_dir(name)
    sd.mkdir(parents=True, exist_ok=True)
    unknown = [c for c in contexts if c not in TEACHER_CONTEXTS]
    if unknown:
        raise SystemExit(f"unknown teacher contexts {unknown}; known: {list(TEACHER_CONTEXTS)}")
    cfg = {"model": MODEL, "lang": lang, "n_problems": n_problems, "k": k, "seed": SEED, "copsd_commit": COPSD_COMMIT,
           "sampling": {"temperature": OFFICIAL["temperature"], "top_p": OFFICIAL["top_p"], "top_k": OFFICIAL["top_k"],
                        "max_tokens": OFFICIAL["max_completion_length"]},
           "kl_temperature": OFFICIAL["temperature"], "kl_clip": OFFICIAL["jsd_token_clip"], "versions": versions()}
    old = json.loads((sd / "config.json").read_text()) if (sd / "config.json").exists() else None
    if old and {k_: old[k_] for k_ in ("model", "lang", "n_problems", "k", "seed", "sampling")} != \
            {k_: cfg[k_] for k_ in ("model", "lang", "n_problems", "k", "seed", "sampling")}:
        raise SystemExit(f"{sd} was made with different settings: {old}")
    write_json(sd / "config.json", {**cfg, "contexts": {c: TEACHER_CONTEXTS[c] for c in contexts}})
    wait_for_gpu()
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = prepare_rows(load_copsd_train(lang), lang)
    idx = select_problems(len(rows), n_problems)
    recs = generate_rollouts(sd, rows, idx, k, tok)

    todo = [c for c in contexts if not (sd / f"tokens_{c}.npz").exists()]
    if not todo:
        log.info("all contexts scored")
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    # student log-probs are shared by all contexts; keep logits only for the current rollout
    tprompts = {}
    for c in todo:
        trows, flags = teacher_rows(rows, c)                    # over all rows, so the derangement is subset-independent
        col = make_collator(tok, OFFICIAL["max_length"], **flags)
        tprompts[c] = {i: _prompt_ids(col, trows[i], "teacher") for i in idx}
        log.info(f"{c}: teacher prompt example (problem {idx[0]}):\n{tok.decode(tprompts[c][idx[0]])}")
    arrays = {c: {"delta": [], "fkl": [], "fkl_clip": []} for c in todo}
    scores = []
    t0 = time.time()
    for n, r in enumerate(recs):
        comp = r["completion_ids"]
        if not comp:
            continue
        s_log = _completion_logits(model, r["student_prompt_ids"], comp)
        for c in todo:
            t_log = _completion_logits(model, tprompts[c][r["problem"]], comp)
            d, f, fc = token_stats(s_log, t_log, comp)
            for key, v in (("delta", d), ("fkl", f), ("fkl_clip", fc)):
                arrays[c][key].append(v.astype(np.float32))
            scores.append({"context": c, "problem": r["problem"], "sample": r["sample"], "n_tokens": len(comp),
                           "teacher_prompt_tokens": len(tprompts[c][r["problem"]]),
                           "D": float(np.abs(d).mean()), "signed": float(d.mean()), "fkl": float(f.mean()),
                           "fkl_clip": float(fc.mean())})
            del t_log
        del s_log
        if (n + 1) % 100 == 0:
            log.info(f"scored {n + 1}/{len(recs)} rollouts x {len(todo)} contexts ({time.time() - t0:.0f}s)")
    for c in todo:
        np.savez_compressed(sd / f"tokens_{c}.npz",
                            **{k_: np.array(v, dtype=object) for k_, v in arrays[c].items()},
                            problem=np.array([r["problem"] for r in recs if r["completion_ids"]]),
                            sample=np.array([r["sample"] for r in recs if r["completion_ids"]]))
    old_scores = read_jsonl(sd / "scores.jsonl") if (sd / "scores.jsonl").exists() else []
    write_jsonl(sd / "scores.jsonl", [s for s in old_scores if s["context"] not in todo] + scores)
    log.info(f"scored {len(todo)} contexts in {time.time() - t0:.0f}s")


def _per_problem(scores, ctx, key, keep=None):
    """Mean of `key` per problem (rollouts averaged first), as {problem: value}."""
    by = {}
    for s in scores:
        if s["context"] == ctx and (keep is None or keep(s)):
            by.setdefault(s["problem"], []).append(s[key])
    return {p: float(np.mean(v)) for p, v in by.items()}


def _boot_mean(x, n=10_000, seed=SEED):
    x = np.asarray(x, dtype=float)
    m = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def _migrate_names(sd):
    """Rename contexts of screens made before a context was renamed (L0+instr -> L0-same)."""
    for old, new in RENAMED.items():
        if (sd / f"tokens_{old}.npz").exists() and not (sd / f"tokens_{new}.npz").exists():
            (sd / f"tokens_{old}.npz").rename(sd / f"tokens_{new}.npz")
            S = read_jsonl(sd / "scores.jsonl")
            write_jsonl(sd / "scores.jsonl", [{**s, "context": RENAMED.get(s["context"], s["context"])} for s in S])
            log.info(f"renamed context {old} -> {new} in {sd}")


def screen_report(name, baseline="L1"):
    from ..stats import paired_bootstrap
    sd = screen_dir(name)
    _migrate_names(sd)
    cfg = json.loads((sd / "config.json").read_text())
    scores = read_jsonl(sd / "scores.jsonl")
    recs = {(r["problem"], r["sample"]): r for r in read_jsonl(sd / "rollouts.jsonl")}
    ctxs = [c for c in TEACHER_CONTEXTS if any(s["context"] == c for s in scores)]
    fin = lambda s: recs[(s["problem"], s["sample"])]["finish_reason"] != "length"   # noqa: E731
    T = [f"Pre-training screen: {name}",
         f"{cfg['n_problems']} problems x {cfg['k']} student rollouts (training prompt and sampling), base model = step-0 student",
         f"rollouts: {len(recs)}; truncated at {cfg['sampling']['max_tokens']} tokens: "
         f"{np.mean([r['finish_reason'] == 'length' for r in recs.values()]):.3f}; contain \\boxed: "
         f"{np.mean([BOXED in r['text'] for r in recs.values()]):.3f}",
         "Per-problem means, 95% bootstrap CI over problems. D = mean |log p_student - log p_teacher| of the sampled "
         "token (Zou et al.); fkl/fkl_clip = COPSD per-token loss at step 0 (unclipped / official clip).", ""]
    T.append(f"  {'context':<18} {'D':<24} {'signed':<9} {'fkl':<9} {'fkl_clip':<9} {'D finished':<11} {'D truncated':<11} prompt tok")
    for c in ctxs:
        D = _per_problem(scores, c, "D")
        m, lo, hi = _boot_mean(list(D.values()))
        g = lambda key: np.mean(list(_per_problem(scores, c, key).values()))   # noqa: E731
        dfin = _per_problem(scores, c, "D", fin)
        dtr = _per_problem(scores, c, "D", lambda s: not fin(s))
        T.append(f"  {c:<18} {m:.4f} [{lo:.4f}, {hi:.4f}]  {g('signed'):+.4f}  {g('fkl'):.4f}   {g('fkl_clip'):.4f}   "
                 f"{np.mean(list(dfin.values())) if dfin else float('nan'):.4f}      "
                 f"{np.mean(list(dtr.values())) if dtr else float('nan'):.4f}      {g('teacher_prompt_tokens'):.0f}")
    if baseline in ctxs:
        T.append(f"\n  paired differences in D vs {baseline} (over problems):")
        B = _per_problem(scores, baseline, "D")
        for c in ctxs:
            if c == baseline:
                continue
            C = _per_problem(scores, c, "D")
            ps = sorted(set(B) & set(C))
            d, lo, hi = paired_bootstrap([B[p] for p in ps], [C[p] for p in ps])
            T.append(f"    {c:<18} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    placebo = [(c, c.replace("L3-shuffled-", "L3-")) for c in ctxs if c.startswith("L3-shuffled-")]
    for pc, rc in placebo:
        if rc not in ctxs:
            continue
        T.append(f"\n  real reference vs placebo ({rc} - {pc}), paired over problems:")
        for key, keep, lab in (("D", None, "D, all rollouts"), ("D", fin, "D, finished rollouts"),
                               ("fkl", None, "fkl (unclipped)"), ("fkl_clip", None, "fkl_clip (official)")):
            A, Bm = _per_problem(scores, pc, key, keep), _per_problem(scores, rc, key, keep)
            ps = sorted(set(A) & set(Bm))
            d, lo, hi = paired_bootstrap([A[p] for p in ps], [Bm[p] for p in ps])
            T.append(f"    {lab:<24} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({len(ps)} problems)")
    T.append("\n  signal concentration and the official clip (per-token arrays):")
    T.append(f"  {'context':<18} {'|delta| mass in first 10% of tokens':<37} {'tokens with fkl > clip':<24} share of KL kept by clip")
    for c in ctxs:
        z = np.load(sd / f"tokens_{c}.npz", allow_pickle=True)
        tot = sum(np.abs(d).sum() for d in z["delta"])
        first = sum(np.abs(d[:max(1, len(d) // 10)]).sum() for d in z["delta"])
        fk, fc = np.concatenate(list(z["fkl"])), np.concatenate(list(z["fkl_clip"]))
        T.append(f"  {c:<18} {first / tot if tot else float('nan'):<37.2f} {np.mean(fk > cfg['kl_clip']):<24.3f} "
                 f"{fc.sum() / fk.sum() if fk.sum() else float('nan'):.2f}")
    T.append("\n  where the divergence sits: mean |delta| by relative position in the rollout (deciles), all rollouts")
    T.append(f"  {'context':<18} " + " ".join(f"{10 * i:>3}%" for i in range(10)))
    for c in ctxs:
        z = np.load(sd / f"tokens_{c}.npz", allow_pickle=True)
        prof = np.zeros(10)
        cnt = np.zeros(10)
        for d in z["delta"]:
            b = np.minimum((np.arange(len(d)) * 10) // len(d), 9)
            np.add.at(prof, b, np.abs(d))
            np.add.at(cnt, b, 1)
        T.append(f"  {c:<18} " + " ".join(f"{v:.2f}" for v in prof / np.maximum(cnt, 1)))
    text = "\n".join(T) + "\n"
    (sd / "report.txt").write_text(text)
    return text
