"""Offline test of the model as its own privileged judge (PREREGISTRATION 2026-11-20).

Can Qwen3-4B, reading a Swahili problem, one Swahili answer and (optionally) its OWN English voted solution, tell a
correct answer from a wrong one? Pointwise judge: score = P(Yes) / (P(Yes) + P(No)) at the first generated token of a
non-thinking chat reply (one forward pass). Gold answers only label the answers; the judge never sees them.

  python -m votedistill.exp2.judge sample   # 8 Swahili answers per question from the untrained model and D300
  python -m votedistill.exp2.judge score    # judge scores for every answer x condition
  python -m votedistill.exp2.judge report   # pair accuracies, bootstrap CIs, go rule
(run with BASE_MODEL=Qwen/Qwen3-4B)
"""
import itertools
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np

from ..answers_math import final_text
from ..utils import read_jsonl, setup_logging, write_json, write_jsonl
from .levels import EXP2_CKPT, EXP2_RESULTS

log = logging.getLogger("votedistill.exp2.judge")
OUT = EXP2_RESULTS / "judge"
N_QUESTIONS, Q_SEED, PAIR_SEED, PAIRS_PER_TYPE = 200, 2026, 7, 4
D300 = "grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s42"
POLICIES = {"base": None, "d300": EXP2_CKPT / D300 / "step-300" / "adapter"}
JUDGES = POLICIES                      # the same two models serve as judges
REFS = ("own", "none")
INPUTS = ("full", "final")

PROMPT_REF = (
    "You are grading an answer to a math word problem written in Swahili.\n"
    "A reference solution in English is provided. It is usually correct but may contain mistakes.\n\n"
    "Problem (Swahili):\n{question}\n\nReference solution (English):\n{ref}\n\nAnswer:\n{answer}\n\n"
    "Are the reasoning and the final answer correct? Reply with one word: Yes or No.")
PROMPT_NOREF = (
    "You are grading an answer to a math word problem written in Swahili.\n\n"
    "Problem (Swahili):\n{question}\n\nAnswer:\n{answer}\n\n"
    "Are the reasoning and the final answer correct? Reply with one word: Yes or No.")


def questions():
    from .grpo import question_rows
    rows = question_rows("gsm8k-mt3k", "votelidref")          # vote-accepted questions, with own English solution
    return sorted(random.Random(Q_SEED).sample(rows, N_QUESTIONS), key=lambda r: r["qid"])


# ---------------------------------------------------------------- 1. sample answers like RL rollouts
def sample():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from ..copsd.collator import make_collator
    from ..copsd.config import OFFICIAL
    from ..gpu import wait_for_gpu
    from ..paths import MODEL
    from .grpo import GROUP, MAX_TOKENS, TEMPERATURE, score
    from .screen import _prompt_ids
    out = OUT / "answers.jsonl"
    if out.exists():
        log.info(f"[skip] {out} exists")
        return
    qs = questions()
    wait_for_gpu()
    tok = AutoTokenizer.from_pretrained(MODEL)
    col = make_collator(tok, OFFICIAL["max_length"], False, False)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_model_len=4096, seed=Q_SEED)
    p_ids = [_prompt_ids(col, r, "student") for r in qs]
    recs = []
    for pi, (pol, adapter) in enumerate(POLICIES.items()):
        lora = LoRARequest(pol, pi + 1, str(adapter)) if adapter else None
        sps = [SamplingParams(n=GROUP, temperature=TEMPERATURE, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=MAX_TOKENS,
                              seed=Q_SEED * 1000 + j) for j in range(len(qs))]
        outs = llm.generate([{"prompt_token_ids": p} for p in p_ids], sps, lora_request=lora)
        for q, o in zip(qs, outs):
            for k, c in enumerate(o.outputs):
                s = score(c.text, c.finish_reason, q["Answer"], q["vote_answer"])
                recs.append({"policy": pol, "qid": q["qid"], "k": k, "text": c.text, "pred": s["pred"],
                             "finished": s["finished"], "correct": s["correct"], "lid_ok": s["lid_ok"],
                             "label": "unfinished" if not s["finished"] else
                                      ("correct" if s["correct"] and s["lid_ok"] else
                                       "wrong" if s["lid_ok"] else "not_swahili")})
        log.info(f"{pol}: " + " ".join(f"{l}={sum(r['label'] == l for r in recs if r['policy'] == pol)}"
                                      for l in ("correct", "wrong", "unfinished", "not_swahili")))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, recs)
    write_json(OUT / "questions.json", [{"qid": q["qid"], "question": q["problem_translated"], "gold": q["Answer"],
                                         "vote": q["vote_answer"], "ref": q["vote_ref"]} for q in qs])


# ---------------------------------------------------------------- 2. build pairs (incl. spliced answers)
def splice(text, old, new):
    """Replace only the final answer (the \\boxed{} content after </think>); the reasoning stays as written.
    (Replacing the old number everywhere after </think> also rewrote the solution's arithmetic, e.g. '68 - 4 = 220'.)"""
    head, sep, tail = text.partition("</think>")
    if not sep or not re.search(r"\\boxed\{", tail):
        return None
    new_tail = re.sub(r"\\boxed\{(?:[^{}]|\{[^{}]*\})*\}", lambda m: "\\boxed{" + str(new) + "}", tail)
    return None if new_tail == tail else head + sep + new_tail


def build_items_and_pairs():
    ans = list(read_jsonl(OUT / "answers.jsonl"))
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    rng = random.Random(PAIR_SEED)
    items, pairs = [], []

    def add(rec, text, kind):
        items.append({"id": len(items), "policy": rec["policy"], "qid": rec["qid"], "kind": kind, "text": text})
        return len(items) - 1

    for (pol, qid), grp in itertools.groupby(sorted(ans, key=lambda r: (r["policy"], r["qid"], r["k"])),
                                             key=lambda r: (r["policy"], r["qid"])):
        grp = list(grp)
        C = [r for r in grp if r["label"] == "correct"]
        W = [r for r in grp if r["label"] == "wrong"]
        U = [r for r in grp if r["label"] == "unfinished"]
        gold = qinfo[qid]["gold"]
        gold = int(gold) if float(gold).is_integer() else gold
        ids = {}

        def real(r):
            key = (r["k"], "real")
            if key not in ids:
                ids[key] = add(r, r["text"], "correct" if r["label"] == "correct" else r["label"])
            return ids[key]

        def take(lst, n):
            return rng.sample(lst, min(n, len(lst)))
        for c, w in take(list(itertools.product(C, W)), PAIRS_PER_TYPE):                      # P1
            pairs.append({"type": "P1", "policy": pol, "qid": qid, "better": real(c), "worse": real(w)})
        for c, w in take(list(itertools.product(C, W)), PAIRS_PER_TYPE):                      # P2: W with gold number
            t = splice(w["text"], w["pred"], gold)
            if t:
                pairs.append({"type": "P2", "policy": pol, "qid": qid, "better": real(c),
                              "worse": add(w, t, "wrong_reasoning_gold_number")})
        for c, w in take(list(itertools.product(C, W)), PAIRS_PER_TYPE):                      # P3: C with W's number
            t = splice(c["text"], c["pred"], w["pred"]) if w["pred"] else None                # vs W itself: same
            if t:                                                                             # final number, only
                pairs.append({"type": "P3", "policy": pol, "qid": qid,                        # reasoning differs
                              "better": add(c, t, "correct_reasoning_wrong_number"), "worse": real(w)})
        for c, u in take(list(itertools.product(C, U)), PAIRS_PER_TYPE):                      # P0: easy group
            pairs.append({"type": "P0", "policy": pol, "qid": qid, "better": real(c), "worse": real(u)})
    write_jsonl(OUT / "items.jsonl", items)
    write_jsonl(OUT / "pairs.jsonl", pairs)
    log.info(f"{len(items)} items, pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}"
                                                       for t in ("P1", "P2", "P3", "P0")))
    return items, pairs, qinfo


# ---------------------------------------------------------------- 3. judge scores
def p_yes(logprobs):
    """P(Yes) / (P(Yes) + P(No)) over the first token's top logprobs (variants like ' Yes', 'yes' pooled)."""
    y = n = 0.0
    for lp in logprobs.values():
        t = (lp.decoded_token or "").strip().lower()
        if t == "yes":
            y += float(np.exp(lp.logprob))
        elif t == "no":
            n += float(np.exp(lp.logprob))
    return 0.5 if y + n == 0 else y / (y + n)


def judge_score():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from ..gpu import wait_for_gpu
    from ..paths import MODEL
    out = OUT / "scores.jsonl"
    if out.exists():
        log.info(f"[skip] {out} exists")
        return
    items, pairs, qinfo = build_items_and_pairs()
    need = sorted({p["better"] for p in pairs} | {p["worse"] for p in pairs})
    tok = AutoTokenizer.from_pretrained(MODEL)

    def prompt(it, ref, inp):
        q = qinfo[it["qid"]]
        ans = it["text"] if inp == "full" else (final_text(it["text"], True) or "").strip()
        body = (PROMPT_REF.format(question=q["question"], ref=q["ref"], answer=ans) if ref == "own"
                else PROMPT_NOREF.format(question=q["question"], answer=ans))
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    wait_for_gpu()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_model_len=6144, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    recs = []
    for ji, (judge, adapter) in enumerate(JUDGES.items()):
        lora = LoRARequest(judge, ji + 1, str(adapter)) if adapter else None
        for ref in REFS:
            for inp in INPUTS:
                outs = llm.generate([prompt(items[i], ref, inp) for i in need], sp, lora_request=lora)
                for i, o in zip(need, outs):
                    recs.append({"item": i, "judge": judge, "ref": ref, "input": inp,
                                 "p_yes": p_yes(o.outputs[0].logprobs[0]),
                                 "first": o.outputs[0].text})
                log.info(f"scored judge={judge} ref={ref} input={inp}: {len(need)} items")
    write_jsonl(out, recs)


# ---------------------------------------------------------------- 4. report
def _boot_ci(per_q, n=5000, seed=0):
    """Question-clustered bootstrap of a mean over pairs: per_q = list of per-question arrays."""
    rng = np.random.default_rng(seed)
    flat = np.concatenate(per_q)
    bs = [np.concatenate([per_q[j] for j in rng.integers(0, len(per_q), len(per_q))]).mean() for _ in range(n)]
    return flat.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def report():
    pairs = list(read_jsonl(OUT / "pairs.jsonl"))
    S = {}
    for r in read_jsonl(OUT / "scores.jsonl"):
        S[(r["judge"], r["ref"], r["input"], r["item"])] = r["p_yes"]
    lines = ["Judge test (PREREGISTRATION 2026-11-20). Pair accuracy = P(better answer scored higher), ties 0.5.",
             "Pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}" for t in ("P1", "P2", "P3", "P0")), ""]

    def acc(judge, ref, inp, ptype):
        by_q = {}
        for p in pairs:
            if p["type"] != ptype:
                continue
            b, w = S[(judge, ref, inp, p["better"])], S[(judge, ref, inp, p["worse"])]
            by_q.setdefault((p["policy"], p["qid"]), []).append(1.0 if b > w else 0.5 if b == w else 0.0)
        return {k: np.array(v) for k, v in by_q.items()}

    for judge in JUDGES:
        for inp in INPUTS:
            for ptype in ("P1", "P2", "P3", "P0"):
                cells = []
                for ref in REFS:
                    m, lo, hi = _boot_ci(list(acc(judge, ref, inp, ptype).values()))
                    cells.append(f"{ref}: {100 * m:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]")
                a, b = acc(judge, "own", inp, ptype), acc(judge, "none", inp, ptype)
                d, lo, hi = _boot_ci([a[k] - b[k] for k in a])
                lines.append(f"judge={judge:5s} input={inp:5s} {ptype}  " + "  ".join(cells)
                             + f"  own-none {100 * d:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
            lines.append("")
    go = {}
    for inp in INPUTS:
        a, b = acc("base", "own", inp, "P1"), acc("base", "none", inp, "P1")
        m = _boot_ci(list(a.values()))[0]
        _, lo, _ = _boot_ci([a[k] - b[k] for k in a])
        go[inp] = (m, m >= 0.65 and lo > 0)
    lines.append("GO RULE (base judge, own solution, P1 >= 65% and own-none CI > 0): "
                 + "  ".join(f"{k}: {100 * m:.1f}% {'PASS' if ok else 'fail'}" for k, (m, ok) in go.items()))
    if any(ok for _, ok in go.values()):
        best = max(go, key=lambda k: go[k][0])
        pick = "final" if go["final"][1] and go[best][0] - go["final"][0] <= 0.02 else best
        lines.append(f"=> GO; RL judge input: {pick}")
    else:
        lines.append("=> NO GO: no judge RL (report as a comprehension result)")
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "logs" / f"{cmd}.log")
    {"sample": sample, "score": judge_score, "report": report}[cmd]()
