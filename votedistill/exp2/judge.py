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


# ================================================================ second test (PREREGISTRATION 2026-11-20 ~15:30)
P4_SEED, P4_MIN_GAP = 11, 0.25


def _gold_solutions():
    from .trainsets import _gsm8k_rows
    return {r["qid"]: r["solution"] for r in _gsm8k_rows("gsm8k-mt3k")}


def p4_pairs(items, qinfo):
    """Wrong vs wrong: 'better' = higher intermediate-step F1 vs the gold rationale (gap >= P4_MIN_GAP)."""
    from .grpo import step_f1
    gold = _gold_solutions()
    rng = random.Random(P4_SEED)
    wrong = [it for it in items if it["kind"] == "wrong"]
    for it in wrong:
        it["step_f1_gold"] = step_f1(it["text"], gold[it["qid"]], qinfo[it["qid"]]["question"])
    pairs = []
    for key, grp in itertools.groupby(sorted(wrong, key=lambda it: (it["policy"], it["qid"], it["id"])),
                                      key=lambda it: (it["policy"], it["qid"])):
        cand = [(a, b) if a["step_f1_gold"] > b["step_f1_gold"] else (b, a)
                for a, b in itertools.combinations(list(grp), 2)
                if abs(a["step_f1_gold"] - b["step_f1_gold"]) >= P4_MIN_GAP]
        for a, b in rng.sample(cand, min(PAIRS_PER_TYPE, len(cand))):
            pairs.append({"type": "P4", "policy": key[0], "qid": key[1], "better": a["id"], "worse": b["id"]})
    return pairs


def mask_ref(ref, vote):
    """Own English solution with the boxed answer and every standalone occurrence of the vote number hidden."""
    t = re.sub(r"\\boxed\{(?:[^{}]|\{[^{}]*\})*\}", lambda m: "\\boxed{?}", ref)
    v = str(vote)
    alts = {v, v.rstrip("0").rstrip(".") if "." in v else v, f"{float(v):,.0f}" if re.fullmatch(r"\d+", v) else v}
    for a in sorted(alts, key=len, reverse=True):
        t = re.sub(r"(?<![\d.,])" + re.escape(a) + r"(?![\d]|\.\d|,\d)", lambda m: "?", t)
    return t


def judge_score2():
    """Judges B (reasoning-only likelihood) and D (masked reference) for every item in pairs + P4 pairs."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from ..copsd.collator import make_collator
    from ..copsd.config import OFFICIAL
    from ..gpu import wait_for_gpu
    from ..paths import MODEL
    from .grpo import question_rows
    from .screen import _prompt_ids
    out = OUT / "scores2.jsonl"
    if out.exists():
        log.info(f"[skip] {out} exists")
        return
    items = list(read_jsonl(OUT / "items.jsonl"))
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    pairs = list(read_jsonl(OUT / "pairs.jsonl")) + p4_pairs(items, qinfo)
    write_jsonl(OUT / "pairs2.jsonl", pairs)
    log.info("pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}" for t in ("P1", "P2", "P3", "P0", "P4")))
    need = sorted({p["better"] for p in pairs} | {p["worse"] for p in pairs})
    tok = AutoTokenizer.from_pretrained(MODEL)
    col = make_collator(tok, OFFICIAL["max_length"], False, False)
    rows = {r["qid"]: r for r in question_rows("gsm8k-mt3k", "votelidref")}
    student = {qid: _prompt_ids(col, rows[qid], "student") for qid in qinfo}

    def b_ids(it):
        """(full token ids, number of target tokens) for judge B."""
        think = it["text"].split("</think>", 1)[0]
        ctx = student[it["qid"]] + tok(think + "</think>\n\n$\\boxed{", add_special_tokens=False)["input_ids"]
        tgt = tok(str(qinfo[it["qid"]]["vote"]) + "}", add_special_tokens=False)["input_ids"]
        return ctx + tgt, len(tgt)

    def d_prompt(it, inp):
        q = qinfo[it["qid"]]
        ans = it["text"] if inp == "full" else (final_text(it["text"], True) or "").strip()
        body = PROMPT_REF.format(question=q["question"], ref=mask_ref(q["ref"], q["vote"]), answer=ans)
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    wait_for_gpu()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_model_len=6144, enable_prefix_caching=True, seed=0)
    recs = []
    for ji, (judge, adapter) in enumerate(JUDGES.items()):
        lora = LoRARequest(judge, ji + 1, str(adapter)) if adapter else None
        enc = [b_ids(items[i]) for i in need]
        outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in enc],
                            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), lora_request=lora)
        for i, (ids, k), o in zip(need, enc, outs):
            lp = sum(o.prompt_logprobs[j][ids[j]].logprob for j in range(len(ids) - k, len(ids)))
            recs.append({"item": i, "judge": judge, "variant": "B", "input": "think", "score": lp})
        log.info(f"scored B judge={judge}")
        for inp in INPUTS:
            outs = llm.generate([d_prompt(items[i], inp) for i in need],
                                SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), lora_request=lora)
            for i, o in zip(need, outs):
                recs.append({"item": i, "judge": judge, "variant": "D", "input": inp,
                             "score": p_yes(o.outputs[0].logprobs[0])})
            log.info(f"scored D judge={judge} input={inp}")
    write_jsonl(out, recs)


def report2():
    from .grpo import ref_f1
    items = {it["id"]: it for it in read_jsonl(OUT / "items.jsonl")}
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    pairs = list(read_jsonl(OUT / "pairs2.jsonl"))
    S = {}
    for r in read_jsonl(OUT / "scores2.jsonl"):
        S[(r["variant"], r["judge"], r["input"], r["item"])] = r["score"]
    for r in read_jsonl(OUT / "scores.jsonl"):
        if r["ref"] == "own":
            S[("yesno", r["judge"], r["input"], r["item"])] = r["p_yes"]
    for i, it in items.items():
        S[("numF1", "-", "full", i)] = ref_f1(it["text"], qinfo[it["qid"]]["ref"])
    variants = [("B", j, "think") for j in JUDGES] + [("D", j, x) for j in JUDGES for x in INPUTS] + \
               [("yesno", j, x) for j in JUDGES for x in INPUTS] + [("numF1", "-", "full")]

    def acc(v, ptype):
        by_q = {}
        for p in pairs:
            if p["type"] == ptype and (*v, p["better"]) in S:
                b, w = S[(*v, p["better"])], S[(*v, p["worse"])]
                by_q.setdefault((p["policy"], p["qid"]), []).append(1.0 if b > w else 0.5 if b == w else 0.0)
        return {k: np.array(x) for k, x in by_q.items()}
    lines = ["Second judge test (PREREGISTRATION 2026-11-20 ~15:30). Pair accuracy, ties 0.5; CI = question-clustered.",
             "Pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}" for t in ("P1", "P2", "P3", "P0", "P4")),
             "", f"{'variant':24s} " + "  ".join(f"{t:>20s}" for t in ("P1", "P3", "P4", "P4 ties")) + "   P4 - numF1"]
    num = acc(("numF1", "-", "full"), "P4")
    res = {}
    for v in variants:
        cells = []
        for t in ("P1", "P3", "P4"):
            m, lo, hi = _boot_ci(list(acc(v, t).values()))
            cells.append(f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]")
            res[(v, t)] = (m, lo)
        a4 = acc(v, "P4")
        ties = np.mean(np.concatenate(list(a4.values())) == 0.5)
        d, lo, hi = _boot_ci([a4[k] - num[k] for k in a4])
        lines.append(f"{'-'.join(v):24s} " + "  ".join(f"{c:>20s}" for c in cells) + f"  {100 * ties:17.1f}%"
                     + f"   {100 * d:+5.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]")
    lines.append("")
    passing = [v for v in variants if v[0] in ("B", "D") and res[(v, "P4")][0] >= 0.60 and res[(v, "P4")][1] > 0.50
               and res[(v, "P1")][0] >= 0.90]
    lines.append("GO RULE 2 (B/D variants: P4 >= 60%, CI lo > 50%, P1 >= 90%): "
                 + (", ".join(f"{'-'.join(v)} PASS (P4 {100 * res[(v, 'P4')][0]:.1f}, lo {100 * res[(v, 'P4')][1]:.1f})"
                              for v in passing) or "no variant passes"))
    if passing:
        best = max(passing, key=lambda v: res[(v, "P4")][0])
        lines.append(f"=> RL candidate: {'-'.join(best)}" + (" (weak: CI lo < 52%)" if res[(best, 'P4')][1] < 0.52 else ""))
    txt = "\n".join(lines)
    (OUT / "report2.txt").write_text(txt + "\n")
    print(txt)


# ================================================================ third test (PREREGISTRATION 2026-11-20 ~15:40)
P5_SEED, P5_PER_GROUP, LABELER = 13, 3, "Qwen/Qwen3-8B"
THINK_SP = dict(temperature=0.6, top_p=0.95, top_k=20)

LABEL_PROMPT = (
    "You are an expert math grader. Below are a math word problem in Swahili, its correct solution in English, and two "
    "student answers in Swahili. Both final answers are wrong. Decide which answer has more correct reasoning, i.e. more "
    "correct steps before its first mistake.\n\nProblem (Swahili):\n{question}\n\nCorrect solution (English):\n{gold}\n\n"
    "Answer A:\n{a}\n\nAnswer B:\n{b}\n\nEnd your reply with exactly 'Verdict: A' or 'Verdict: B'.")
HEAD = ("You are grading an answer to a math word problem written in Swahili.\n"
        "A reference solution in English is provided. It is usually correct but may contain mistakes.\n\n"
        "Problem (Swahili):\n{question}\n\nReference solution (English):\n{ref}\n\n")
GRADED = HEAD + ("Answer:\n{answer}\n\nHow much of the answer's reasoning is correct, on a scale from 0 (nothing correct) "
                 "to 9 (fully correct)? Reply with one digit.")
GRADED_THINK = HEAD + ("Answer:\n{answer}\n\nHow much of the answer's reasoning is correct, on a scale from 0 (nothing "
                       "correct) to 9 (fully correct)? End your reply with exactly 'Score: k' where k is one digit.")
GRADED_EN = HEAD.replace("written in Swahili", "originally written in Swahili (translated to English below)") + (
    "Answer (translated to English):\n{answer}\n\nHow much of the answer's reasoning is correct, on a scale from 0 "
    "(nothing correct) to 9 (fully correct)? Reply with one digit.")
PAIRWISE = HEAD + ("Answer A:\n{a}\n\nAnswer B:\n{b}\n\nWhich answer is closer to a fully correct solution? "
                   "Reply with one letter: A or B.")
TRANSLATE = ("Translate the following Swahili solution of a math word problem into English. Keep every number and "
             "equation exactly as written. Output only the translation.\n\n{answer}")


def _final(it):
    return (final_text(it["text"], True) or "").strip()


def _expected_digit(lps):
    p = np.zeros(10)
    for lp in lps.values():
        t = (lp.decoded_token or "").strip()
        if len(t) == 1 and t in "0123456789":
            p[int(t)] += float(np.exp(lp.logprob))
    return float((p * np.arange(10)).sum() / p.sum()) if p.sum() > 0 else float("nan")


def _p_letter(lps):
    a = b = 0.0
    for lp in lps.values():
        t = (lp.decoded_token or "").strip()
        a += float(np.exp(lp.logprob)) if t == "A" else 0.0
        b += float(np.exp(lp.logprob)) if t == "B" else 0.0
    return 0.5 if a + b == 0 else a / (a + b)


def label3():
    """P5: wrong-vs-wrong pairs labelled by Qwen3-8B (thinking) with the gold solution, both orders, agreed only."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from ..gpu import wait_for_gpu
    out = OUT / "pairs5.jsonl"
    if out.exists():
        log.info(f"[skip] {out} exists")
        return
    items = list(read_jsonl(OUT / "items.jsonl"))
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    gold = _gold_solutions()
    rng = random.Random(P5_SEED)
    cand = []
    wrong = [it for it in items if it["kind"] == "wrong"]
    for key, grp in itertools.groupby(sorted(wrong, key=lambda it: (it["policy"], it["qid"], it["id"])),
                                      key=lambda it: (it["policy"], it["qid"])):
        combos = list(itertools.combinations(list(grp), 2))
        cand += [(key, a, b) for a, b in rng.sample(combos, min(P5_PER_GROUP, len(combos)))]
    tok = AutoTokenizer.from_pretrained(LABELER)

    def prompt(qid, a, b):
        body = LABEL_PROMPT.format(question=qinfo[qid]["question"], gold=gold[qid], a=_final(a), b=_final(b))
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    wait_for_gpu()
    llm = LLM(model=LABELER, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=10240, seed=P5_SEED)
    sp = SamplingParams(max_tokens=6144, seed=P5_SEED, **THINK_SP)
    prompts = [prompt(k[1], a, b) for k, a, b in cand] + [prompt(k[1], b, a) for k, a, b in cand]
    outs = llm.generate(prompts, sp)

    def verdict(text):
        m = re.findall(r"Verdict:\s*\**\s*([AB])", text.split("</think>")[-1])
        return m[-1] if m else None
    n = len(cand)
    pairs, agree, parsed = [], 0, 0
    for i, ((pol, qid), a, b) in enumerate(cand):
        v1, v2 = verdict(outs[i].outputs[0].text), verdict(outs[n + i].outputs[0].text)
        parsed += v1 is not None and v2 is not None
        if v1 and v2 and (v1 == "A") == (v2 == "B"):            # same answer chosen in both orders
            agree += 1
            better, worse = (a, b) if v1 == "A" else (b, a)
            pairs.append({"type": "P5", "policy": pol, "qid": qid, "better": better["id"], "worse": worse["id"]})
    write_jsonl(out, pairs)
    write_json(OUT / "pairs5.meta.json", {"candidates": n, "both_parsed": parsed, "agreed": agree, "labeler": LABELER})
    log.info(f"P5: {n} candidate pairs, both verdicts parsed {parsed}, agreed {agree}")


def score3():
    """Judges A, C, E, F (untrained 4B, own English solution, final solution only)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from ..gpu import wait_for_gpu
    from ..paths import MODEL
    out = OUT / "scores3.jsonl"
    if out.exists():
        log.info(f"[skip] {out} exists")
        return
    items = list(read_jsonl(OUT / "items.jsonl"))
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    pairs = [p for p in read_jsonl(OUT / "pairs2.jsonl") if p["type"] in ("P1", "P3", "P4")] + \
        list(read_jsonl(OUT / "pairs5.jsonl"))
    need = sorted({p["better"] for p in pairs} | {p["worse"] for p in pairs})
    tok = AutoTokenizer.from_pretrained(MODEL)

    def chat(body, think=False):
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=think)

    def fmt(tpl, it, **kw):
        q = qinfo[it["qid"]]
        return tpl.format(question=q["question"], ref=q["ref"], **kw)
    wait_for_gpu()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=8192,
              enable_prefix_caching=True, seed=0)
    one = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    recs = []
    # A graded
    outs = llm.generate([chat(fmt(GRADED, items[i], answer=_final(items[i]))) for i in need], one)
    recs += [{"variant": "A", "item": i, "score": _expected_digit(o.outputs[0].logprobs[0])} for i, o in zip(need, outs)]
    log.info("scored A")
    # E translate, then graded in English
    tr = llm.generate([chat(TRANSLATE.format(answer=_final(items[i]))) for i in need],
                      SamplingParams(max_tokens=1024, temperature=0.0))
    en = {i: o.outputs[0].text.strip() for i, o in zip(need, tr)}
    outs = llm.generate([chat(fmt(GRADED_EN, items[i], answer=en[i])) for i in need], one)
    recs += [{"variant": "E", "item": i, "score": _expected_digit(o.outputs[0].logprobs[0]), "translation": en[i]}
             for i, o in zip(need, outs)]
    log.info("scored E")
    # F thinking judge
    outs = llm.generate([chat(fmt(GRADED_THINK, items[i], answer=_final(items[i])), think=True) for i in need],
                        SamplingParams(max_tokens=3072, seed=0, logprobs=20, **THINK_SP))
    for i, o in zip(need, outs):
        c = o.outputs[0]
        score = float("nan")
        toks = [tok.decode([t]) for t in c.token_ids]
        starts = np.cumsum([0] + [len(t) for t in toks])          # char offset where each token starts
        text = "".join(toks)
        think_end = text.find("</think>")
        m = list(re.finditer(r"Score:\s*\**\s*(?=\d)", text[think_end:])) if think_end >= 0 else []
        if m:
            digit_at = think_end + m[-1].end()                     # char index of the score digit
            pos = int(np.searchsorted(starts, digit_at, side="right") - 1)
            if pos < len(c.logprobs) and toks[pos].strip()[:1] in list("0123456789"):
                score = _expected_digit(c.logprobs[pos])
        recs.append({"variant": "F", "item": i, "score": score, "finished": c.finish_reason == "stop",
                     "tail": c.text[-200:]})
    log.info("scored F")
    # C pairwise (pair-level records)
    pc = [p for p in pairs]
    prompts = []
    for p in pc:
        b, w = items[p["better"]], items[p["worse"]]
        prompts.append(chat(fmt(PAIRWISE, b, a=_final(b), b=_final(w))))     # better shown as A
        prompts.append(chat(fmt(PAIRWISE, b, a=_final(w), b=_final(b))))     # better shown as B
    outs = llm.generate(prompts, one)
    for k, p in enumerate(pc):
        pa = _p_letter(outs[2 * k].outputs[0].logprobs[0])
        pb = _p_letter(outs[2 * k + 1].outputs[0].logprobs[0])
        recs.append({"variant": "C", "type": p["type"], "better": p["better"], "worse": p["worse"],
                     "pref_better": 0.5 * (pa + (1 - pb)), "pA_first": pa, "pA_second": pb})
    log.info("scored C")
    write_jsonl(out, recs)


def report3():
    from .grpo import ref_f1
    items = {it["id"]: it for it in read_jsonl(OUT / "items.jsonl")}
    qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
    pairs = [p for p in read_jsonl(OUT / "pairs2.jsonl") if p["type"] in ("P1", "P3", "P4")] + \
        list(read_jsonl(OUT / "pairs5.jsonl"))
    meta = json.load(open(OUT / "pairs5.meta.json"))
    S, C = {}, {}
    for r in read_jsonl(OUT / "scores3.jsonl"):
        if r["variant"] == "C":
            C[(r["better"], r["worse"])] = r["pref_better"]
        else:
            S[(r["variant"], r["item"])] = r["score"]
    for r in read_jsonl(OUT / "scores2.jsonl"):
        if r["judge"] == "base" and (r["variant"] == "B" or r["input"] == "final"):
            S[(r["variant"], r["item"])] = r["score"]
    for r in read_jsonl(OUT / "scores.jsonl"):
        if r["judge"] == "base" and r["ref"] == "own" and r["input"] == "final":
            S[("yesno", r["item"])] = r["p_yes"]
    for i, it in items.items():
        S[("numF1", i)] = ref_f1(it["text"], qinfo[it["qid"]]["ref"])

    def acc(v, ptype):
        by_q = {}
        for p in pairs:
            if p["type"] != ptype:
                continue
            if v == "C":
                x = C.get((p["better"], p["worse"]))
                if x is None:
                    continue
                a = 1.0 if x > 0.5 else 0.5 if x == 0.5 else 0.0
            else:
                b, w = S.get((v, p["better"]), float("nan")), S.get((v, p["worse"]), float("nan"))
                if np.isnan(b) or np.isnan(w):
                    continue
                a = 1.0 if b > w else 0.5 if b == w else 0.0
            by_q.setdefault((p["policy"], p["qid"]), []).append(a)
        return {k: np.array(x) for k, x in by_q.items()}
    f_nan = np.mean([np.isnan(S[k]) for k in S if k[0] == "F"])
    lines = ["Third judge test (PREREGISTRATION 2026-11-20 ~15:40). Untrained 4B judge, own English solution, final "
             "solution only. Pair accuracy, ties 0.5; CI = question-clustered.",
             "Pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}" for t in ("P1", "P3", "P4", "P5"))
             + f"   (P5 labeller {meta['labeler']}: {meta['candidates']} candidates, both parsed {meta['both_parsed']},"
             f" agreed {meta['agreed']})   F unparsed score: {100 * f_nan:.1f}%", "",
             f"{'judge':8s} " + "  ".join(f"{t:>20s}" for t in ("P1", "P3", "P4 (gold numbers)", "P5 (8B label)"))
             + "   P5 ties"]
    res = {}
    for v in ("A", "C", "E", "F", "yesno", "B", "D", "numF1"):
        cells = []
        for t in ("P1", "P3", "P4", "P5"):
            a = acc(v, t)
            if not a:
                cells.append(f"{'n/a':>20s}")
                continue
            m, lo, hi = _boot_ci(list(a.values()))
            res[(v, t)] = (m, lo)
            cells.append(f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]")
        a5 = acc(v, "P5")
        ties = np.mean(np.concatenate(list(a5.values())) == 0.5) if a5 else float("nan")
        lines.append(f"{v:8s} " + "  ".join(f"{c:>20s}" for c in cells) + f"   {100 * ties:6.1f}%")
    passing = [v for v in ("A", "C", "E", "F") if (v, "P5") in res and res[(v, "P5")][0] >= 0.60
               and res[(v, "P5")][1] > 0.50 and res[(v, "P1")][0] >= 0.90]
    lines += ["", "GO RULE 3 (A/C/E/F: P5 >= 60%, CI lo > 50%, P1 >= 90%): "
              + (", ".join(f"{v} PASS (P5 {100 * res[(v, 'P5')][0]:.1f}, lo {100 * res[(v, 'P5')][1]:.1f})"
                           for v in passing) or "no judge passes")]
    txt = "\n".join(lines)
    (OUT / "report3.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "logs" / f"{cmd}.log")
    {"sample": sample, "score": judge_score, "report": report, "score2": judge_score2, "report2": report2,
     "label3": label3, "score3": score3, "report3": report3}[cmd]()
