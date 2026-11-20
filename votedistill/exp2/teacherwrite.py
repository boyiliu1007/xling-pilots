"""Teacher-written self-SFT: sequence-level self-distillation (PREREGISTRATION 2026-11-16 entry).

Jeff's penalty-decoded self-SFT (selfsft.py) samples the base model from the *student's* prompt (Swahili question
only), so it can only learn from questions the model already solves in Swahili (1.7B, GSM8K-MT 3k: 751 kept answers
on 498 of 2805 questions). Here the same base model *generates* from a *teacher* context (Swahili question + English
question [+ reference], teacher_rows / the COPSD collator), with the student pool's sampling (T 1.1, top-p 0.95,
top-k 20, 2048 tokens, vLLM repetition penalty 1.1, n = GEN_N, same seed rule). A sample is kept when it finished,
has a final answer after </think>, is in Swahili and its answer equals the arm's target; at most PER_Q per question;
the kept set is subsampled (FILTER_SEED) to N_TARGET, the size of Jeff's student-written arms, so arms differ in
where the text comes from, not in how much. Each kept record carries the *student* prompt ids: the student is
fine-tuned (selfsft.train_sft, unchanged) to produce the teacher's answer from the Swahili question alone.

Arms (runs.TW_ARMS): twvote = L3-vote context (the model's own English majority-vote solution), filter = vote
(gold-free; the method); twl1 = L1 context (English question only), filter = vote (gold-free); twgold = L3-gold
context, filter = gold (oracle). Questions: the vote-accepted ones (source_rows(data, True)) for every arm.
"""
import fcntl
import json
import logging
import random
import time
from collections import Counter, defaultdict
from datetime import datetime

from ..answers_math import equiv, extract_pred, parse_answer
from ..copsd.collator import make_collator
from ..copsd.config import OFFICIAL
from ..paths import MODEL
from ..utils import read_jsonl, versions, write_json, write_jsonl
from .levels import teacher_rows
from .screen import _prompt_ids
from .selfsft import FILTER_SEED, GEN_N, GEN_SEED, PER_Q, SFT_DIR
from .trainsets import source_rows

log = logging.getLogger("votedistill.exp2.teacherwrite")
REP_PEN = 1.1
N_TARGET = {"gsm8k-mt3k": 751}    # Jeff's equalised student-written arm size (gen_gsm8k-mt3k_rp1.1.filters.json)


def pool_file(data, context):
    return SFT_DIR / f"tgen_{data}_{context}_rp{REP_PEN}.jsonl"


def generate(data, context):
    """All samples of one teacher context (shared by every arm and seed that uses it); generated once."""
    from ..copsd.evaluate import language_of
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    out = pool_file(data, context)
    with open(SFT_DIR / f".lock_{out.stem}", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if out.exists():
            return read_jsonl(out)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        rows = source_rows(data, True)
        trows, flags = teacher_rows(rows, context)
        tok = AutoTokenizer.from_pretrained(MODEL)
        col_t = make_collator(tok, OFFICIAL["max_length"], **flags)
        col_s = make_collator(tok, OFFICIAL["max_length"], False, False)
        tp = [_prompt_ids(col_t, r, "teacher") for r in trows]
        sp = [_prompt_ids(col_s, r, "student") for r in rows]
        llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=8192, seed=GEN_SEED)
        sps = [SamplingParams(n=GEN_N, temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"],
                              top_k=OFFICIAL["top_k"], min_p=0.0, max_tokens=OFFICIAL["max_completion_length"],
                              repetition_penalty=REP_PEN, seed=GEN_SEED * 1_000_003 + i) for i in range(len(rows))]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in tp], sps, use_tqdm=False)
        recs = []
        for r, s_ids, o in zip(rows, sp, outs):
            gold = str(r["Answer"])
            for j, c in enumerate(o.outputs):
                s, pp = extract_pred(c.text, thinking=True)
                recs.append({"dataset": r["dataset"], "qid": r["qid"], "sample": j, "teacher_context": context,
                             "student_prompt_ids": s_ids, "completion_ids": list(c.token_ids), "text": c.text,
                             "finish_reason": c.finish_reason, "pred": s, "lang": language_of(c.text),
                             "vote_answer": r["vote_answer"], "gold": gold,
                             "correct": bool(s is not None and equiv(gold, parse_answer(gold), s, pp))})
        write_jsonl(out, recs)
        fin = [r for r in recs if r["finish_reason"] == "stop"]
        write_json(out.with_suffix(".meta.json"), {
            "doc": __doc__, "context": context, "questions": len(rows), "gen_n": GEN_N, "seed": GEN_SEED,
            "repetition_penalty": REP_PEN, "samples": len(recs), "finished": len(fin),
            "acc_vs_gold": sum(r["correct"] for r in recs) / len(recs),
            "acc_vs_gold_finished": sum(r["correct"] for r in fin) / max(len(fin), 1),
            "lang": dict(Counter(r["lang"] for r in recs)), "generation_s": round(time.time() - t0),
            "versions": versions(), "created": datetime.now().isoformat(timespec="seconds")})
        log.info(f"teacher pool {context}: {len(recs)} samples, acc {sum(r['correct'] for r in recs) / len(recs):.3f}, "
                 f"finished {len(fin) / len(recs):.3f} in {time.time() - t0:.0f}s")
        return recs


def kept(data, arm):
    """Kept samples of one arm, subsampled to N_TARGET[data]; the selection is cached next to the pool."""
    from .runs import TW_ARMS
    context, filt = TW_ARMS[arm]
    recs = generate(data, context)
    cache = pool_file(data, context).with_suffix(f".{arm}.kept.json")
    key = {(r["qid"], r["sample"]): r for r in recs}
    if not cache.exists():
        by_q = defaultdict(list)
        for r in recs:
            if r["finish_reason"] == "stop" and r["pred"] is not None and r["lang"] == "sw":
                by_q[r["qid"]].append(r)
        out = []
        for ss in by_q.values():
            t = ss[0]["vote_answer"] if filt == "vote" else ss[0]["gold"]
            out += [r for r in ss if equiv(t, parse_answer(t), r["pred"], parse_answer(r["pred"]))][:PER_Q]
        n = min(N_TARGET[data], len(out))
        idx = sorted(random.Random(FILTER_SEED).sample(range(len(out)), n))
        sel = [out[i] for i in idx]
        write_json(cache, {"arm": arm, "context": context, "filter": filt, "before_subsample": len(out),
                           "questions_before": len({r["qid"] for r in out}), "size": n,
                           "questions": len({r["qid"] for r in sel}),
                           "kept_correct_vs_gold": sum(r["correct"] for r in sel) / max(n, 1),
                           "seed": FILTER_SEED, "kept": [[r["qid"], r["sample"]] for r in sel]})
    sel = json.loads(cache.read_text())
    log.info(f"{arm}: {sel['size']} kept answers on {sel['questions']} questions (before subsampling "
             f"{sel['before_subsample']}), correct vs gold {sel['kept_correct_vs_gold']:.3f}")
    return [key[tuple(k)] for k in sel["kept"]]
