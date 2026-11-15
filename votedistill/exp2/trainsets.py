"""Experiment 2 training sets and label-free (voted) references.

Datasets (the student always sees the Swahili question):
  copsd     the released COPSD training set (500 OpenThoughts olympiad problems, COPSD's Swahili translation)
  gsm8k-mt  a fixed 500-question GSM8K *train* sample (data.gsm8k_items, seed 42), machine-translated to Swahili
            (votedistill/exp2/translate.py; questions whose numbers do not survive translation are dropped)
  mix       half of each

Voted reference (contexts *-vote): from an English thinking-mode consensus run (votedistill.consensus, n = 8) on
the same questions. A question is accepted when its majority is unique and has vote share >= VOTE_THRESHOLD; its
reference is the final solution (text after </think>) of the first finished sample that agrees with the majority.
Rejected questions are dropped. So that the datasets differ only in their problems, every dataset of a vote
context is subsampled to the same size N = min(accepted copsd, accepted gsm8k-mt); mix = N/2 + N/2.
"""
import hashlib
import random
import re

from ..answers_math import equiv, extract_pred, final_text, parse_answer
from ..answers_numeric import extract_answer, numeric_equal
from ..consensus import eligible
from ..data import load_copsd_train
from ..utils import read_jsonl
from ..copsd.collator import prepare_rows
from ..paths import RESULTS
from .levels import EXP2_RESULTS, TEACHER_CONTEXTS

DATASETS = ("copsd", "gsm8k-mt", "mix", "gsm8k-mt3k")
MT_FILE = RESULTS / "exp2" / "data" / "gsm8k_swa_mt.jsonl"   # NLLB translation: shared by every model
CONSENSUS = EXP2_RESULTS / "consensus"
# gsm8k-mt3k: a 3000-question GSM8K train sample (same seed, NLLB, same QC), used by the large self-SFT ablation only
MT_FILES = {"gsm8k-mt": MT_FILE, "gsm8k-mt3k": RESULTS / "exp2" / "data" / "gsm8k_swa_mt3k.jsonl"}
CONSENSUS_FILE = {"copsd": "copsd_en_thinking.jsonl", "gsm8k-mt": "gsm8k_en_thinking.jsonl",
                  "gsm8k-mt3k": "gsm8k3k_en_thinking.jsonl"}
VOTE_THRESHOLD = 5 / 8
SUBSAMPLE_SEED = 42


def gsm8k_reference(answer):
    """GSM8K's gold rationale as a reference: calculator annotations removed, '#### x' -> a boxed final answer."""
    body, final = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe final answer is \\boxed{{{final.strip().replace(',', '')}}}."


def _copsd_rows():
    rows = prepare_rows(load_copsd_train("swa"), "swa")
    return [{**r, "qid": i, "dataset": "copsd"} for i, r in enumerate(rows)]


def _gsm8k_rows(dataset="gsm8k-mt"):
    f = MT_FILES[dataset]
    if not f.exists():
        raise SystemExit(f"{f} missing: run `python -m votedistill.exp2 translate` first")
    out = []
    for m in read_jsonl(f):
        if not m["qc_ok"]:
            continue
        r = {"problem": m["question_en"], "problem_translated": m["question_sw"], "Answer": m["gold"],
             "solution": gsm8k_reference(m["answer_en"]), "source": "gsm8k_train", "language": "swa"}
        out.append({**prepare_rows([r], "swa")[0], "qid": m["qid"], "dataset": dataset})
    return out


def _agrees(rec, text, mode):
    if mode == "numeric":
        return numeric_equal(extract_answer(final_text(text, True) or ""), rec["majority_answer"])
    s, p = extract_pred(text, thinking=True)
    m = rec["majority_answer"]
    return equiv(m, parse_answer(m), s, p)


def voted_references(dataset, threshold=VOTE_THRESHOLD):
    """{qid: (reference_text, majority_answer, vote_share)} for the accepted questions of one source dataset."""
    p = CONSENSUS / CONSENSUS_FILE[dataset]
    if not p.exists():
        raise SystemExit(f"{p} missing: run the English thinking-mode consensus first")
    mode = "numeric" if dataset.startswith("gsm8k") else "math"
    out = {}
    for rec in read_jsonl(p):
        if not eligible(rec, threshold):
            continue
        for text, fin in zip(rec["raw_outputs"], rec["finish_reasons"]):
            if fin == "stop" and "</think>" in text and _agrees(rec, text, mode):
                ref = final_text(text, True).strip()
                if ref:
                    out[rec["question_id"]] = (ref, str(rec["majority_answer"]), rec["vote_share"])
                    break
    return out


def _subsample(rows, n, seed=SUBSAMPLE_SEED):
    if n >= len(rows):
        return list(rows)
    keep = set(random.Random(seed).sample(range(len(rows)), n))
    return [r for i, r in enumerate(rows) if i in keep]


def source_rows(dataset, uses_vote, threshold=VOTE_THRESHOLD):
    """Rows of copsd / gsm8k-mt; with uses_vote only accepted questions, carrying vote_solution / vote_answer."""
    rows = _copsd_rows() if dataset == "copsd" else _gsm8k_rows(dataset)
    if not uses_vote:
        return rows
    refs = voted_references(dataset, threshold)
    return [{**r, "vote_solution": refs[r["qid"]][0], "vote_answer": refs[r["qid"]][1], "vote_share": refs[r["qid"]][2]}
            for r in rows if r["qid"] in refs]


def training_rows(dataset, context, threshold=VOTE_THRESHOLD):
    """Training rows (before teacher_rows) and a description for config.json."""
    ref = TEACHER_CONTEXTS[context]["reference"] or ""
    uses_vote = "vote" in ref
    if uses_vote:   # equal-size datasets for every vote context
        # datasets with a consensus run (Qwen3-1.7B: both; Qwen3-4B: COPSD only, so no equalisation there)
        have = [d for d in ("copsd", "gsm8k-mt") if (CONSENSUS / CONSENSUS_FILE[d]).exists()]
        pools = {d: source_rows(d, True, threshold) for d in have or [dataset]}
        n = min(len(v) for v in pools.values())
        sizes = {d: len(v) for d, v in pools.items()}
    else:
        pools = {d: source_rows(d, False) for d in (("copsd", "gsm8k-mt") if dataset == "mix" else (dataset,))}
        n = min(len(v) for v in pools.values())
        sizes = {d: len(v) for d, v in pools.items()}
    if dataset == "mix":
        rows = _subsample(pools["copsd"], n // 2, SUBSAMPLE_SEED) + _subsample(pools["gsm8k-mt"], n - n // 2, SUBSAMPLE_SEED + 1)
    else:
        rows = _subsample(pools[dataset], n, SUBSAMPLE_SEED)
    desc = {"dataset": dataset, "rows": len(rows), "available_rows": sizes, "equalized_size": n,
            "vote_threshold": threshold if uses_vote else None, "subsample_seed": SUBSAMPLE_SEED,
            "qids_sha1": hashlib.sha1(",".join(f"{r['dataset']}:{r['qid']}" for r in rows).encode()).hexdigest()}
    return rows, desc
