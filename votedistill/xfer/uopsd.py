"""U-OPSD baseline on English questions (proposal 5.3), built on the shared COPSD trainer (copsd/train.py).

Run name: <method>__<data>__<clip>__s<seed>
  method  uopsd      longest agreeing reference, loss only on rollouts that disagree with the vote (U-OPSD as published)
          uopsd-all  same teacher, loss on every rollout (isolates the disagree-only filter)
  data    gsm8k-mt   English side of Experiment 2's GSM8K train sample (same questions as L3-*__gsm8k-mt)
          copsd      English side of the COPSD olympiad set
  clip    clip       official per-(token, vocab) clip 0.05
          noclip     no clip (does the clip matter when the student does not loop? English rollouts rarely do)
  seed    s42, s43, ...

Everything else is Experiment 2's COPSD setup (exact layout, official hyperparameters, 100 steps, thinking mode).
The vote is the English thinking-mode consensus Experiment 2 already ran (results/exp2/consensus, n = 8).
"""
import hashlib
import logging

from ..answers_math import extract_pred, final_text
from ..answers_numeric import extract_answer, numeric_equal
from ..consensus import eligible
from ..copsd.config import OFFICIAL
from ..paths import CKPT_ROOT, RESULTS
from ..utils import read_jsonl, write_jsonl
from ..exp2.trainsets import CONSENSUS, CONSENSUS_FILE, _agrees, _copsd_rows, _gsm8k_rows

log = logging.getLogger("votedistill.xfer.uopsd")
XFER = RESULTS / "xfer"
RUNS = XFER / "runs"
CKPT = CKPT_ROOT / "xfer"
THRESHOLD = 0.5            # U-OPSD: discard questions with < 50% agreement
METHODS = {"uopsd": True, "uopsd-all": False}   # -> train only on disagreeing rollouts?
CLIPS = {"clip": OFFICIAL["jsd_token_clip"], "noclip": None}
MAX_STEPS = 100


def parse(name):
    parts = name.split("__")
    if len(parts) != 4 or parts[0] not in METHODS or parts[1] not in ("gsm8k-mt", "copsd") or parts[2] not in CLIPS \
            or not parts[3].startswith("s"):
        raise SystemExit(f"bad U-OPSD run name {name}: <uopsd|uopsd-all>__<gsm8k-mt|copsd>__<clip|noclip>__s<seed>")
    return parts[0], parts[1], CLIPS[parts[2]], int(parts[3][1:])


def paths(name):
    return RUNS / name, CKPT / name


def _mode(dataset):
    return "numeric" if dataset.startswith("gsm8k") else "math"


def voted_longest(dataset, threshold=THRESHOLD):
    """{qid: (longest agreeing final solution, majority answer, vote share)}; length = characters after </think>."""
    out = {}
    for rec in read_jsonl(CONSENSUS / CONSENSUS_FILE[dataset]):
        if not eligible(rec, threshold):
            continue
        best = None
        for text, fin in zip(rec["raw_outputs"], rec["finish_reasons"]):
            if fin == "stop" and "</think>" in text and _agrees(rec, text, _mode(dataset)):
                ref = final_text(text, True).strip()
                if ref and (best is None or len(ref) > len(best)):
                    best = ref
        if best:
            out[rec["question_id"]] = (best, rec["majority_answer"], rec["vote_share"])
    return out


def training_rows(dataset):
    """English-student rows for the official collator: target_lang EN, solution = longest voted reference."""
    src = _copsd_rows() if dataset == "copsd" else _gsm8k_rows(dataset)
    refs = voted_longest(dataset)
    rows = []
    for r in src:
        if r["qid"] not in refs:
            continue
        ref, maj, share = refs[r["qid"]]
        rows.append({"qid": r["qid"], "dataset": dataset, "problem": r["problem_en"], "problem_en": r["problem_en"],
                     "target_lang": "EN", "solution": ref, "vote_answer": maj, "vote_share": share,
                     "gold": r.get("Answer")})
    desc = {"dataset": dataset, "student": "English question", "rows": len(rows), "available": len(src),
            "vote_threshold": THRESHOLD, "reference": "longest finished English rollout agreeing with the majority",
            "qids_sha1": hashlib.sha1(",".join(str(r["qid"]) for r in rows).encode()).hexdigest()}
    return rows, desc


def disagrees(dataset):
    """keep_fn: True when the rollout's final answer is missing or differs from the English majority vote."""
    mode = _mode(dataset)

    def keep(row, text, finish_reason):
        if mode == "numeric":
            pred = extract_answer(final_text(text, True) or "")
            return pred is None or not numeric_equal(pred, row["vote_answer"])
        s, p = extract_pred(text, thinking=True)
        if s is None:
            return True
        return not _agrees({"majority_answer": row["vote_answer"]}, text, mode)
    return keep


def train_uopsd(name, max_steps=MAX_STEPS):
    from ..copsd.styles import apply_style
    from ..copsd.train import train
    method, dataset, clip, seed = parse(name)
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    apply_style("copsd")   # English prompts come from COPSD's own EN language config
    rows, desc = training_rows(dataset)
    vote_gold = [numeric_equal(r["vote_answer"], r["gold"]) for r in rows] if _mode(dataset) == "numeric" else None
    if vote_gold:
        desc["vote_correct_vs_gold (analysis only)"] = f"{sum(vote_gold)}/{len(vote_gold)}"
    desc["train_on"] = "rollouts disagreeing with the vote" if METHODS[method] else "all rollouts"
    write_jsonl(rd / "train_rows.jsonl", [{k: r.get(k) for k in ("qid", "problem_en", "solution", "vote_answer",
                                                                 "vote_share")} for r in rows])
    log.info(f"{name}: {desc}")
    # include_problem_en=False: the student's question already is the English one (no duplicate in the teacher prompt)
    train(rd, ck, "en", max_steps, include_problem_en=False, include_reference_solution_en=True, seed=seed,
          layout="exact", rows=rows, data_desc=desc, reference_desc="U-OPSD: longest voted English rollout",
          clip=clip, keep_fn=disagrees(dataset) if METHODS[method] else None)
