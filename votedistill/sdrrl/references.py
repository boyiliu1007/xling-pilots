"""Read a vote file and pick one English reference trace per problem.

Two input formats are accepted, one JSON object per line:
  1. `votedistill.consensus` records (question_id, raw_outputs, preds, majority_answer, vote_share, is_tie):
     the reference is chosen here, as in Experiment 1: accept if vote share >= min_share, take the shortest
     (or longest) untruncated sample whose answer agrees with the majority.
  2. Already-selected references: question_id (or qid / id / idx), reference (or trace / pseudo_reference /
     reference_solution / solution) and answer (or majority_answer / voted_answer).

A reference is split into reasoning and final answer text: a thinking-mode sample "<think>...</think> final"
splits at </think>; a plain solution is all reasoning and gets a one-line final answer from the voted answer.
"""
import json
import re

from ..answers_numeric import last_boxed, numeric_equal

QID_KEYS = ("question_id", "qid", "id", "idx")
REF_KEYS = ("reference", "trace", "pseudo_reference", "reference_solution", "solution")
ANS_KEYS = ("answer", "majority_answer", "voted_answer")


def _first(rec, keys):
    for k in keys:
        if rec.get(k) is not None:
            return rec[k]
    return None


def _norm(s):
    return re.sub(r"\s+", "", str(s)).replace("$", "").lower()


def agrees(pred, majority):
    """Does one sample's extracted answer equal the voted answer? Numbers numerically, strings via math-verify."""
    if pred is None or majority is None:
        return False
    if isinstance(pred, (int, float)) and isinstance(majority, (int, float)):
        return numeric_equal(pred, majority)
    if _norm(pred) == _norm(majority):
        return True
    from ..answers_math import equiv, parse_answer
    a, b = str(majority), str(pred)
    return equiv(a, parse_answer(a), b, parse_answer(b)) or equiv(b, parse_answer(b), a, parse_answer(a))


def split_trace(trace, answer):
    """-> (reasoning, final answer text or None)."""
    trace = trace.strip()
    if "</think>" in trace:
        reasoning, final = trace.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1).strip()
        return reasoning, final.strip() or None
    return trace.replace("<think>", "", 1).strip(), None


def select_reference(rec, min_share=0.625, pick="shortest"):
    """Consensus record -> chosen sample text, or None if the vote is not confident enough."""
    if rec["is_tie"] or rec["majority_answer"] is None or rec["vote_share"] < min_share - 1e-9:
        return None
    finish = rec.get("finish_reasons") or ["stop"] * len(rec["raw_outputs"])
    ok = [o for o, p, f in zip(rec["raw_outputs"], rec["preds"], finish)
          if f != "length" and agrees(p, rec["majority_answer"])]
    if not ok:
        return None
    return (min if pick == "shortest" else max)(ok, key=len)


def load_references(path, min_share=0.625, pick="shortest"):
    """-> {question_id: {"reasoning", "final", "answer", "vote_share"}} and a dict of counts."""
    refs, stats = {}, {"records": 0, "accepted": 0, "rejected_low_agreement": 0, "no_boxed_answer": 0}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["records"] += 1
            qid = _first(rec, QID_KEYS)
            if qid is None:
                raise SystemExit(f"{path}: no question id; expected one of {QID_KEYS}")
            if "raw_outputs" in rec:
                trace = select_reference(rec, min_share, pick)
            else:
                trace = _first(rec, REF_KEYS)
            if not trace:
                stats["rejected_low_agreement"] += 1
                continue
            # prefer the answer exactly as the trace wrote it (keeps the model's own LaTeX)
            answer = last_boxed(trace) or _first(rec, ANS_KEYS)
            if answer is None:
                stats["no_boxed_answer"] += 1
                continue
            reasoning, final = split_trace(trace, answer)
            refs[int(qid)] = {"reasoning": reasoning, "final": final, "answer": str(answer),
                              "vote_share": rec.get("vote_share")}
            stats["accepted"] += 1
    return refs, stats
