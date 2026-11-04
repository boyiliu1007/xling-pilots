"""math-verify answer checking and equivalence-class voting for Pilot A2 (mirrors COPSD's grade_answer)."""
from math_verify import parse, verify

from extract import last_boxed


def final_text(text, thinking):
    """Text the final answer is read from. In thinking mode only what follows </think> counts;
    a generation truncated inside its thinking block has no final answer."""
    if text is None:
        return None
    if thinking:
        return text.split("</think>", 1)[1] if "</think>" in text else None
    return text


def _wrap(s):
    s = s.strip()
    return s if ("$" in s or "\\boxed{" in s) else f"${s}$"


def parse_answer(s):
    """Parse an answer string (e.g. boxed content or gold). Returns (parsed_list or [], error flag)."""
    if s is None or not s.strip():
        return []
    try:
        return parse(_wrap(s))
    except Exception:
        return []


def extract_pred(text, thinking=False):
    """Last \\boxed{} content of the final text, and its math-verify parse ([] if unparseable)."""
    t = final_text(text, thinking)
    b = last_boxed(t) if t is not None else None
    if b is None or not b.strip():
        return None, []
    return b.strip(), parse_answer(b)


def _norm(s):
    return s.replace("$", "").replace(" ", "").lower().strip()


def equiv(a_str, a_parsed, b_str, b_parsed):
    """Gold-first equivalence as in COPSD: verify(gold, pred); string-normalized fallback on errors."""
    if a_str is None or b_str is None:
        return False
    if a_parsed and b_parsed:
        try:
            return bool(verify(a_parsed, b_parsed, timeout_seconds=5))
        except Exception:
            pass
    return _norm(a_str) == _norm(b_str)


def is_parseable(s, p):
    return s is not None and bool(p)


def cluster_vote(pred_strs, pred_parsed, n):
    """Cluster parseable predictions into math-verify equivalence classes (a joins the first class whose
    representative is equivalent in either direction), then vote over classes.
    Returns (majority_rep_index or None, class sizes with representative strings, vote_share, is_tie)."""
    reps, sizes = [], []  # reps: index into preds of each class representative
    for i, (s, p) in enumerate(zip(pred_strs, pred_parsed)):
        if not is_parseable(s, p):
            continue
        for c, r in enumerate(reps):
            if equiv(pred_strs[r], pred_parsed[r], s, p) or equiv(s, p, pred_strs[r], pred_parsed[r]):
                sizes[c] += 1
                break
        else:
            reps.append(i)
            sizes.append(1)
    if not reps:
        return None, {}, 0.0, False
    top = max(sizes)
    leaders = [c for c, k in enumerate(sizes) if k == top]
    counts = {pred_strs[r]: k for r, k in sorted(zip(reps, sizes), key=lambda x: -x[1])}
    if len(leaders) > 1:
        return None, counts, top / n, True
    return reps[leaders[0]], counts, top / n, False
