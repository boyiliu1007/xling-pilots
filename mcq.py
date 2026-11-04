"""Pilot C pure logic: cyclic option shifts, mapping back, debiasing, voting. No GPU deps."""
from collections import Counter

import numpy as np

K = 4
LETTERS = "ABCD"


def shift_order(s, k=K):
    """Original option index shown at each position under cyclic shift s: position p shows option (p+s)%k."""
    return [(p + s) % k for p in range(k)]


def position_of(orig, s, k=K):
    """Position (letter index) where original option `orig` appears under shift s."""
    return (orig - s) % k


def map_back(pos_probs, s, k=K):
    """Probabilities indexed by displayed position -> indexed by original option."""
    out = [0.0] * k
    for p, o in enumerate(shift_order(s, k)):
        out[o] = float(pos_probs[p])
    return out


def renorm(letter_probs):
    """Renormalize raw letter probabilities; returns (probs, letter_mass)."""
    m = float(sum(letter_probs))
    if m <= 0:
        return [1.0 / len(letter_probs)] * len(letter_probs), 0.0
    return [float(x) / m for x in letter_probs], m


def argmax(v):
    """First index of the max (deterministic). Exact float ties are vanishingly rare with real probabilities."""
    return int(np.argmax(v))


def vote(answers):
    """Majority of mapped argmaxes; None if the top count is shared."""
    c = Counter(answers).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def item_record(mapped_probs, positions_chosen, gold):
    """mapped_probs: list over shifts of 4-vectors in original-option order."""
    am = [argmax(p) for p in mapped_probs]
    avg = np.mean(np.array(mapped_probs), axis=0)
    deb = argmax(avg)
    v = vote(am)
    return {
        "gold": gold,
        "probs_mapped": mapped_probs,
        "argmax_mapped": am,
        "chosen_position": positions_chosen,
        "single_order_correct": am[0] == gold,
        "mean_shift_accuracy": sum(a == gold for a in am) / len(am),
        "debiased_probs": avg.tolist(),
        "debiased_answer": deb,
        "debiased_correct": deb == gold,
        "vote_answer": v,
        "vote_correct": v is not None and v == gold,
        "order_sensitive": len(set(am)) > 1,
    }
