"""Gold-free per-item capability signals from the base model (proposal §3.2, §6.3), for grouping evaluation items.

Inputs: L0 (Swahili) and L0-eng (English) evaluations of the same AfriMGSM items, n samples each. Only answers in
\\boxed{} count (the evaluation protocol's own rule); answers are compared numerically.
  en_share   share of the n English samples that give the English majority answer (0 if no majority / tie)
  sw_agree   share of the n Swahili samples whose answer equals the English majority answer
  sw_share   share of the n Swahili samples that give the Swahili majority answer
  cross      the Swahili majority answer equals the English majority answer
Gold answers are used only to check that the signals predict accuracy, never to form the groups.
"""
import numpy as np

from ..answers_numeric import majority_vote, numeric_equal, parse_number
from ..stats import fmt

# Group rule, fixed before any trained level is evaluated. Thresholds are design choices, not tuned on outcomes:
# English "confident" = a strict majority of all n samples agree; Swahili "transfers" = its majority matches English.
EN_CONFIDENT = 0.5
GROUPS = {
    "A: English solves, Swahili does not": lambda f: f["en_share"] > EN_CONFIDENT and not f["cross"],
    "B: English solves, Swahili agrees": lambda f: f["en_share"] > EN_CONFIDENT and f["cross"],
    "C: English not confident": lambda f: f["en_share"] <= EN_CONFIDENT,
}


def _answers(rec):
    return [parse_number(p) if f else None for p, f in zip(rec["preds"], rec["formatted"])]


def item_features(en_rec, sw_rec):
    assert en_rec["item"] == sw_rec["item"]
    ea, sa = _answers(en_rec), _answers(sw_rec)
    n = len(ea)
    en_maj, _, en_share, en_tie = majority_vote(ea, n)
    sw_maj, _, sw_share, sw_tie = majority_vote(sa, n)
    if en_tie or en_maj is None:
        en_maj, en_share = None, 0.0
    gold = parse_number(str(en_rec["gold"]))
    return {"item": en_rec["item"], "en_share": en_share, "en_maj": en_maj,
            "sw_agree": sum(numeric_equal(a, en_maj) for a in sa) / n if en_maj is not None else 0.0,
            "sw_share": 0.0 if sw_tie or sw_maj is None else sw_share,
            "cross": en_maj is not None and not sw_tie and numeric_equal(sw_maj, en_maj),
            # gold-checked, for validation only
            "en_maj_correct": numeric_equal(en_maj, gold), "sw_maj_correct": bool(sw_rec["maj_correct"]),
            "sw_mean_acc": sw_rec["mean_acc"]}


def features(en_recs, sw_recs):
    assert [r["item"] for r in en_recs] == [r["item"] for r in sw_recs], "item order differs"
    return [item_features(e, s) for e, s in zip(en_recs, sw_recs)]


def group_of(f):
    hits = [g for g, rule in GROUPS.items() if rule(f)]
    assert len(hits) == 1, (f, hits)
    return hits[0]


def table(F, title):
    T = [f"  {title} ({len(F)} items)",
         f"    English majority correct (gold check of the teacher signal): {fmt(sum(f['en_maj_correct'] for f in F), len(F))}",
         f"    {'group':<38} {'items':>5}  {'English maj correct':<26} {'Swahili maj correct':<26} sw mean_acc"]
    for g in GROUPS:
        G = [f for f in F if group_of(f) == g]
        if not G:
            T.append(f"    {g:<38} {0:>5}")
            continue
        T.append(f"    {g:<38} {len(G):>5}  {fmt(sum(f['en_maj_correct'] for f in G), len(G)):<26} "
                 f"{fmt(sum(f['sw_maj_correct'] for f in G), len(G)):<26} {np.mean([f['sw_mean_acc'] for f in G]):.3f}")
    T.append("    English confidence vs English accuracy (does the label-free reliability signal work?):")
    for lo, hi in ((0, 0.5), (0.5, 0.75), (0.75, 1.01)):
        G = [f for f in F if lo <= f["en_share"] < hi]
        if G:
            T.append(f"      en_share in [{lo:.2f}, {min(hi, 1):.2f}{']' if hi > 1 else ')'}: {len(G):>4} items, "
                     f"English maj correct {fmt(sum(f['en_maj_correct'] for f in G), len(G))}")
    return T
