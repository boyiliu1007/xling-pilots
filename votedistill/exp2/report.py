"""Experiment 2 reports: per-level metrics, the L0 reproducibility check and gold-free per-item capability signals."""
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from ..stats import fmt, paired_bootstrap
from ..utils import read_json, read_jsonl
from ..copsd.evaluate import summarize
from ..copsd.report import _line
from .levels import LEVELS, REFERENCE_BASE, run_dir

SPLITS = ("dev", "test")


def is_repetitive(text, n=8, min_repeats=5):
    """Degenerate looping: some n-word window occurs at least min_repeats times in the sample."""
    w = text.split()
    if len(w) < n * min_repeats:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(1)[0][1] >= min_repeats


def majority_share(rec):
    """Gold-free agreement: share of all n samples that give the most common formatted answer (0 if none formatted)."""
    fp = [p for p, f in zip(rec["preds"], rec["formatted"]) if f]
    return Counter(fp).most_common(1)[0][1] / len(rec["preds"]) if fp else 0.0


def _load(level, split, name="base"):
    p = run_dir(level) / "eval" / f"{name}_{split}.jsonl"
    return read_jsonl(p) if p.exists() else None


def reproducibility(ours, ref, split):
    """Item-by-item comparison of two base-model evaluations with the same protocol and item order."""
    T = [f"  {split.upper()} ({len(ours)} items)"]
    assert [r["item"] for r in ours] == [r["item"] for r in ref], "item order differs"
    T.append("  " + _line("ours", ours).strip())
    T.append("  " + _line("reference", ref).strip())
    for key, label in (("pass_at_n", "pass@n"), ("maj_correct", "maj@n"), ("mean_acc", "mean_acc")):
        d, lo, hi = paired_bootstrap([r[key] for r in ref], [r[key] for r in ours])
        T.append(f"    Δ {label:<9} (ours - reference) {d * 100:+.2f} pts [{lo * 100:+.2f}, {hi * 100:+.2f}]")
    same_out = sum(a == b for x, y in zip(ours, ref) for a, b in zip(x["outputs"], y["outputs"]))
    tot = sum(len(x["outputs"]) for x in ours)
    T.append(f"    identical generations: {same_out}/{tot}; items with the same pass@n: "
             f"{sum(x['pass_at_n'] == y['pass_at_n'] for x, y in zip(ours, ref))}/{len(ours)}, "
             f"same maj@n: {sum(x['maj_correct'] == y['maj_correct'] for x, y in zip(ours, ref))}/{len(ours)}")
    return T


def diagnostics(R):
    outs = [o for r in R for o in r["outputs"]]
    trunc = [t for r in R for t in r["truncated"]]
    fmt_ = [f for r in R for f in r["formatted"]]
    rep = [is_repetitive(o) for o in outs]
    rep_t = [x for x, t in zip(rep, trunc) if t]
    ch = np.array([len(o) for o in outs])
    no_box_done = sum(1 for t, f in zip(trunc, fmt_) if not t and not f)
    return [f"  samples {len(outs)}: truncated {np.mean(trunc):.3f}; formatted {np.mean(fmt_):.3f}; "
            f"finished without \\boxed {no_box_done / len(outs):.3f}",
            f"  repetitive (an 8-word window repeated >= 5 times): {np.mean(rep):.3f} of all samples, "
            f"{np.mean(rep_t) if rep_t else 0:.3f} of truncated ones",
            f"  output length (characters): median {np.median(ch):.0f}, p10 {np.percentile(ch, 10):.0f}, p90 {np.percentile(ch, 90):.0f}"]


def capability_table(R):
    """Gold-free capability signal (majority share) against gold correctness, for checking that the signal predicts
    accuracy before it is used to group items (proposal §6.3)."""
    bins = [("no formatted answer", lambda s: s == 0), ("share 1-3/12", lambda s: 0 < s <= 3 / 12),
            ("share 4-7/12", lambda s: 3 / 12 < s <= 7 / 12), ("share 8-12/12", lambda s: s > 7 / 12)]
    T = ["  bin                   items  maj@n                       mean_acc"]
    for name, f in bins:
        S = [r for r in R if f(majority_share(r))]
        if S:
            T.append(f"  {name:<21} {len(S):>5}  {fmt(sum(r['maj_correct'] for r in S), len(S)):<26}  "
                     f"{np.mean([r['mean_acc'] for r in S]):.3f}")
        else:
            T.append(f"  {name:<21} {0:>5}")
    return T


def build_report(level):
    rd = run_dir(level)
    T = [f"Experiment 2 report: {level}", f"generated {datetime.now().isoformat(timespec='seconds')}",
         f"teacher context: {LEVELS[level]['teacher']}"]
    ecfg = read_json(rd / "eval_config.json")
    if ecfg:
        T += [f"eval sampling: {ecfg['sampling']}", f"eval protocol: {ecfg['protocol']}"]
    meta = read_json(rd / "env.json")
    if meta:
        T.append(f"environment: {meta}")
    for split in SPLITS:
        R = _load(level, split)
        if R:
            T += [f"\n## {split.upper()} ({len(R)} items)", _line("base", R), *diagnostics(R)]
    if level == "L0":
        T.append(f"\n## Reproducibility: this run vs the earlier base evaluation ({REFERENCE_BASE})")
        for split in SPLITS:
            ours, ref = _load(level, split), (read_jsonl(REFERENCE_BASE / f"base_{split}.jsonl")
                                             if (REFERENCE_BASE / f"base_{split}.jsonl").exists() else None)
            if ours and ref:
                T += reproducibility(ours, ref, split)
        allR = [r for s in SPLITS for r in (_load(level, s) or [])]
        if allR:
            T.append(f"\n## Gold-free capability signal vs gold accuracy (DEV+TEST, {len(allR)} items)")
            T.append("  majority share = samples giving the most common formatted answer / n")
            T += capability_table(allR)
    if level == "L0-eng":
        from .capability import EN_CONFIDENT, features, table
        T.append(f"\n## Capability groups from L0-eng x L0 (gold-free; English confident = en_share > {EN_CONFIDENT})")
        for split in SPLITS:
            en, sw = _load("L0-eng", split), _load("L0", split)
            if en and sw:
                T += table(features(en, sw), split.upper())
    text = "\n".join(T) + "\n"
    (rd / "report.txt").write_text(text)
    return text
