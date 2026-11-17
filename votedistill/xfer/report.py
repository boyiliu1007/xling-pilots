"""Report for xeval: per view, the trained checkpoint vs the base model on the same 200 TEST items.

Per view: mean_acc / maj@12 / pass@12 with paired-bootstrap deltas, format and truncation rates, and maj@12 flips
(fixed = base wrong -> trained right, broken = base right -> trained wrong).
Across views:
  comprehension gap = enen mean_acc - swen mean_acc (per item, averaged). Both views reason in English; they differ only
                      in the question's language, so a smaller gap means the model reads Swahili better.
  transfer set      = items the base model solves from the English question but not from the Swahili one
                      (base enen maj right, base swen maj wrong; gold is used for this analysis only). Fixed share there
                      is the direct test of "English understanding reaches Swahili".
"""
from pathlib import Path

import numpy as np

from ..stats import fmt, paired_bootstrap
from ..utils import read_jsonl
from .xeval import VIEWS, XEVAL, base_file, out_dir

METRICS = ("mean_acc", "maj_correct", "pass_at_n")


def _load(p):
    return {r["item"]: r for r in read_jsonl(p)} if Path(p).exists() else None


def _rates(R):
    S = sum(len(r["correct"]) for r in R.values())
    return (sum(sum(r["formatted"]) for r in R.values()) / S, sum(sum(r["truncated"]) for r in R.values()) / S)


def _view_lines(label, view, step):
    f = out_dir(label, view) / "eval" / f"step{step}_test.jsonl"
    T, B = _load(f), _load(base_file(view))
    if T is None or B is None:
        return [f"  {view:8s} (missing: {'trained' if T is None else 'base'})"], None, None
    ids = sorted(T)
    L = []
    for m in METRICS:
        b = [float(B[i][m]) for i in ids]
        t = [float(T[i][m]) for i in ids]
        d, lo, hi = paired_bootstrap(b, t)
        L.append(f"{m} {np.mean(b):.3f} -> {np.mean(t):.3f}  Δ {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    (bf, bt), (tf, tt) = _rates(B), _rates(T)
    fixed = sum(not B[i]["maj_correct"] and T[i]["maj_correct"] for i in ids)
    broken = sum(B[i]["maj_correct"] and not T[i]["maj_correct"] for i in ids)
    out = [f"  {view:8s} " + " | ".join(L),
           f"  {'':8s} format {bf:.3f} -> {tf:.3f}  truncated {bt:.3f} -> {tt:.3f}  maj flips: fixed {fixed}, "
           f"broken {broken} (base right {sum(B[i]['maj_correct'] for i in ids)})"]
    return out, B, T


def report(label, step=100):
    lines = [f"xeval report: {label} step{step} (TEST 200 items; Δ = trained - base, points, paired bootstrap 95% CI)"]
    got = {}
    for view in VIEWS:
        L, B, T = _view_lines(label, view, step)
        lines += L
        if T is not None:
            got[view] = (B, T)
        ck = out_dir(label, view) / "checkpoint.txt"
        if ck.exists() and view == "swsw":
            lines.insert(1, f"checkpoint: {ck.read_text().strip()}")
    if "swen" in got and "enen" in got:
        (Bs, Ts), (Be, Te) = got["swen"], got["enen"]
        ids = sorted(set(Ts) & set(Te))
        gb = [Be[i]["mean_acc"] - Bs[i]["mean_acc"] for i in ids]
        gt = [Te[i]["mean_acc"] - Ts[i]["mean_acc"] for i in ids]
        d, lo, hi = paired_bootstrap(gb, gt)
        lines.append(f"  comprehension gap (enen - swen mean_acc): base {np.mean(gb):.3f} -> trained {np.mean(gt):.3f}  "
                     f"Δ {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  (negative = reads Swahili better)")
        A = [i for i in ids if Be[i]["maj_correct"] and not Bs[i]["maj_correct"]]
        R = [i for i in ids if Bs[i]["maj_correct"]]
        lines.append(f"  transfer set (base: English right, Swahili->English wrong): {len(A)} items; swen fixed "
                     f"{fmt(sum(Ts[i]['maj_correct'] for i in A), len(A))}; swen broken among base-right "
                     f"{fmt(sum(not Ts[i]['maj_correct'] for i in R), len(R))}")
        if "swsw" in got:
            Bw, Tw = got["swsw"]
            lines.append(f"  same set, swsw fixed {fmt(sum(Tw[i]['maj_correct'] for i in A), len(A))}")
    return "\n".join(lines)


def report_all(step=100):
    # "base" holds the frozen model's own views (non-default BASE_MODEL), not a trained run
    labels = sorted({p.name.rsplit("__", 1)[0] for p in XEVAL.glob("*__*") if p.is_dir()} - {"base"})
    txt = "\n\n".join(report(l, step) for l in labels)
    (XEVAL / "REPORT.txt").write_text(txt + "\n")
    return txt
