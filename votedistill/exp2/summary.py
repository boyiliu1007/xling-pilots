"""Experiment 2 cross-run summary (results/exp2/SUMMARY.txt).

For every run with a TEST evaluation of its DEV-selected checkpoint: TEST metrics and the paired-bootstrap gain over
L0 (base model, same items, same protocol). Then the planned paired contrasts between runs, and the gains split by
the candidate capability axis (number of the 12 L0 samples that finish; gold-free).
"""
from datetime import datetime

import numpy as np

from ..copsd.evaluate import select_checkpoint
from ..stats import paired_bootstrap
from ..utils import read_json, read_jsonl
from .levels import run_dir as level_dir
from .runs import RUNS, SELECT_METRIC

METRICS = (("maj_correct", "maj@12"), ("pass_at_n", "pass@12"), ("mean_acc", "mean_acc"))
# (label, run A, run B): B - A on TEST. Seeds are matched by the suffix when both runs exist.
CONTRASTS = [
    ("layout: exact - official (gold COPSD)", "L3-gold__copsd__official", "L3-gold__copsd__exact"),
    ("English question: L1 - L0", None, "L1__copsd__exact"),
    ("answer only: L2-gold - L1", "L1__copsd__exact", "L2-gold__copsd__exact"),
    ("full trace: L3-gold - L1", "L1__copsd__exact", "L3-gold__copsd__exact"),
    ("trace vs answer: L3-gold - L2-gold", "L2-gold__copsd__exact", "L3-gold__copsd__exact"),
    ("trace content: L3-gold - L3-shuffled-gold", "L3-shuffled-gold__copsd__exact", "L3-gold__copsd__exact"),
    ("placebo vs question: L3-shuffled-gold - L1", "L1__copsd__exact", "L3-shuffled-gold__copsd__exact"),
    ("data (gold): gsm8k-mt - copsd", "L3-gold__copsd__exact", "L3-gold__gsm8k-mt__exact"),
    ("data (gold): mix - copsd", "L3-gold__copsd__exact", "L3-gold__mix__exact"),
    ("data (vote): gsm8k-mt - copsd", "L3-vote__copsd__exact", "L3-vote__gsm8k-mt__exact"),
    ("data (vote): mix - copsd", "L3-vote__copsd__exact", "L3-vote__mix__exact"),
    ("vote vs gold (copsd): L3-vote - L3-gold", "L3-gold__copsd__exact", "L3-vote__copsd__exact"),
    ("vote vs gold (gsm8k-mt): L3-vote - L3-gold", "L3-gold__gsm8k-mt__exact", "L3-vote__gsm8k-mt__exact"),
    # loss-clip ablation (official clip 0.05 vs none / 0.5): does the information level matter without the clip?
    ("no clip: L3-gold - L1", "L1__copsd__exact__noclip", "L3-gold__copsd__exact__noclip"),
    ("no clip: L3-gold - L3-shuffled-gold", "L3-shuffled-gold__copsd__exact__noclip", "L3-gold__copsd__exact__noclip"),
    ("clip 0.5: L3-gold - L1", "L1__copsd__exact__clip0.5", "L3-gold__copsd__exact__clip0.5"),
    ("clip effect (L1): none - 0.05", "L1__copsd__exact", "L1__copsd__exact__noclip"),
    ("clip effect (L3-gold): none - 0.05", "L3-gold__copsd__exact", "L3-gold__copsd__exact__noclip"),
    ("clip effect (L1): 0.5 - 0.05", "L1__copsd__exact", "L1__copsd__exact__clip0.5"),
    ("clip effect (L3-gold): 0.5 - 0.05", "L3-gold__copsd__exact", "L3-gold__copsd__exact__clip0.5"),
]
FINISH_BINS = ((0, 0), (1, 2), (3, 5), (6, 12))


def _base(split="test"):
    return read_jsonl(level_dir("L0") / "eval" / f"base_{split}.jsonl")


def final_test(name, step=100):
    p = RUNS / name / "eval" / f"step{step}_test.jsonl"
    return read_jsonl(p) if p.exists() else None


def selected_test(name):
    """(selected checkpoint, TEST records) or (None, None)."""
    rd = RUNS / name
    steps = [f"step{s}" for s in range(10, 101, 10)]
    sel = select_checkpoint(rd, SELECT_METRIC, steps) if (rd / "eval").exists() else None
    p = rd / "eval" / f"{sel}_test.jsonl"
    return (sel, read_jsonl(p)) if sel and p.exists() else (sel, None)


def _gain(A, B, key):
    assert [r["item"] for r in A] == [r["item"] for r in B], "item order differs"
    return paired_bootstrap([r[key] for r in A], [r[key] for r in B])


def _fmt(d, lo, hi):
    return f"{d * 100:+5.1f} [{lo * 100:+5.1f}, {hi * 100:+5.1f}]"


def build_summary():
    base = _base()
    T = [f"Experiment 2 summary, generated {datetime.now().isoformat(timespec='seconds')}",
         "TEST 200 items; checkpoint selected by DEV maj@12 among steps 10..100; gains in points with 95% paired-bootstrap CI",
         "", f"{'run':<42} {'sel':>7} {'maj@12':>7} {'pass@12':>8} {'mean':>6} {'trunc':>6}   "
             f"{'Δmaj@12 vs L0':<22} {'Δpass@12 vs L0':<22} {'Δmean_acc vs L0':<22}"]
    T.append(f"{'L0 (base)':<42} {'-':>7} {np.mean([r['maj_correct'] for r in base]):7.3f} "
             f"{np.mean([r['pass_at_n'] for r in base]):8.3f} {np.mean([r['mean_acc'] for r in base]):6.3f} "
             f"{np.mean([np.mean(r['truncated']) for r in base]):6.3f}")
    res = {}
    for rd in sorted(RUNS.glob("*__*")):
        sel, R = selected_test(rd.name)
        cfg = read_json(rd / "config.json", {})
        if R is None:
            T.append(f"{rd.name:<42} {sel or '-':>7}   (no TEST yet)")
            continue
        res[rd.name] = R
        T.append(f"{rd.name:<42} {sel:>7} {np.mean([r['maj_correct'] for r in R]):7.3f} "
                 f"{np.mean([r['pass_at_n'] for r in R]):8.3f} {np.mean([r['mean_acc'] for r in R]):6.3f} "
                 f"{np.mean([np.mean(r['truncated']) for r in R]):6.3f}   "
                 + " ".join(f"{_fmt(*_gain(base, R, k)):<22}" for k, _ in METRICS)
                 + f"  rows {cfg.get('data', {}).get('rows', '?')}")

    T += ["", "## Selection-free: TEST at the final checkpoint (step 100)"]
    fres = {}
    for rd in sorted(RUNS.glob("*__*")):
        R = final_test(rd.name)
        if R is not None:
            fres[rd.name] = R
            T.append(f"{rd.name:<42} {'step100':>7} {np.mean([r['maj_correct'] for r in R]):7.3f} "
                     f"{np.mean([r['pass_at_n'] for r in R]):8.3f} {np.mean([r['mean_acc'] for r in R]):6.3f} "
                     f"{np.mean([np.mean(r['truncated']) for r in R]):6.3f}   "
                     + " ".join(f"{_fmt(*_gain(base, R, k)):<22}" for k, _ in METRICS))

    T += ["", "## Planned contrasts (B - A on TEST, same seed)"]
    for tag, pool in (("DEV-selected", res), ("step100", fres)):
        T.append(f"  [{tag}]")
        seeds = sorted({n.rsplit("__", 1)[1] for n in pool})
        for label, a, b in CONTRASTS:
            for s in seeds:
                A = base if a is None else pool.get(f"{a}__{s}")
                B = pool.get(f"{b}__{s}")
                if A is None or B is None:
                    continue
                T.append(f"  {label:<46} {s:<4} " + "  ".join(f"{lab} {_fmt(*_gain(A, B, k))}" for k, lab in METRICS))

    T += ["", "## Mechanism: does training stop the looping, or also make finished answers more accurate?",
          f"  {'run':<42} {'trunc':>6} {'loop':>6} {'acc|finished':>13} {'finished/item':>14} {'sw-lang':>8}"]
    from .report import is_repetitive

    def mech(R):
        tr = [t for r in R for t in r["truncated"]]
        cf = [c for r in R for c, t in zip(r["correct"], r["truncated"]) if not t]
        lp = [is_repetitive(o) for r in R for o in r["outputs"]]
        sw = [l == "sw" for r in R for l in r["language"]]
        return (f"{np.mean(tr):6.3f} {np.mean(lp):6.3f} {np.mean(cf) if cf else float('nan'):13.3f} "
                f"{12 * (1 - np.mean(tr)):14.2f} {np.mean(sw):8.3f}")
    T.append(f"  {'L0 (base)':<42} " + mech(base))
    for name, R in res.items():
        T.append(f"  {name + ' (sel)':<42} " + mech(R))
    for name, R in fres.items():
        T.append(f"  {name + ' (step100)':<42} " + mech(R))

    T += ["", "## Gain over L0 by capability bin (number of the 12 L0 samples that finish; gold-free)"]
    fin = np.array([12 - sum(r["truncated"]) for r in base])
    bins = [(lo, hi, np.where((fin >= lo) & (fin <= hi))[0]) for lo, hi in FINISH_BINS]
    T.append(f"  {'run':<42} " + " ".join(f"{f'finish {lo}-{hi} (n={len(ix)})':<26}" for lo, hi, ix in bins))
    T.append(f"  {'L0 maj@12 / mean_acc':<42} " + " ".join(
        f"{np.mean([base[i]['maj_correct'] for i in ix]):.3f} / {np.mean([base[i]['mean_acc'] for i in ix]):.3f}{'':<11}"
        for _, _, ix in bins))
    for name, R in res.items():
        cells = []
        for _, _, ix in bins:
            d, lo, hi = paired_bootstrap([base[i]["mean_acc"] for i in ix], [R[i]["mean_acc"] for i in ix])
            cells.append(f"{_fmt(d, lo, hi):<26}")
        T.append(f"  {name:<42} " + " ".join(cells))
    T.append("  (cells: Δ mean_acc in points; maj@12 is too coarse within bins)")
    text = "\n".join(T) + "\n"
    (RUNS.parent / "SUMMARY.txt").write_text(text)
    return text
