"""Text report for one COPSD run directory: setup, training curve, DEV curve, and TEST base vs DEV-selected checkpoint."""
from datetime import datetime
from pathlib import Path

import numpy as np

from ..stats import fmt, paired_bootstrap
from ..utils import read_json
from .evaluate import evaluated, select_checkpoint, summarize

METRICS = {"pass_at_n": "pass@n", "maj_correct": "maj@n", "mean_acc": "mean_acc"}


def _line(name, R):
    s = summarize(R)
    n = len(R)
    return (f"  {name:<8} pass@n {fmt(sum(r['pass_at_n'] for r in R), n):<26} maj@n {fmt(sum(r['maj_correct'] for r in R), n):<26} "
            f"mean_acc {s['mean_acc']:.3f}  format {s['format_rate']:.3f}  trunc {s['trunc_rate']:.3f}  sw-lang {s['sw_rate']:.3f}")


def build_report(run_dir, select_metric="pass_at_n"):
    run_dir = Path(run_dir)
    T = [f"COPSD run report: {run_dir.name}", f"generated {datetime.now().isoformat(timespec='seconds')}"]
    cfg, ecfg = read_json(run_dir / "config.json", {}), read_json(run_dir / "eval_config.json")
    T.append("\n## Setup")
    if cfg.get("teacher_context"):
        T.append(f"  teacher context: {cfg['teacher_context']}")
    for d in cfg.get("deviations", []):
        T.append(f"  - {d}")
    if ecfg:
        T.append(f"  eval sampling: {ecfg['sampling']}")
        T.append(f"  eval protocol: {ecfg['protocol']}")
        if ecfg.get("checkpoint_dir"):
            T.append(f"  checkpoints: {ecfg['checkpoint_dir']}")
        T.append("  sw-lang: share of samples whose function words are mostly Swahili (LaTeX removed)")

    lc = read_json(run_dir / "loss_check.json")
    if lc:
        T.append("\n## Loss-sign check (unclipped KL must be >= 0; the official 0.05 clip can make the loss negative)")
        for k, v in lc.items():
            T.append(f"  {k}: {v}")

    p = run_dir / "train_loss.csv"
    if p.exists():
        L = [l.split(",") for l in p.read_text().splitlines()]
        col = {k: i for i, k in enumerate(L[0])}
        rows = L[1:]
        st = [float(r[col["step_s"]]) for r in rows]
        T.append("\n## Training")
        T.append(f"  steps logged: {len(rows)}; seconds/step mean {np.mean(st):.1f} (min {min(st):.1f}, max {max(st):.1f}); "
                 f"total {sum(st) / 3600:.2f} h; peak torch alloc {max(float(r[col['peak_alloc_mib']]) for r in rows):.0f} MiB; "
                 f"peak GPU used (incl. vLLM) {max(int(r[col['gpu_used_mib']]) for r in rows)} MiB")
        T.append("  step  loss      grad_norm  mean_len  trunc  boxed")
        for r in rows:
            if int(r[0]) % 5 == 0 or int(r[0]) <= 3:
                T.append(f"  {r[0]:>4}  {float(r[1]):.5f}  {float(r[2]):.4f}    {float(r[col['mean_completion_tokens']]):>6.0f}   "
                         f"{float(r[col['frac_truncated']]):.2f}   {float(r[col['frac_boxed']]):.2f}")

    dev, test = evaluated(run_dir, "dev"), evaluated(run_dir, "test")
    if dev:
        T.append(f"\n## DEV ({len(next(iter(dev.values())))} items; used only for checkpoint selection)")
        for nm, R in dev.items():
            T.append(_line(nm, R))
    sel = select_checkpoint(run_dir, select_metric)
    if sel:
        T.append(f"  SELECTED by DEV {METRICS[select_metric]} (ties -> earlier step): {sel}")

    B, S = test.get("base"), test.get(sel) if sel else None
    if B and S:
        assert [r["item"] for r in B] == [r["item"] for r in S]
        T.append(f"\n## TEST ({len(B)} items): base vs selected checkpoint")
        T.append(_line("base", B))
        T.append(_line(sel, S))
        T.append("  Δ (selected - base), paired bootstrap over items, 10,000 resamples:")
        for key, label in (("pass_at_n", "pass@n"), ("maj_correct", "maj@n"), ("mean_acc", "mean_acc")):
            d, lo, hi = paired_bootstrap([r[key] for r in B], [r[key] for r in S])
            T.append(f"    {label:<9} {d * 100:+.2f} pts  [{lo * 100:+.2f}, {hi * 100:+.2f}]")
        d, lo, hi = paired_bootstrap([np.mean(r["formatted"]) for r in B], [np.mean(r["formatted"]) for r in S])
        T.append(f"    {'format':<9} {d * 100:+.2f} pts  [{lo * 100:+.2f}, {hi * 100:+.2f}]   (do gains track format?)")
        up = sum(s["pass_at_n"] and not b["pass_at_n"] for b, s in zip(B, S))
        dn = sum(b["pass_at_n"] and not s["pass_at_n"] for b, s in zip(B, S))
        T.append(f"  pass@n discordant items: gained {up}, lost {dn}")
        for nm, R in (("base", B), (sel, S)):
            fc = sum(c for r in R for c, f in zip(r["correct"], r["formatted"]) if f)
            T.append(f"  {nm:<8} accuracy among formatted samples: {fmt(fc, sum(sum(r['formatted']) for r in R))}"
                     "  (composition-biased: the base formats mostly its short, easy samples)")
    other = [n for n in test if n not in ("base", sel)]
    if other:
        T.append("\n## Other checkpoints on TEST (not used for selection)")
        for nm in other:
            T.append(_line(nm, test[nm]))
    if "base" in dev and B:
        allp = (sum(r["pass_at_n"] for r in B) + sum(r["pass_at_n"] for r in dev["base"])) / (len(B) + len(dev["base"]))
        T.append(f"\n  base pass@n on all {len(B) + len(dev['base'])} items (DEV+TEST): {100 * allp:.1f} "
                 "(COPSD paper, Qwen3-1.7B Swahili, pass@12 at 4096 tokens: 14.8)")
    text = "\n".join(T) + "\n"
    (run_dir / "report.txt").write_text(text)
    return text
