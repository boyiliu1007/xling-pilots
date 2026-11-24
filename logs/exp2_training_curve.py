# Training curve for the slides: D (ours) vs B (gold), Qwen3-4B s42, AfriMGSM-swa TEST Swahili accuracy.
# Step 0 = untrained (base_test), 300 from the 300-step runs, 350-600 from steps600, 700-1200 from steps1200.
import json, glob, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from votedistill.exp2.grpo import lid_fraction, LID_MIN
R = "results/exp2_qwen3-4b/runs/grpo__gsm8k-mt3k__exact__{}-lr5e-6-q16-steps{}__s42/eval/"
def acc(f): return 100*np.mean([np.mean([c and lid_fraction(o,"swa")>=LID_MIN for c,o in zip(r["correct"],r["outputs"])]) for r in map(json.loads,open(f))])
curves = {}
for k in ("votelidref","goldlidref"):
    pts = {0: acc(R.format(k,300)+"base_test.jsonl"), 300: acc(R.format(k,300)+"step300_test.jsonl")}
    for n in (600,1200):
        for f in glob.glob(R.format(k,n)+"step*_test.jsonl"):
            s = int(re.search(r"step(\d+)_", f).group(1))
            if s > 300 and s not in pts: pts[s] = acc(f)
    curves[k] = dict(sorted(pts.items()))
json.dump(curves, open("results/exp2_qwen3-4b/figures/training_curve.json","w"), indent=1)
for k,v in curves.items(): print(k, {s: round(a,1) for s,a in v.items()})

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15})
fig, ax = plt.subplots(figsize=(10, 5))
style = {"votelidref": dict(color="#2a78d6", label="Ours (D): no gold", ls="-"),
         "goldlidref": dict(color="#898781", label="Gold answers + solution (B)", ls=(0,(5,3)))}
for k, st in style.items():
    x, y = list(curves[k]), list(curves[k].values())
    # no checkpoints between 0 and 300 for B: draw that span as a thin dotted connector, not as data
    ax.plot(x[:2], y[:2], color=st["color"], ls=":", lw=1.2, zorder=2)
    ax.plot(x[1:], y[1:], color=st["color"], ls=st["ls"], lw=2.2, zorder=3, label=st["label"])
    ax.plot(x, y, ls="none", marker="o", ms=7, color=st["color"], markeredgecolor="#fcfcfb",
            markeredgewidth=1.5, zorder=4)
    dy = 8 if y[-1] >= max(c[1200] for c in curves.values()) else -8
    ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(10, dy), textcoords="offset points",
                va="center", color=INK, fontsize=15)
ax.annotate("untrained 11.9", (0, curves["votelidref"][0]), xytext=(12, -4), textcoords="offset points",
            va="center", color=INK2, fontsize=13)
ax.axvline(300, color=AXIS, lw=1, ls=":", zorder=1)
ax.text(305, 13.5, "main runs\nstop here (300)", color=MUTED, fontsize=12, va="bottom")
ax.set_xlim(-30, 1300); ax.set_ylim(0, 40)
ax.set_xticks(range(0, 1201, 200)); ax.set_yticks(range(0, 41, 10))
ax.set_xlabel("Training step", color=INK2); ax.set_ylabel("Swahili accuracy (%)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(AXIS)
ax.tick_params(colors=INK2, length=0)
ax.legend(loc="lower right", handlelength=2.6, frameon=False, fontsize=14)
fig.text(0.99, 0.01, "Qwen3-4B, AfriMGSM Swahili test, 1 seed (s42)", ha="right", color=MUTED, fontsize=11)
fig.tight_layout()
for ext in ("png","svg","pdf"):
    fig.savefig(f"results/exp2_qwen3-4b/figures/training_curve.{ext}", dpi=300, transparent=(ext!="png"))
