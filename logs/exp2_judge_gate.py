"""Gated self-judge design selection (PREREGISTRATION 2026-11-23). Offline, CPU.
usage: BASE_MODEL=Qwen/Qwen3-4B PYTHONPATH=. python logs/exp2_judge_gate.py -> results/exp2_qwen3-4b/judge/report_gate.txt"""
import json, numpy as np
from votedistill.exp2.grpo import ref_f1
from votedistill.exp2.judge import OUT, _boot_ci
from votedistill.utils import read_jsonl
items = {it["id"]: it for it in read_jsonl(OUT / "items.jsonl")}
qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
f1 = {i: ref_f1(it["text"], qinfo[it["qid"]]["ref"]) for i, it in items.items()}
ptype = {}
for p in [p for p in read_jsonl(OUT / "pairs2.jsonl") if p["type"] in ("P1", "P3")] + list(read_jsonl(OUT / "pairs5.jsonl")):
    ptype[(p["better"], p["worse"])] = p
R = [r for r in read_jsonl(OUT / "scores3.jsonl") if r["variant"] == "C" and (r["better"], r["worse"]) in ptype]
L = ["Gated pairwise judge: accuracy (ties 0.5) [question-clustered CI] and coverage = share of pairs kept", ""]
best = None
for cons in (False, True):
    for agree in (False, True):
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            row, cov = [], {}
            for typ in ("P1", "P3", "P5"):
                by, n_all = {}, 0
                for r in R:
                    p = ptype[(r["better"], r["worse"])]
                    if p["type"] != typ: continue
                    n_all += 1
                    pref = r["pref_better"]
                    if abs(pref - 0.5) < t: continue
                    if cons and not ((r["pA_first"] > 0.5) == (r["pA_second"] < 0.5)): continue
                    if agree:
                        fb, fw = f1[r["better"]], f1[r["worse"]]
                        if fb == fw or ((fb > fw) != (pref > 0.5)): continue
                    by.setdefault((p["policy"], p["qid"]), []).append(1.0 if pref > 0.5 else 0.5 if pref == 0.5 else 0.0)
                k = sum(len(v) for v in by.values()); cov[typ] = k / n_all
                if k:
                    m, lo, hi = _boot_ci([np.array(v) for v in by.values()])
                    row.append(f"{typ} {100*m:5.1f} [{100*lo:4.1f},{100*hi:5.1f}] cov {100*k/n_all:5.1f}%")
                    cov[typ + "acc"] = m
                else:
                    row.append(f"{typ}   n/a"); cov[typ + "acc"] = float("nan")
            tag = f"conf>={t:.1f}{' +consistent' if cons else ''}{' +numF1 agree' if agree else ''}"
            L.append(f"{tag:38s} " + " | ".join(row))
            if cov["P5"] >= 0.4 and cov.get("P1acc", 0) >= 0.95 and (best is None or cov["P5acc"] > best[1]):
                best = (tag, cov["P5acc"], cov["P5"])
L += ["", f"SELECTED (rule: max P5 acc, P5 coverage >= 40%, P1 >= 95%): {best}"]
open(OUT / "report_gate.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
