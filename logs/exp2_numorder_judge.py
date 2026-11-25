"""Order-aware number-matching judge (PREREGISTRATION 2026-11-23). Offline, CPU.
usage: BASE_MODEL=Qwen/Qwen3-4B PYTHONPATH=. python logs/exp2_numorder_judge.py  -> results/exp2_qwen3-4b/judge/report_numorder.txt"""
import json, numpy as np
from votedistill.exp2.grpo import _NUM, ref_f1
from votedistill.exp2.judge import OUT, _boot_ci
from votedistill.utils import read_jsonl

def seq(text):
    out, seen = [], set()
    for x in _NUM.findall(text or ""):
        try: v = round(float(x.replace(",", "")), 4)
        except ValueError: continue
        if v not in seen: seen.add(v); out.append(v)
    return out

def lcs(a, b):
    d = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = d[j + 1]
            d[j + 1] = prev + 1 if x == y else max(d[j + 1], d[j])
            prev = cur
    return d[-1]

def num_lcs(text, ref):
    a, b = seq(text), seq(ref)
    return 0.0 if not a or not b else 2 * lcs(a, b) / (len(a) + len(b))

items = {it["id"]: it for it in read_jsonl(OUT / "items.jsonl")}
qinfo = {q["qid"]: q for q in json.load(open(OUT / "questions.json"))}
pairs = [p for p in read_jsonl(OUT / "pairs2.jsonl") if p["type"] in ("P1", "P3", "P4")] + list(read_jsonl(OUT / "pairs5.jsonl"))
S = {}
for i, it in items.items():
    ref = qinfo[it["qid"]]["ref"]
    f, l = ref_f1(it["text"], ref), num_lcs(it["text"], ref)
    S[("numF1", i)], S[("numLCS", i)], S[("numMix", i)] = f, l, 0.5 * f + 0.5 * l
C = {(r["better"], r["worse"]): r["pref_better"] for r in read_jsonl(OUT / "scores3.jsonl") if r["variant"] == "C"}

def acc(v, t):
    by = {}
    for p in pairs:
        if p["type"] != t: continue
        if v == "C":
            x = C.get((p["better"], p["worse"]))
            if x is None: continue
        else:
            b, w = S[(v, p["better"])], S[(v, p["worse"])]
            x = 1.0 if b > w else 0.5 if b == w else 0.0
            by.setdefault((p["policy"], p["qid"]), []).append(x); continue
        by.setdefault((p["policy"], p["qid"]), []).append(1.0 if x > 0.5 else 0.5 if x == 0.5 else 0.0)
    return {k: np.array(x) for k, x in by.items()}

L = ["Order-aware number judge (PREREGISTRATION 2026-11-23). Pair accuracy, ties 0.5, question-clustered 95% CI.",
     "Pairs: " + " ".join(f"{t}={sum(p['type'] == t for p in pairs)}" for t in ("P1", "P3", "P4", "P5")), "",
     f"{'judge':8s} " + "  ".join(f"{t:>20s}" for t in ("P1", "P3", "P4", "P5")) + "   P5 ties"]
for v in ("numF1", "numLCS", "numMix", "C"):
    cells = []
    for t in ("P1", "P3", "P4", "P5"):
        m, lo, hi = _boot_ci(list(acc(v, t).values())); cells.append(f"{100*m:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]")
    ties = np.mean(np.concatenate(list(acc(v, "P5").values())) == 0.5)
    L.append(f"{v:8s} " + "  ".join(f"{c:>20s}" for c in cells) + f"   {100*ties:5.1f}%")
L.append("")
for t in ("P5", "P1", "P3"):
    for a, b in (("numLCS", "numF1"), ("numMix", "numF1"), ("numMix", "C"), ("numLCS", "C")):
        A, B = acc(a, t), acc(b, t); ks = [k for k in A if k in B and len(A[k]) == len(B[k])]
        m, lo, hi = _boot_ci([A[k] - B[k] for k in ks])
        L.append(f"{t} {a} - {b}: {100*m:+5.1f} [{100*lo:+5.1f}, {100*hi:+5.1f}]  ({sum(len(A[k]) for k in ks)} pairs)")
open(OUT / "report_numorder.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
