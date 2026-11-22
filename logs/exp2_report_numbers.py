"""Reproduces every number in the main-method report doc (2026-11-22) from the saved TEST outputs.
usage: PYTHONPATH=. python logs/exp2_report_numbers.py
Metric: Swahili accuracy = sample correct AND lid_fraction >= LID_MIN (lingua, all languages); any-language = correct.
Seeds are averaged per item, then paired bootstrap over items (votedistill/stats.py, 5000 resamples, 95%)."""
import json, os, numpy as np
from votedistill.exp2.grpo import lid_fraction, LID_MIN
from votedistill.stats import paired_bootstrap

R4, R8 = "results/exp2_qwen3-4b", "results/exp2_qwen3-8b"
RUN = "grpo__gsm8k-mt3k__exact__{k}-lr5e-6-q16-steps{n}__s{s}"
NAMES = {"votelidrefpw": "J", "votelidref": "D", "goldlidref": "B", "votelid": "C", "voteonly": "G"}

def load(p):
    rows = {}
    for l in open(p):
        r = json.loads(l)
        lid = [lid_fraction(o) >= LID_MIN for o in r["outputs"]]
        rows[r["item"]] = {"sw": np.mean([c and x for c, x in zip(r["correct"], lid)]), "any": np.mean(r["correct"]),
                           "trunc": np.mean(r["truncated"])}
    return rows

def ci(t): return "%+.1f [%+.1f, %+.1f]" % tuple(100 * x for x in t)

def table(tag, files, contrasts):
    D = {k: {s: load(p) for s, p in v.items() if os.path.exists(p)} for k, v in files.items()}
    D = {k: v for k, v in D.items() if v}
    print(f"\n== {tag}")
    for k, v in D.items():
        per = " / ".join(f"{100*np.mean([r['sw'] for r in x.values()]):.1f}" for x in v.values())
        m = {q: 100 * np.mean([np.mean([r[q] for r in x.values()]) for x in v.values()]) for q in ("sw", "any")}
        print(f"{NAMES.get(k, k):3s} seeds {list(v)}: sw {per} | mean sw {m['sw']:.1f} any {m['any']:.1f}")
    for a, b in contrasts:
        if a in D and b in D:
            seeds = sorted(set(D[a]) & set(D[b]))
            it = sorted(set.intersection(*[set(D[a][s]) & set(D[b][s]) for s in seeds]))
            for q in ("sw", "any"):
                A = [np.mean([D[a][s][i][q] for s in seeds]) for i in it]
                B = [np.mean([D[b][s][i][q] for s in seeds]) for i in it]
                print(f"{NAMES[b]} - {NAMES[a]} {q} ({len(seeds)} seeds, {len(it)} items): {ci(paired_bootstrap(A, B))}")
    return D

C = [("votelidref", "votelidrefpw"), ("goldlidref", "votelidrefpw"), ("votelid", "votelidrefpw"),
     ("goldlidref", "votelidref"), ("votelid", "votelidref"), ("voteonly", "votelidref")]
# 1. 4B AfriMGSM TEST, 3 seeds, step 300
A4 = table("4B AfriMGSM TEST step 300", {k: {s: f"{R4}/runs/{RUN.format(k=k, n=300, s=s)}/eval/step300_test.jsonl"
                                              for s in (42, 43, 44)} for k in NAMES}, C)
# 2. 4B MSVAMP, 3 seeds
table("4B MSVAMP step 300", {k: {s: f"{R4}/bench/{RUN.format(k=k, n=300, s=s)}__msvamp/eval/step300_test.jsonl"
                                 for s in (42, 43, 44)} for k in NAMES}, C)
# 3. 8B AfriMGSM TEST, 2 seeds
table("8B AfriMGSM TEST step 300", {k: {s: f"{R8}/runs/{RUN.format(k=k, n=300, s=s)}/eval/step300_test.jsonl"
                                         for s in (42, 43)} for k in ("votelidrefpw", "votelidref", "goldlidref")}, C)
# 4. Where J's gain comes from: items binned by the untrained model's any-language accuracy (post hoc)
base = {r["item"]: np.mean(r["correct"]) for r in map(json.loads, open(f"{R4}/runs/{RUN.format(k='votelidref', n=300, s=42)}/eval/base_test.jsonl"))}
j = {i: np.mean([A4["votelidrefpw"][s][i]["sw"] for s in (42, 43, 44)]) for i in base}
d = {i: np.mean([A4["votelidref"][s][i]["sw"] for s in (42, 43, 44)]) for i in base}
print("\n== J - D by item difficulty (untrained any-language accuracy)")
for lo, hi, lab in [(0, 1e-9, "never correct"), (1e-9, 0.34, "1-33%"), (0.34, 1.01, "over 33%")]:
    s = [i for i in base if lo <= base[i] < hi]
    print(f"{lab:14s} n={len(s):3d} D {100*np.mean([d[i] for i in s]):.1f} J {100*np.mean([j[i] for i in s]):.1f} "
          f"J-D {ci(paired_bootstrap([d[i] for i in s], [j[i] for i in s]))}")
# 5. Share of training groups the judge scored
print("\n== judged groups (groups where every rollout has a 'pw' field)")
for R, s in [(R4, 42), (R4, 43), (R4, 44), (R8, 42), (R8, 43)]:
    g = {}
    for l in open(f"{R}/runs/{RUN.format(k='votelidrefpw', n=300, s=s)}/rollouts.jsonl"):
        r = json.loads(l); g.setdefault((r["step"], r["qid"]), []).append(r)
    jd = [v for v in g.values() if all("pw" in x for x in v)]
    print(f"{R.split('_')[-1]} s{s}: {100*len(jd)/len(g):.0f}% judged, win-rate sd {np.std([x['pw'] for v in jd for x in v]):.2f}")
# 6. 1200-step curves (s42): D and B
print("\n== curves s42 (sw): D / B, D - B")
for st in (600, 700, 800, 900, 1000, 1100, 1200):
    n = 600 if st == 600 else 1200
    p = {k: f"{R4}/runs/{RUN.format(k=k, n=n, s=42)}/eval/step{st}_test.jsonl" for k in ("votelidref", "goldlidref")}
    if all(os.path.exists(x) for x in p.values()):
        a, b = load(p["goldlidref"]), load(p["votelidref"]); it = sorted(set(a) & set(b))
        print(f"{st:5d} D {100*np.mean([b[i]['sw'] for i in it]):.1f} B {100*np.mean([a[i]['sw'] for i in it]):.1f} "
              f"D-B {ci(paired_bootstrap([a[i]['sw'] for i in it], [b[i]['sw'] for i in it]))}")
