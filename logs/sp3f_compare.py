"""TEST comparison of SP3F-style arms vs base / base+penalty (Swahili accuracy by LID, mean_acc). Missing files skipped."""
import json, os, numpy as np
from votedistill.exp2.grpo import lid_fraction, LID_MIN
from votedistill.stats import paired_bootstrap
def load(p):
    rows={}
    for l in open(p):
        r=json.loads(l)
        lid=[lid_fraction(o)>=LID_MIN for o in r["outputs"]]
        rows[r["item"]]=dict(acc=np.mean(r["correct"]),swl=np.mean([c and l for c,l in zip(r["correct"],lid)]),
            lid=np.mean(lid),trunc=np.mean(r["truncated"]),maj=float(r["maj_correct"]))
    return rows
E="results/exp2_qwen3-4b"
F={"base":f"{E}/L0/eval/base_test.jsonl","base+pen":f"{E}/L0-reppen1.1/eval/base_test.jsonl"}
for k,a in [("goldlid","A"),("goldlidref","B"),("votelid","C"),("votelidref","D"),("lidonly","E"),("selfvotelid","F"),
            ("voteonly","G"),("votelidstep","H"),("votelidrefanc","I")]:
    n=f"grpo__gsm8k-mt3k__exact__{k}-lr5e-6-q16-steps300__s42"
    F[a]=f"{E}/runs/{n}/eval/step300_test.jsonl"; F[a+"+pen"]=f"{E}/decode/{n}__L0-reppen1.1/eval/step300_test.jsonl"
R={k:load(v) for k,v in F.items() if os.path.exists(v)}
items=sorted(R["base"])
print("n",len(items))
for k in R: print(f"{k:9s}", " ".join(f"{m}={100*np.mean([R[k][i][m] for i in items]):.1f}" for m in ["swl","acc","lid","trunc","maj"]))
f=lambda t:"%+.1f [%+.1f, %+.1f]"%tuple(100*x for x in t)
for a,b in [("base","D"),("base+pen","D"),("A","D"),("B","D"),("C","D"),("base+pen","D+pen"),("B+pen","D+pen"),
            ("base","C"),("base+pen","C"),("A","C"),("E","C"),("F","C"),("G","C"),("C","H"),("D","I"),("base","E"),("base+pen","E"),("base+pen","A"),("base+pen","B"),("A","B")]:
    if a in R and b in R:
        print(f"{b}-{a}: swl {f(paired_bootstrap([R[a][i]['swl'] for i in items],[R[b][i]['swl'] for i in items]))}  acc {f(paired_bootstrap([R[a][i]['acc'] for i in items],[R[b][i]['acc'] for i in items]))}")
