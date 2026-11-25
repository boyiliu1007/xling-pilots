"""Numbers for the "why not COPSD / self-SFT" slide (2026-11-22): Qwen3-4B, AfriMGSM Swahili TEST, same metric as
logs/exp2_report_numbers.py. usage: PYTHONPATH=. python logs/exp2_sft_slide_numbers.py"""
import glob, numpy as np
exec(open("logs/exp2_report_numbers.py").read().split("def table")[0])
R="results/exp2_qwen3-4b/"
def last(run):
    fs=[f for f in glob.glob(R+"runs/"+run+"/eval/step*_test.jsonl")]
    return max(fs,key=lambda f:int(f.split("step")[-1].split("_")[0])) if fs else None
rows=[("Base model","L0",None),("Base + repetition penalty 1.1","L0-reppen1.1",None)]
for n,lab in [("L1__copsd__exact","COPSD L1"),("L3-vote__copsd__exact","COPSD L3-vote"),("L1__copsd__exact__rp","COPSD L1, penalty rollouts"),("L3-vote__copsd__exact__rp","COPSD L3-vote, penalty rollouts"),("L3-vote__copsd__exact__noclip","COPSD L3-vote, no clip"),("L0-same__copsd__exact__hot1.5","Hot-self distillation (no teacher info)"),
 ("selfsft-vote__copsd__exact__rp-vote","Self-SFT, English-vote filter"),("selfsft-vote__copsd__exact__rp-swvote","Self-SFT, Swahili self-vote filter"),("selfsft-vote__copsd__exact__rp-nofilter","Self-SFT, no filter"),("selfsft-vote__copsd__exact__rp-gold","Self-SFT, gold filter (oracle)"),
 ("selfsft-vote__gsm8k-mt3k__exact__lidvote","Offline RFT (GSM8K-MT 3k, vote+LID)"),("grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300","D (ours, GRPO)")]:
    rows.append((lab,n,1))
D={}
for lab,n,t in rows:
    if t is None: fs={"-":R+n+"/eval/base_test.jsonl"}
    else: fs={s:last(f"{n}__s{s}") for s in (42,43,44)}; fs={s:f for s,f in fs.items() if f}
    X={s:load(f) for s,f in fs.items()}; D[lab]=X
    sw=[100*np.mean([r["sw"] for r in x.values()]) for x in X.values()]
    an=[100*np.mean([r["any"] for r in x.values()]) for x in X.values()]
    tr=[100*np.mean([r["trunc"] for r in x.values()]) for x in X.values()]
    st=",".join(sorted({f.split("/")[-1].split("_")[0] for f in fs.values()}))
    print(f"{lab:42s} seeds {len(X)} {st:8s} sw {np.mean(sw):5.1f} ({'/'.join('%.1f'%v for v in sw)})  any {np.mean(an):5.1f}  trunc {np.mean(tr):5.1f}")
def comp(a,b,q="sw"):
    A,B=D[a],D[b]; it=sorted(set.intersection(*[set(x) for x in list(A.values())+list(B.values())]))
    av=[np.mean([x[i][q] for x in A.values()]) for i in it]; bv=[np.mean([x[i][q] for x in B.values()]) for i in it]
    print(f"{a} - {b} [{q}]: {ci(paired_bootstrap(bv,av))}")
for a,b in [("Self-SFT, English-vote filter","Self-SFT, no filter"),("Self-SFT, English-vote filter","Self-SFT, Swahili self-vote filter"),("Self-SFT, gold filter (oracle)","Self-SFT, English-vote filter"),("Self-SFT, English-vote filter","Base model"),("Self-SFT, English-vote filter","Base + repetition penalty 1.1"),("COPSD L3-vote","COPSD L1"),("COPSD L3-vote, penalty rollouts","COPSD L1, penalty rollouts"),("D (ours, GRPO)","Self-SFT, English-vote filter"),("D (ours, GRPO)","Offline RFT (GSM8K-MT 3k, vote+LID)")]:
    for q in ("sw","any"): comp(a,b,q)
