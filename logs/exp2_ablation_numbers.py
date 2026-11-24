# Ablation slide numbers: 3-seed means + paired bootstrap vs D (AfriMGSM-swa TEST, step 300; RFT final step)
import glob, sys, os
exec(open("logs/exp2_copsd_setup_numbers.py").read().split('print("== Zulu")')[0])
def last(d):
    fs=sorted(glob.glob(d+"/eval/step*_test.jsonl"), key=lambda p:int(p.split("/step")[-1].split("_")[0])); return fs[-1]
X={}
for k in ("votelidref","goldlidref","votelid","voteonly","lidonly","selfvotelid","votelidrefpw"):
    f=lambda s: R+f"grpo__gsm8k-mt3k__exact__{k}-lr5e-6-q16-steps300__s{s}/eval/step300_test.jsonl"
    X[k]={s:load(f(s)) for s in (42,43,44) if os.path.exists(f(s))}
X["rft"]={s:load(last(R+f"selfsft-vote__gsm8k-mt3k__exact__lidvote__s{s}")) for s in (42,43,44)}
X["base"]={42:load(R+"grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s42/eval/base_test.jsonl")}
for k,v in X.items(): rep(k,v)
for b in ("voteonly","lidonly","selfvotelid","votelid","rft","base"): comp("votelidref",b,X["votelidref"],X[b])
comp("votelid","lidonly",X["votelid"],X["lidonly"]); comp("votelid","selfvotelid",X["votelid"],X["selfvotelid"])
comp("lidonly","base",X["lidonly"],X["base"]); comp("selfvotelid","base",X["selfvotelid"],X["base"])
# same-seed contrasts where a control is missing seeds
def comp_shared(a,b):
    sh=sorted(set(X[a])&set(X[b])); print(f"  [shared seeds {sh}]", end=""); comp(a,b,{s:X[a][s] for s in sh},{s:X[b][s] for s in sh})
for b in ("lidonly","selfvotelid"): comp_shared("votelidref",b); comp_shared("votelid",b)
