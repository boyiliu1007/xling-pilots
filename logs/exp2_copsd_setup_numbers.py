import json, numpy as np
from votedistill.exp2.grpo import lid_fraction, LID_MIN
from votedistill.stats import paired_bootstrap
R="results/exp2_qwen3-4b/runs/"
def load(p, lang="swa"):
    out={}
    for l in open(p):
        r=json.loads(l); lid=[lid_fraction(o,lang)>=LID_MIN for o in r["outputs"]]
        out[r["item"]]={"sw":np.mean([c and x for c,x in zip(r["correct"],lid)]),"any":np.mean(r["correct"]),"trunc":np.mean(r["truncated"]),"pass":float(any(c and x for c,x in zip(r["correct"],lid)))}
    return out
def ci(t): return "%+.1f [%+.1f, %+.1f]"%tuple(100*x for x in t)
def rep(tag,X):
    print(tag, " | ".join(f"s{s}: sw {100*np.mean([r['sw'] for r in x.values()]):.1f} any {100*np.mean([r['any'] for r in x.values()]):.1f} trunc {100*np.mean([r['trunc'] for r in x.values()]):.1f} swpass {100*np.mean([r['pass'] for r in x.values()]):.1f}" for s,x in X.items()))
def comp(a,b,A,B):
    it=sorted(set.intersection(*[set(x) for x in list(A.values())+list(B.values())]))
    for q in ("sw","any","pass"):
        av=[np.mean([x[i][q] for x in A.values()]) for i in it]; bv=[np.mean([x[i][q] for x in B.values()]) for i in it]
        print(f"  {a} - {b} [{q}] ({len(A)} seeds, {len(it)} items): {ci(paired_bootstrap(bv,av))}")
print("== Zulu")
Z={k:{42:load(R+f"grpo__gsm8k-mt3k-zulonly__exact__{k}-lr5e-6-q16-steps300__s42/eval/step300_test.jsonl","zul")} for k in ("votelidref","votelidrefpw")}
Z["base"]={42:load(R+"grpo__gsm8k-mt3k-zulonly__exact__votelidref-lr5e-6-q16-steps300__s42/eval/base_test.jsonl","zul")}
for k,v in Z.items(): rep(k,v)
comp("J","D",Z["votelidrefpw"],Z["votelidref"]); comp("J","base",Z["votelidrefpw"],Z["base"])
print("== COPSD setup")
C={}
for k,(pat,st) in {"D":("grpo__copsd__exact__votelidref-lr5e-6-q16-steps200__s{}",200),"J":("grpo__copsd__exact__votelidrefpw-lr5e-6-q16-steps200__s{}",200),"L1":("L1__copsd__exact__s{}",100),"L3":("L3-vote__copsd__exact__s{}",100)}.items():
    C[k]={s:load(R+pat.format(s)+f"/eval/step{st}_test.jsonl") for s in (42,43)}; rep(k,C[k])
for a,b in [("D","L1"),("D","L3"),("J","D"),("J","L1"),("J","L3")]: comp(a,b,C[a],C[b])
