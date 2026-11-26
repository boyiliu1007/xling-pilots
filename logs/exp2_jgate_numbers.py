# J-gate / D-more scoring vs D and J (preregistered predictions, PREREGISTRATION.md)
import os, sys
sys.argv=[sys.argv[0]]
exec(open("logs/exp2_copsd_setup_numbers.py").read().split('print("== Zulu")')[0])
P="grpo__gsm8k-mt3k__exact__{}-lr5e-6-q16-steps300__s{}/eval/step300_test.jsonl"
for size,seeds in (("8b",(42,43)),("4b",(42,43,44))):
    R=f"results/exp2_qwen3-{size}/runs/"
    X={}
    for k in ("votelidref","votelidrefpw","votelidrefpwg","votelidrefmore"):
        d={s:load(R+P.format(k,s)) for s in seeds if os.path.exists(R+P.format(k,s))}
        if d: X[k]=d
    print("==",size, {k:sorted(v) for k,v in X.items()})
    for k,v in X.items(): rep(k,v)
    def c(a,b):
        if a in X and b in X:
            s=sorted(set(X[a])&set(X[b])); comp(a,b,{i:X[a][i] for i in s},{i:X[b][i] for i in s})
    c("votelidrefpwg","votelidref"); c("votelidrefpwg","votelidrefpw"); c("votelidrefpw","votelidref")
    c("votelidrefpwg","votelidrefmore"); c("votelidrefpw","votelidrefmore"); c("votelidrefmore","votelidref")
