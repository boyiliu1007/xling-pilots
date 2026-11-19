"""Why does the own-English-vote reward (D) match or beat the gold reward (B)? CPU-only analysis of the saved
training rollouts and the English consensus file (PREREGISTRATION 2026-11-18 evening). Usage: python logs/why_vote.py"""
import json, collections, statistics as st
from votedistill.exp2.trainsets import voted_references, _gsm8k_rows
from votedistill.exp2.levels import EXP2_RESULTS
from votedistill.answers_numeric import numeric_equal

RUNS = EXP2_RESULTS / "runs"   # run with BASE_MODEL=Qwen/Qwen3-4B
D = "grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps600__s42"
B = "grpo__gsm8k-mt3k__exact__goldlidref-lr5e-6-q16-steps600__s42"

rows = {r["qid"]: r for r in _gsm8k_rows("gsm8k-mt3k")}
refs = voted_references("gsm8k-mt3k")
acc = {q: v for q, v in refs.items() if q in rows}
right = {q for q, v in acc.items() if numeric_equal(v[1], rows[q]["Answer"])}
print(f"pool {len(rows)} questions; vote accepts {len(acc)} ({100*len(acc)/len(rows):.1f}%); "
      f"vote = gold on {100*len(right)/len(acc):.1f}% of accepted")

def groups(run):
    g = collections.defaultdict(list)
    for l in open(RUNS / run / "rollouts.jsonl"):
        r = json.loads(l); g[(r["step"], r["qid"])].append(r)
    return g

def frac(xs): return 100 * sum(xs) / max(1, len(xs))

for name, run, key in (("B gold", B, "correct"), ("D vote", D, "vote_ok")):
    g = groups(run)
    print(f"\n== {name} ({len(g)} logged groups)")
    for lo, hi in ((1, 100), (101, 300), (301, 600)):
        sel = [(k, v) for k, v in g.items() if lo <= k[0] <= hi]
        if not sel: continue
        sig = [len({x[key] for x in v}) > 1 for k, v in sel]          # answer reward varies inside the group
        allwrong = [not any(x[key] for x in v) for k, v in sel]
        inpool = [k[1] in acc for k, v in sel]
        voteright = [k[1] in right for k, v in sel if k[1] in acc]
        gold = [x["correct"] for k, v in sel for x in v]
        print(f" steps {lo}-{hi}: groups {len(sel)}; answer signal {frac(sig):.1f}%; all-wrong {frac(allwrong):.1f}%; "
              f"in vote pool {frac(inpool):.1f}%; vote=gold {frac(voteright):.1f}%; gold acc {frac(gold):.1f}%")
        if name.startswith("B"):
            for lab, cond in (("vote-accepted", lambda q: q in acc), ("vote-rejected", lambda q: q not in acc)):
                s = [(k, v) for k, v in sel if cond(k[1])]
                print(f"   {lab:13s}: groups {len(s)}; gold acc {frac([x['correct'] for k,v in s for x in v]):.1f}%; "
                      f"all-wrong {frac([not any(x['correct'] for x in v) for k,v in s]):.1f}%; "
                      f"signal {frac([len({x['correct'] for x in v})>1 for k,v in s]):.1f}%")
        else:
            for lab, cond in (("vote right", lambda q: q in right), ("vote wrong", lambda q: q in acc and q not in right)):
                s = [(k, v) for k, v in sel if cond(k[1])]
                print(f"   {lab:13s}: groups {len(s)}; rewarded(vote) {frac([x['vote_ok'] for k,v in s for x in v]):.1f}%; "
                      f"gold acc {frac([x['correct'] for k,v in s for x in v]):.1f}%")

# --- the reference term: own English solution (D) vs GSM8K human rationale (B)
print("\n== reference term (ref_f1 = number-set F1 vs the reference)")
for name, run, key in (("B gold rationale", B, "correct"), ("D own English solution", D, "vote_ok")):
    g = groups(run)
    for lo, hi in ((1, 100), (101, 300), (301, 600)):
        sel = [v for k, v in g.items() if lo <= k[0] <= hi]
        f_ok = [x["ref_f1"] for v in sel for x in v if x["correct"]]
        f_bad = [x["ref_f1"] for v in sel for x in v if not x["correct"]]
        # in groups where the answer reward is flat (all wrong), does ref_f1 still rank rollouts, and toward what?
        flat = [v for v in sel if not any(x[key] for x in v)]
        spread = [max(x["ref_f1"] for x in v) - min(x["ref_f1"] for x in v) for v in flat]
        # within signal groups: is the highest-ref_f1 rollout gold-correct more often than chance?
        sig = [v for v in sel if len({x["correct"] for x in v}) > 1]
        top = [max(v, key=lambda x: x["ref_f1"])["correct"] for v in sig]
        base = [st.mean(x["correct"] for x in v) for v in sig]
        print(f" {name:24s} steps {lo}-{hi}: ref_f1 correct {st.mean(f_ok):.3f} / wrong {st.mean(f_bad):.3f}; "
              f"flat groups {len(flat)} with ref spread {st.mean(spread):.3f}; "
              f"top-ref rollout correct {frac(top):.1f}% vs group mean {100*st.mean(base):.1f}%")
