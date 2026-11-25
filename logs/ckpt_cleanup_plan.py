# Plan checkpoint cleanup: keep final adapter per run (+ init sources + TEST-evaluated steps of COPSD-style runs);
# delete state.pt only for finished runs. Dry run unless --apply.
import os, re, sys, glob, json, shutil
C="/data/user_data/jeffc2/votedistill_ckpt"; RES={"exp2":"results/exp2/runs","exp2_qwen3-4b":"results/exp2_qwen3-4b/runs","exp2_qwen3-8b":"results/exp2_qwen3-8b/runs"}
keep_init={(m.group(1), int(m.group(2))) for f in glob.glob("results/exp2*/runs/*/config.json")
           for m in [re.search(r'"init_from": "(\S+) step (\d+)', open(f).read())] if m}
def size(p): return sum(os.path.getsize(os.path.join(a,f)) for a,_,fs in os.walk(p) for f in fs)
dele, keep_unfinished, freed = [], [], 0
for grp, rdir in RES.items():
    for run in sorted(os.listdir(f"{C}/{grp}")):
        rp=f"{C}/{grp}/{run}"; steps=sorted(int(d.split("-")[1]) for d in os.listdir(rp) if re.fullmatch(r"step-\d+", d))
        if not steps: continue
        m=re.search(r"steps(\d+)", run); target=int(m.group(1)) if m else None
        cfgp=f"{rdir}/{run}/config.json"
        if target is None and os.path.exists(cfgp):
            c=json.load(open(cfgp)); target=c.get("steps") or c.get("max_steps") or c.get("total_steps")
        finished = target is None or steps[-1] >= int(target)
        keep={steps[-1]} | {s for r,s in keep_init if r==run}
        if grp=="exp2":   # COPSD-style: keep TEST-evaluated (DEV-selected) steps
            keep |= {int(x) for x in re.findall(r"step(\d+)_test", " ".join(os.listdir(f"{rdir}/{run}/eval"))) } if os.path.isdir(f"{rdir}/{run}/eval") else set()
        for s in steps:
            if s not in keep: dele.append(f"{rp}/step-{s}")
        st=f"{rp}/state.pt"
        if os.path.exists(st):
            if finished: dele.append(st)
            else: keep_unfinished.append(f"{run} (at step {steps[-1]}/{target})")
for p in dele: freed += size(p) if os.path.isdir(p) else os.path.getsize(p)
print(f"delete {len(dele)} items, frees {freed/1e9:.1f} GB"); print("unfinished runs (state.pt kept):", *keep_unfinished, sep="\n  ")
print("init sources kept:", sorted(keep_init))
if "--apply" in sys.argv:
    for p in dele: shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    print("APPLIED")
