"""Delete non-final step-* adapters of finished, evaluated GRPO runs (user cleanup rule, 2026-11-15).
Kept: the final step; every 50th step of the main method (votelidref q16 300/600, for the training curve); any run
whose final checkpoint or step*_test eval is missing. usage: prune_grpo.py [--do]"""
import re, shutil, sys
from pathlib import Path
CK = Path("/data/user_data/jeffc2/votedistill_ckpt"); RES = Path("results")
CURVE = {"grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s42", "grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps600__s42"}
do = "--do" in sys.argv; tot = 0
for root in CK.glob("exp2*"):
    res = RES / root.name / "runs"
    for run in sorted(p for p in root.glob("grpo__*") if p.is_dir()):
        m = re.search(r"-steps(\d+)", run.name); final = int(m.group(1)) if m else None
        if final is None or not (run / f"step-{final}" / "COMPLETE").exists() or not list((res / run.name / "eval").glob(f"step{final}_test.jsonl")):
            continue
        for d in run.glob("step-*"):
            s = int(d.name.split("-")[1])
            if s == final or (run.name in CURVE and s % 50 == 0):
                continue
            tot += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if do: shutil.rmtree(d)
print(f"{'deleted' if do else 'would delete'} {tot/2**30:.1f} GiB")
