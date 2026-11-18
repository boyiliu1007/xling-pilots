"""Submit train + eval-final job pairs for a list of run names and write a watchdog state file.
usage: submit_grid.py STATE MODEL TRAIN_GPU NAME... (MODEL = default | Qwen/Qwen3-4B; TRAIN_GPU e.g. A6000)
A name prefixed with 'eval:' gets only the eval-final job (run already trained)."""
import json, os, subprocess, sys
state, model, gpu, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
X = "--partition=preempt --qos=preempt_qos --requeue"
bm = "" if model == "default" else f",BASE_MODEL={model}"
tag = "17b-" if model == "default" else model.split("-")[-1].lower() + "-"
jobs = {}
tm = f"--time={os.environ['TIME']}" if os.environ.get("TIME") else ""
def sb(args):
    return subprocess.run(f"sbatch --parsable --exclude=babel-u9-20,babel-w9-28,babel-w9-26,babel-l9-32 {args}",
                          shell=True, capture_output=True, text=True, check=True).stdout.strip()
for n in names:
    ev_only = n.startswith("eval:")
    n = n.removeprefix("eval:")
    dep = None
    if not ev_only:
        a = f"{X} --gres=gpu:{gpu}:1 {tm} --job-name=e2-{tag}tr-{n} --export=ALL,NAME={n}{bm} slurm/exp2_train.sbatch"
        jobs[f"train:{n}"] = {"id": sb(a), "dep": None, "retries": 0, "sbatch": a}
        dep = f"train:{n}"
    a = f"{X} --job-name=e2-{tag}ev-{n} --export=ALL,NAME={n}{bm} slurm/exp2_evalfinal.sbatch"
    d = f"--dependency=afterok:{jobs[dep]['id']} " if dep else ""
    jobs[f"eval:{n}"] = {"id": sb(d + a), "dep": dep, "retries": 0, "sbatch": a}
json.dump(jobs, open(state, "w"), indent=1)
print(json.dumps({k: v["id"] for k, v in jobs.items()}, indent=0))
