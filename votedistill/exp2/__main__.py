"""Experiment 2 command line. Every command is idempotent (finished evaluations are skipped).

  python -m votedistill.exp2 eval   --level L0 --split dev test      # base model, COPSD protocol
  python -m votedistill.exp2 report --level L0                       # writes results/exp2/L0/report.txt
  python -m votedistill.exp2 screen --name copsd_k2 --contexts L0-same L1 L2-gold L3-gold L3-shuffled-gold
  python -m votedistill.exp2 screen-report --name copsd_k2           # pre-training D screen (Zou et al. 2026)

Outputs: results/exp2/<level>/{eval/*.jsonl, eval_config.json, env.json, report.txt, logs/}.
"""
import argparse
import os
import platform
import subprocess
import sys

from ..paths import COPSD_COMMIT, MODEL, REPO
from ..utils import read_json, setup_logging, versions, write_json
from .levels import LEVELS, run_dir


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def record_env(rd):
    """Pin the software/hardware environment of a level's evaluations; refuse to mix environments in one run_dir."""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    env = {"model": MODEL, "copsd_commit": COPSD_COMMIT, "versions": versions(),
           "gpu": gpu.strip().splitlines()[0] if gpu.strip() else "none"}
    old = read_json(rd / "env.json")
    if old is not None and {k: old[k] for k in env} != env:
        raise SystemExit(f"{rd / 'env.json'} pins {old}, but this environment is {env}; use a new level directory")
    write_json(rd / "env.json", {**env, "git_commit": _git("rev-parse", "HEAD"),
                                 "git_dirty": bool(_git("status", "--porcelain", "--", "votedistill"))})


def main():
    ap = argparse.ArgumentParser(prog="python -m votedistill.exp2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--level", required=True, choices=list(LEVELS))
    e.add_argument("--split", nargs="+", choices=["dev", "test"], required=True)
    r = sub.add_parser("report")
    r.add_argument("--level", required=True, choices=list(LEVELS))
    sc = sub.add_parser("screen")
    sc.add_argument("--name", required=True)
    sc.add_argument("--contexts", nargs="+", required=True)
    sc.add_argument("--n_problems", type=int, default=500)
    sc.add_argument("--k", type=int, default=2, help="student rollouts per problem")
    sr = sub.add_parser("screen-report")
    sr.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.cmd == "screen-report":
        from .screen import screen_report
        print(screen_report(args.name))
        return
    if args.cmd == "screen":
        from .screen import screen, screen_dir
        setup_logging(screen_dir(args.name) / "logs" / "screen.log")
        import logging
        logging.getLogger("votedistill.exp2").info(f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
        screen(args.name, args.contexts, args.n_problems, args.k)
        return
    rd = run_dir(args.level)
    if args.cmd == "report":
        from .report import build_report
        print(build_report(args.level))
        return
    if LEVELS[args.level]["trains"]:
        raise SystemExit(f"{args.level} needs a training run first; only L0 is implemented so far")
    rd.mkdir(parents=True, exist_ok=True)
    setup_logging(rd / "logs" / "eval.log")
    import logging
    logging.getLogger("votedistill.exp2").info(
        f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
    record_env(rd)
    from ..copsd.config import EVAL
    from ..copsd.evaluate import evaluate
    lang = LEVELS[args.level].get("lang", "swa")
    if lang != "swa":   # the per-item comparison with Swahili assumes AfriMGSM rows are aligned across languages
        from ..data import afrimgsm_items, check_alignment
        lines = []
        check_alignment(afrimgsm_items("en", lines), afrimgsm_items("swa", lines), "swa", lines)
        (rd / "alignment.txt").write_text("\n".join(lines) + "\n")
    for split in args.split:
        evaluate(rd, ["base"], split, dict(EVAL), ckpt_dir=None, lang=lang)


if __name__ == "__main__":
    main()
