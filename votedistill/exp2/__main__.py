"""Experiment 2 command line. Every command is idempotent (finished evaluations are skipped).

  python -m votedistill.exp2 eval   --level L0 --split dev test      # base model, COPSD protocol
  python -m votedistill.exp2 report --level L0                       # writes results/exp2/L0/report.txt
  python -m votedistill.exp2 screen --name copsd_k2 --contexts L0-same L1 L2-gold L3-gold L3-shuffled-gold
  python -m votedistill.exp2 screen-report --name copsd_k2           # pre-training D screen (Zou et al. 2026)
  python -m votedistill.exp2 clipstats --name copsd_k2               # where the official loss clip binds
  python -m votedistill.exp2 translate                               # GSM8K train sample -> Swahili (NLLB)
  python -m votedistill.exp2 trainsets                               # sizes of every training set / vote acceptance
  python -m votedistill.exp2 train      --name L3-gold__copsd__exact__s42   # <context>__<data>__<layout>__s<seed>
  python -m votedistill.exp2 eval-run   --name ... --split dev|test  # DEV: all checkpoints; TEST: DEV-selected
  python -m votedistill.exp2 run-report --name ...
  python -m votedistill.exp2 summary                                 # all runs: results/exp2/SUMMARY.txt

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
    cs = sub.add_parser("clipstats", help="where the official loss clip binds (re-scores the screen rollouts)")
    cs.add_argument("--name", default="copsd_k2")
    cs.add_argument("--report_only", action="store_true")
    cs.add_argument("--ckpt", nargs=2, metavar=("RUN", "STEP"), help="student = that run's checkpoint (new rollouts)")
    sub.add_parser("summary")
    de = sub.add_parser("decode-eval", help="TEST-evaluate a trained run's checkpoint under a base-model level's "
                        "decoding settings (e.g. repetition penalty); writes results/exp2/decode/<run>__<level>/")
    de.add_argument("--name", required=True)
    de.add_argument("--step", type=int, default=100)
    de.add_argument("--level", required=True, choices=list(LEVELS))
    sub.add_parser("selfsft-gen-r2").add_argument("--name", required=True)
    g = sub.add_parser("selfsft-gen")
    g.add_argument("--data", required=True)
    g.add_argument("--rep_pen", type=float, default=1.1)
    tr = sub.add_parser("translate")
    tr.add_argument("--data", default="gsm8k-mt", choices=["gsm8k-mt", "gsm8k-mt3k"])
    tr.add_argument("--lang", default="swa", choices=["swa", "zul", "yor"])
    sub.add_parser("trainsets")
    for c in ("train", "eval-run", "run-report", "eval-final", "eval-run-final"):
        x = sub.add_parser(c)
        x.add_argument("--name", required=True)
        x.add_argument("--max_steps", type=int, default=None)
        if c == "eval-run":
            x.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()

    if args.cmd == "summary":
        from .summary import build_summary
        print(build_summary())
        return
    if args.cmd in ("translate", "trainsets", "train", "eval-run", "run-report", "eval-final", "eval-run-final",
                    "selfsft-gen", "selfsft-gen-r2"):
        return run_commands(args)

    if args.cmd == "decode-eval":
        from .runs import decode_eval
        print(decode_eval(args.name, args.step, args.level))
        return
    if args.cmd == "clipstats":
        from .clipstats import ckpt_rollouts, clipstats, clipstats_report
        from .screen import screen_dir
        if args.ckpt:
            setup_logging(screen_dir("copsd_k2") / "logs" / "clipstats_ckpt.log")
            args.name = ckpt_rollouts(args.ckpt[0], int(args.ckpt[1]))
        if args.report_only:
            print(clipstats_report(args.name))
            return
        setup_logging(screen_dir(args.name) / "logs" / "clipstats.log")
        print(clipstats(args.name))
        return
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
    cfg = dict(EVAL)
    if LEVELS[args.level].get("style"):
        cfg["prompt_style"] = LEVELS[args.level]["style"]
    cfg.update(LEVELS[args.level].get("sampling", {}))
    for split in args.split:
        evaluate(rd, ["base"], split, cfg, ckpt_dir=None, lang=lang)


def run_commands(args):
    import logging
    from . import runs
    if args.cmd == "trainsets":
        from .trainsets import DATASETS, training_rows
        for ctx in ("L3-gold", "L3-vote"):
            for d in DATASETS:
                try:
                    rows, desc = training_rows(d, ctx)
                    print(ctx, d, desc)
                except SystemExit as e:
                    print(ctx, d, "unavailable:", e)
        return
    if args.cmd == "selfsft-gen-r2":
        from .runs import paths
        from .selfsft import round2_pool
        setup_logging(paths(args.name)[0] / "logs" / "round2_pool.log")
        round2_pool(args.name)
        return
    if args.cmd == "selfsft-gen":
        from .selfsft import SFT_DIR, filter_arms, generate
        setup_logging(SFT_DIR / "logs" / f"gen_{args.data}.log")
        generate(args.data, args.rep_pen)
        filter_arms(args.data, args.rep_pen)
        return
    if args.cmd == "translate":
        from .trainsets import MT_FILE
        setup_logging(MT_FILE.parent / "logs" / "translate.log")
        from .translate import translate
        translate(dataset=args.data, lang=args.lang)
        return
    if args.cmd == "run-report":
        print(runs.report_run(args.name))
        return
    runs.parse_name(args.name)
    rd, _ = runs.paths(args.name)
    setup_logging(rd / "logs" / f"{args.cmd}.log")
    logging.getLogger("votedistill.exp2").info(
        f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')} git {_git('rev-parse', 'HEAD')}"
        f"{'+dirty' if _git('status', '--porcelain', '--', 'votedistill') else ''}")
    steps = args.max_steps or runs.MAX_STEPS
    if args.cmd == "train":
        runs.train_run(args.name, steps)
    elif args.cmd == "eval-final":
        print(runs.eval_final(args.name))
    elif args.cmd == "eval-run-final":
        runs.eval_run_final(args.name)
    else:
        print("selected:", runs.eval_run(args.name, args.split, steps))


if __name__ == "__main__":
    main()
