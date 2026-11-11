"""COPSD (crosslingual on-policy self-distillation) on AfriMGSM. Every command is idempotent.

  python -m votedistill.copsd train  --run_dir results/copsd_gold --ckpt_dir /data/.../copsd_gold_ckpt [--max_steps 100]
  python -m votedistill.copsd eval   --run_dir R --ckpt_dir C --names base step20 step40 --split dev
  python -m votedistill.copsd select --run_dir R                    # prints the DEV-selected checkpoint or NONE
  python -m votedistill.copsd report --run_dir R
  python -m votedistill.copsd losscheck --run_dir R --label base_step0 [--adapter C/step-60/adapter]

A run directory holds config.json, train_loss.csv, eval_config.json, eval/*.jsonl and report.txt.
An evaluation-only run (e.g. a different eval protocol) is a new --run_dir pointing at an existing --ckpt_dir.
"""
import argparse
import os
import platform
import sys
from pathlib import Path

from ..utils import setup_logging
from .config import EVAL, OFFICIAL


def main():
    ap = argparse.ArgumentParser(prog="python -m votedistill.copsd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run_dir", required=True)
    common.add_argument("--lang", default="swa")
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--select_metric", choices=["pass_at_n", "maj_correct", "mean_acc"], default="pass_at_n")

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--ckpt_dir", required=True)
    t.add_argument("--max_steps", type=int, default=OFFICIAL["max_steps"])
    t.add_argument("--stop_after", type=int, default=0, help="run at most this many steps in this invocation")
    t.add_argument("--no_save", action="store_true", help="timing run from step 0: no checkpoints, writes timing_steps.csv")
    t.add_argument("--no_problem_en", action="store_true", help="teacher does not see the English question")
    t.add_argument("--no_reference", action="store_true", help="teacher does not see an English reference solution")

    e = sub.add_parser("eval", parents=[common])
    e.add_argument("--ckpt_dir", default=None)
    e.add_argument("--names", nargs="+", required=True, help="base and/or stepN")
    e.add_argument("--split", choices=["dev", "test"], required=True)
    e.add_argument("--max_new_tokens", type=int, default=EVAL["max_new_tokens"])
    e.add_argument("--n", type=int, default=EVAL["val_n"])

    s = sub.add_parser("select", parents=[common, sel])
    s.add_argument("--names", nargs="*", default=None, help="restrict to these checkpoints; NONE unless all have DEV evals")
    sub.add_parser("report", parents=[common, sel])
    lc = sub.add_parser("losscheck", parents=[common])
    lc.add_argument("--label", required=True)
    lc.add_argument("--adapter", default=None)
    lc.add_argument("--n_microbatches", type=int, default=2)
    args = ap.parse_args()

    if args.cmd == "select":
        from .evaluate import select_checkpoint
        print(select_checkpoint(args.run_dir, args.select_metric, args.names) or "NONE")
        return
    if args.cmd == "report":
        from .report import build_report
        print(build_report(args.run_dir, args.select_metric))
        return
    setup_logging(Path(args.run_dir) / "logs" / f"{args.cmd}.log")
    import logging
    logging.getLogger("votedistill.copsd").info(f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
    if args.cmd == "train":
        from .train import train
        train(args.run_dir, args.ckpt_dir, args.lang, args.max_steps, args.stop_after, args.no_save,
              not args.no_problem_en, not args.no_reference)
    elif args.cmd == "eval":
        from .evaluate import evaluate
        cfg = {**EVAL, "max_new_tokens": args.max_new_tokens, "val_n": args.n}
        evaluate(args.run_dir, args.names, args.split, cfg, args.ckpt_dir, args.lang)
    else:
        from .losscheck import loss_check
        loss_check(args.run_dir, args.label, args.adapter, args.n_microbatches, args.lang)


if __name__ == "__main__":
    main()
