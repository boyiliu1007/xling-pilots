"""CLI for Szu-Ju's experiments (see votedistill/xfer/__init__.py).

  python -m votedistill.xfer train   --name uopsd__gsm8k-mt__clip__s42        # U-OPSD (uopsd.py)
  python -m votedistill.xfer train   --name rc__gsm8k-mt__first256__noclip__s42  # reading OPSD (rc.py)
  python -m votedistill.xfer xeval   --label L3-gold__copsd__exact__s42 --ckpt_dir <dir with step-100/adapter> --view enen
  python -m votedistill.xfer xevals  --label <label> --ckpt_dir <dir> [--views swsw swen enen swsw-rp]
  python -m votedistill.xfer run     --name uopsd__gsm8k-mt__clip__s42      # train, then xevals of its step 100
  python -m votedistill.xfer report
"""
import argparse
import os
import platform
import subprocess
import sys

from ..utils import setup_logging
from .xeval import VIEWS


def _xevals(label, ckpt_dir, step, views):
    # one vLLM engine per process (an engine is not released inside a process)
    for v in views:
        subprocess.run([sys.executable, "-m", "votedistill.xfer", "xeval", "--label", label, "--ckpt_dir", str(ckpt_dir),
                        "--step", str(step), "--view", v], check=True)


def main():
    ap = argparse.ArgumentParser(prog="python -m votedistill.xfer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("train", "run"):
        x = sub.add_parser(c)
        x.add_argument("--name", required=True)
        x.add_argument("--max_steps", type=int, default=100)
        x.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    x = sub.add_parser("xeval")
    x.add_argument("--label", required=True)
    x.add_argument("--ckpt_dir", required=True)
    x.add_argument("--step", type=int, default=100)
    x.add_argument("--view", required=True, choices=list(VIEWS))
    x = sub.add_parser("xevals")
    x.add_argument("--label", required=True)
    x.add_argument("--ckpt_dir", required=True)
    x.add_argument("--step", type=int, default=100)
    x.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    x = sub.add_parser("report")
    x.add_argument("--step", type=int, default=100)
    args = ap.parse_args()

    from .uopsd import XFER, paths
    import logging
    if args.cmd == "report":
        from .report import report_all
        print(report_all(args.step))
        return
    if args.cmd in ("train", "run"):
        rd, ck = paths(args.name)
        setup_logging(rd / "logs" / f"{args.cmd}.log")
        logging.getLogger("votedistill.xfer").info(f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
        if args.name.startswith("rc__"):
            from .rc import train_rc
            train_rc(args.name, args.max_steps)
        else:
            from .uopsd import train_uopsd
            train_uopsd(args.name, args.max_steps)
        if args.cmd == "run":
            _xevals(args.name, ck, args.max_steps, args.views)
            from .report import report
            print(report(args.name, args.max_steps))
        return
    if args.cmd == "xevals":
        _xevals(args.label, args.ckpt_dir, args.step, args.views)
        from .report import report
        print(report(args.label, args.step))
        return
    from .xeval import out_dir, xeval
    setup_logging(out_dir(args.label, args.view) / "logs" / "eval.log")
    logging.getLogger("votedistill.xfer").info(f"argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
    xeval(args.label, args.ckpt_dir, args.step, args.view)


if __name__ == "__main__":
    main()
