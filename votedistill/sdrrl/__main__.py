"""Label-free SDRRL baseline. Stages write into one data directory and can be rerun independently.

Experiment 2 (Qwen3-4B, gsm8k-mt3k; --out defaults to results/exp2_qwen3-4b/sdrrl/gsm8k-mt3k, --votes to the model's
English thinking-mode consensus file):
  export BASE_MODEL=Qwen/Qwen3-4B
  python -m votedistill.sdrrl translate --data gsm8k-mt3k                        # NLLB: voted English traces -> Swahili
  COMET_PY -m votedistill.sdrrl.quality --out results/exp2_qwen3-4b/sdrrl/gsm8k-mt3k   # CometKiwi (separate env)
  python -m votedistill.sdrrl build --data gsm8k-mt3k                            # four language combinations,
                                                                                 # code-switching, parallel text
  python -m votedistill.exp2 train      --name sdrrl__gsm8k-mt3k__exact__s42     # SFT (exp2/runs.py, context sdrrl)
  python -m votedistill.exp2 eval-final --name sdrrl__gsm8k-mt3k__exact__s42     # TEST, default + penalty decoding
slurm/sdrrl.sbatch runs all of it.

COPSD set (the original setup): --data copsd (default) with an explicit --votes and --out, then
  python -m votedistill.sdrrl train --out results/sdrrl --ckpt_dir CKPT
  python -m votedistill.copsd eval --run_dir results/sdrrl --ckpt_dir CKPT --names base step10 ... --split dev

VOTES.jsonl is an English vote file: either `votedistill.consensus` records or one already-selected reference per
problem (see votedistill/sdrrl/references.py).
"""
import argparse
from pathlib import Path

from ..utils import setup_logging
from .translate import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(prog="python -m votedistill.sdrrl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="copsd", help="training questions: copsd, or an Experiment 2 set (gsm8k-mt3k)")
    common.add_argument("--out", default=None, help="data directory (default for Experiment 2 sets: "
                                                    "results/exp2[_<model>]/sdrrl/<data>; required for copsd)")
    common.add_argument("--lang", default="swa")

    t = sub.add_parser("translate", parents=[common])
    t.add_argument("--votes", default=None, help="English vote file (default for Experiment 2 sets: the consensus file)")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--min_share", type=float, default=0.625, help="minimum vote share (consensus records only)")
    t.add_argument("--pick", choices=["shortest", "longest"], default="shortest")
    t.add_argument("--limit", type=int, default=None, help="first N problems only (smoke test)")

    b = sub.add_parser("build", parents=[common])
    b.add_argument("--threshold", type=float, default=0.8, help="minimum mean CometKiwi score of a translated reference")
    b.add_argument("--swap_p", type=float, default=0.15, help="code-switch probability per dictionary word")
    b.add_argument("--n_parallel", type=int, default=100, help="natural parallel sentence pairs (SDRRL: 1,000 per 52k)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--no_qe", action="store_true", help="skip the CometKiwi filter (deviation; for smoke tests)")

    r = sub.add_parser("train", parents=[common])
    r.add_argument("--ckpt_dir", required=True)
    r.add_argument("--epochs", type=int, default=4)
    r.add_argument("--lr", type=float, default=5e-5)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--max_length", type=int, default=4096)
    r.add_argument("--save_every", type=int, default=10)
    r.add_argument("--max_steps", type=int, default=None, help="stop early (smoke test or a fixed step budget)")
    r.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exp2 = args.data != "copsd"
    if args.out is None:
        if not exp2:
            raise SystemExit("--out is required for --data copsd")
        from . import exp2_data_dir
        args.out = str(exp2_data_dir(args.data))
    setup_logging(Path(args.out) / "logs" / f"sdrrl_{args.cmd}.log")
    if args.cmd == "translate":
        from .translate import run
        votes, keep = args.votes, None
        if exp2:
            from . import exp2_votes
            from .build import question_rows
            votes = votes or str(exp2_votes(args.data))
            keep = set(question_rows(args.data, args.lang))     # only problems with a QC-passing Swahili question
        elif votes is None:
            raise SystemExit("--votes is required for --data copsd")
        if not Path(votes).exists():
            raise SystemExit(f"{votes} missing. Experiment 2 keeps the merged consensus file out of git; rebuild it with "
                             f"`python -m votedistill.consensus merge --out {Path(votes).parent} "
                             f"--names {Path(votes).stem} --k <shards>`")
        run(votes, args.out, args.lang, args.model, args.min_share, args.pick, args.limit, keep)
    elif args.cmd == "build":
        from .build import run
        run(args.out, args.lang, args.threshold, args.swap_p, args.n_parallel, args.seed, args.no_qe, args.data)
    else:
        from .train import train
        train(args.out, args.ckpt_dir, args.epochs, args.lr, args.batch, max_length=args.max_length,
              save_every=args.save_every, seed=args.seed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
