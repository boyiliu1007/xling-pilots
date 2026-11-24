"""PASMR data stage. Training and evaluation go through Experiment 2's run commands:

  export BASE_MODEL=Qwen/Qwen3-4B
  python -m votedistill.pasmr build --data gsm8k-mt3k --sdrrl_dir <folder with SDRRL's translations.jsonl, qe_scores.jsonl>
  python -m votedistill.exp2 train      --name pasmr-pam__gsm8k-mt3k__exact__s42     # stage 1, PAM SFT
  python -m votedistill.exp2 train      --name pasmr__gsm8k-mt3k__exact__s42         # stage 2, SRL from PAM
  python -m votedistill.exp2 eval-final --name pasmr__gsm8k-mt3k__exact__s42         # TEST, default + penalty 1.1
On the cluster: slurm/exp2_train.sbatch for each stage (PAM, then SRL with --dependency on PAM) and
slurm/exp2_evalfinal.sbatch for each run's TEST evaluation.
"""
import argparse
from pathlib import Path

from ..utils import setup_logging


def main():
    ap = argparse.ArgumentParser(prog="python -m votedistill.pasmr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", default="gsm8k-mt3k")
    b.add_argument("--sdrrl_dir", default=None, help="SDRRL data folder (default: this checkout's sdrrl data folder)")
    b.add_argument("--threshold", type=float, default=0.8, help="minimum mean CometKiwi score (as SDRRL)")
    args = ap.parse_args()
    from . import data_dir
    from .data import build
    setup_logging(data_dir(args.data) / "logs" / "build.log")
    if args.sdrrl_dir is None:
        from ..sdrrl import exp2_data_dir
        args.sdrrl_dir = exp2_data_dir(args.data)
    if not (Path(args.sdrrl_dir) / "translations.jsonl").exists():
        raise SystemExit(f"{args.sdrrl_dir} has no translations.jsonl: run SDRRL's translate and quality stages first")
    build(args.data, args.sdrrl_dir, args.threshold)


if __name__ == "__main__":
    main()
