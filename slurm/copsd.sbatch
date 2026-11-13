#!/bin/bash
#SBATCH --job-name=copsd
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=logs/%x-%j.out
# Train + evaluate one COPSD run. Submit from the repo root (mkdir -p logs first).
# Every stage is idempotent: resubmit until the report has a TEST section.
#   RUN=copsd_gold MAX_STEPS=100 sbatch slurm/copsd.sbatch
#   RUN=copsd_noref TRAIN_ARGS="--no_reference" sbatch slurm/copsd.sbatch
# Evaluation only (new protocol on existing checkpoints): EVAL_ONLY=1 CKPT_DIR=... RUN=... sbatch slurm/copsd.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
RUN=${RUN:?set RUN to the run name}
R=results/$RUN
CKPT_DIR=${CKPT_DIR:-/data/user_data/$USER/votedistill_ckpt/$RUN}
MAX_STEPS=${MAX_STEPS:-100}
EVAL_ARGS=${EVAL_ARGS:-}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"

CK=$(seq -f "step%g" 10 10 "$MAX_STEPS" | tr '\n' ' ')
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  $PY -m votedistill.copsd train --run_dir $R --ckpt_dir $CKPT_DIR --max_steps $MAX_STEPS ${TRAIN_ARGS:-}
fi
$PY -m votedistill.copsd eval --run_dir $R --ckpt_dir $CKPT_DIR --names base $CK --split dev $EVAL_ARGS
SEL=$($PY -m votedistill.copsd select --run_dir $R | tail -1)
echo "selected checkpoint: $SEL"
[[ "$SEL" != "NONE" ]] && $PY -m votedistill.copsd eval --run_dir $R --ckpt_dir $CKPT_DIR --names base "$SEL" --split test $EVAL_ARGS
$PY -m votedistill.copsd report --run_dir $R
