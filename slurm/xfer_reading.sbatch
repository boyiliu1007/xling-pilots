#!/bin/bash
#SBATCH --job-name=xfer-reading
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|L40S|L40|6000Ada|RTX_PRO_6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=logs/%x-%j.out
# Explicit vs implicit reading (votedistill/xfer/reading.py), base model only. Idempotent (finished conditions are
# skipped), so a preempted job is requeued.   MODEL=Qwen/Qwen3-8B sbatch slurm/xfer_reading.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model $MODEL git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
$PY -m votedistill.xfer reading --model "${MODEL:?set MODEL}"
echo "XFER_READING_DONE $MODEL"
