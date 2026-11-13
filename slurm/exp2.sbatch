#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=logs/%x-%j.out
# Experiment 2, one teacher-context level. Submit from the repo root (mkdir -p logs first). Idempotent.
#   LEVEL=L0 sbatch slurm/exp2.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
LEVEL=${LEVEL:?set LEVEL, e.g. L0}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
$PY -m votedistill.exp2 eval --level $LEVEL --split dev test
$PY -m votedistill.exp2 report --level $LEVEL
