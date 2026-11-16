#!/bin/bash
#SBATCH --job-name=xfer
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=logs/%x-%j.out
# Szu-Ju's runs (votedistill/xfer). Idempotent: training resumes from the last checkpoint, finished evaluations are
# skipped, so a preempted job is simply requeued.
#   NAME=uopsd__gsm8k-mt__clip__s42 sbatch slurm/xfer_run.sbatch                 # U-OPSD: train + 4-view eval
#   LABEL=L3-gold__copsd__exact__s42 CKPT=<dir with step-100/> sbatch slurm/xfer_run.sbatch   # eval an existing adapter
#   MAX_STEPS=2 VIEWS=enen ... : smoke test
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
VIEWS=${VIEWS:-swsw swen enen swsw-rp}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
if [[ -n "${NAME:-}" ]]; then
  $PY -m votedistill.xfer run --name "$NAME" --max_steps "${MAX_STEPS:-100}" --views $VIEWS
else
  $PY -m votedistill.xfer xevals --label "${LABEL:?set NAME or LABEL}" --ckpt_dir "${CKPT:?set CKPT}" --step "${STEP:-100}" --views $VIEWS
fi
echo "XFER_DONE ${NAME:-$LABEL}"
