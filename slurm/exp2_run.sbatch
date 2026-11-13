#!/bin/bash
#SBATCH --job-name=exp2run
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
# One Experiment 2 training run: train (resumable), DEV eval of every checkpoint, TEST eval of the DEV-selected
# checkpoint (by maj@12), report. Every stage is idempotent: resubmit the same NAME to continue.
#   NAME=L3-gold__copsd__exact__s42 sbatch slurm/exp2_run.sbatch       # <context>__<data>__<layout>__s<seed>
# All evaluations run on A6000s, like L0 (vLLM sampling is hardware-dependent). On the preempt partition a
# preempted job is requeued and resumes from the last checkpoint / the first missing evaluation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
NAME=${NAME:?set NAME}
MS=${MAX_STEPS:+--max_steps $MAX_STEPS}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
$PY -m votedistill.exp2 train --name "$NAME" $MS
$PY -m votedistill.exp2 eval-run --name "$NAME" --split dev $MS
$PY -m votedistill.exp2 eval-run --name "$NAME" --split test $MS
$PY -m votedistill.exp2 run-report --name "$NAME" $MS > /dev/null
echo "RUN_DONE $NAME"
