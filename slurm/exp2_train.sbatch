#!/bin/bash
#SBATCH --job-name=exp2train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# One Experiment 2 training run, training only (resumable; a preempted job is requeued and resumes from the last
# checkpoint). Evaluation is a separate job (slurm/exp2_evalfinal.sbatch). Pick the GPU at submission, e.g.
#   sbatch --gres=gpu:RTX_PRO_6000:1 --export=ALL,NAME=L3-vote__copsd__exact__rp__s42,BASE_MODEL=Qwen/Qwen3-4B slurm/exp2_train.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model ${BASE_MODEL:-default} git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
$PY -m votedistill.exp2 train --name "${NAME:?set NAME}"
