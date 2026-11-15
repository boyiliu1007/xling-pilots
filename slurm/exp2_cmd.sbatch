#!/bin/bash
#SBATCH --job-name=exp2cmd
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Any `python -m <module> ...` step on one GPU (resumable steps only: a preempted job is requeued and reruns CMD), e.g.
#   sbatch --gres=gpu:A6000:1 --export=ALL,CMD="votedistill.exp2 translate --data gsm8k-mt3k" slurm/exp2_cmd.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model ${BASE_MODEL:-default} git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"
$PY -m ${CMD:?set CMD}
