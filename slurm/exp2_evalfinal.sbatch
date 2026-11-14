#!/bin/bash
#SBATCH --job-name=exp2evalfinal
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
# Selection-free evaluation of one run on A6000 (like every other evaluation): TEST of the final checkpoint with
# default decoding and with repetition penalty 1.1. Idempotent (finished evaluations are skipped).
#   sbatch --export=ALL,NAME=...,BASE_MODEL=Qwen/Qwen3-4B slurm/exp2_evalfinal.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/venvs/votedistill/bin/python}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model ${BASE_MODEL:-default} git $(git rev-parse --short HEAD)"
$PY -m votedistill.exp2 eval-final --name "${NAME:?set NAME}"
