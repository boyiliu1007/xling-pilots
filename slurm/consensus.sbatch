#!/bin/bash
#SBATCH --job-name=consensus
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=/home/boyil2/xling_pilot/logs/%x-%j.out
# Majority-vote sampling. Finished runs are skipped, so resubmitting resumes.
#   OUT=results/consensus_train_thinking ARGS="--runs copsd:en gsm8k:en --thinking --max_tokens 4096" sbatch slurm/consensus.sbatch
set -euo pipefail
cd /home/boyil2/xling_pilot
PY=${PY:-/data/user_data/boyil2/envs/vllm_pilot/bin/python}
$PY -m votedistill.consensus generate --out ${OUT:?set OUT} ${ARGS:?set ARGS}
