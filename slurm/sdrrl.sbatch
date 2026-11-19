#!/bin/bash
#SBATCH --job-name=sdrrl
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
# SDRRL baseline in Experiment 2 (PREREGISTRATION 2026-11-19 SDRRL entry): data stages -> SFT -> selection-free TEST.
#   sbatch --export=ALL,BASE_MODEL=Qwen/Qwen3-4B,SEED=42 slurm/sdrrl.sbatch
# Data stages (shared by all seeds, results/exp2_<model>/sdrrl/<DATA>/): merge the English consensus shards if needed ->
# NLLB-200-3.3B translation of the voted English traces -> CometKiwi (separate env) -> build the SFT set. Then
# `exp2 train` and `exp2 eval-final` (TEST of the final checkpoint, default and repetition-penalty-1.1 decoding), the
# same evaluation as every other Qwen3-4B run. Every stage skips finished work: on preemption the job requeues and
# continues (training resumes from the last checkpoint; an interrupted translation restarts).
# Optional: DATA (gsm8k-mt3k), SEED (42), VARIANT (e.g. ep2 -> sdrrl__<DATA>__exact__ep2__s<SEED>),
#           BUILD_ARGS (default "--n_parallel 200"), SHARDS (12: consensus shards to merge), PY, COMET_PY,
#           STAGES (all | data: stop after building the SFT set).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d votedistill ]] || { echo "submit from the repository root"; exit 1; }
PY=${PY:-/data/user_data/$USER/envs/vllm_pilot/bin/python}
COMET_PY=${COMET_PY:-/data/user_data/$USER/envs/comet/bin/python}   # separate env: uv pip install unbabel-comet "setuptools<81"
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export HF_HOME=${HF_HOME:-/data/user_data/$USER/hf_cache}            # the comet env does not import votedistill.paths
DATA=${DATA:-gsm8k-mt3k}
NAME=sdrrl__${DATA}__exact${VARIANT:+__$VARIANT}__s${SEED:-42}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model $BASE_MODEL run $NAME git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- votedistill slurm || echo '+dirty')"

D=$($PY -c "from votedistill.sdrrl import exp2_data_dir; print(exp2_data_dir('$DATA'))")
VOTES=$($PY -c "from votedistill.sdrrl import exp2_votes; print(exp2_votes('$DATA'))")
if [[ ! -f $VOTES ]]; then
  $PY -m votedistill.consensus merge --out "$(dirname "$VOTES")" --names "$(basename "$VOTES" .jsonl)" --k "${SHARDS:-12}"
fi
[[ -f $D/translations.jsonl ]] || $PY -m votedistill.sdrrl translate --data "$DATA"
[[ -f $D/qe_scores.jsonl ]]    || $COMET_PY -m votedistill.sdrrl.quality --out "$D"
[[ -f $D/sft_data.jsonl ]]     || $PY -m votedistill.sdrrl build --data "$DATA" ${BUILD_ARGS:---n_parallel 200}
if [[ ${STAGES:-all} == data ]]; then   # data stages only: reveal no result (e.g. before the prediction is preregistered)
  echo "DATA_DONE $D"; exit 0
fi

$PY -m votedistill.exp2 train --name "$NAME"
$PY -m votedistill.exp2 eval-final --name "$NAME"
echo "RUN_DONE $NAME"
