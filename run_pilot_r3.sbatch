#!/bin/bash
#SBATCH --job-name=pilot_r3
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=/home/boyil2/xling_pilot/slurm-pilot_r3-%j.out
# Pilot R3: re-evaluate Pilot R checkpoints at 4096 tokens (thinking). Idempotent: resubmit until report.txt has a verdict.
#   sbatch ~/xling_pilot/run_pilot_r3.sbatch
set -euo pipefail
cd /home/boyil2/xling_pilot
export HF_HOME=/data/user_data/boyil2/hf_cache
PY=/data/user_data/boyil2/envs/vllm_pilot/bin/python
V="--variant r3"
CK="step20 step40 step60"     # budget fallback (see config.json deviations)

[[ -f pilot_r3_out/split.json ]] || cp pilot_r_out/split.json pilot_r3_out/split.json
$PY pilot_r.py config $V
$PY pilot_r.py eval $V --names base --split dev
$PY pilot_r.py eval $V --names $CK --split dev
SEL=$($PY pilot_r.py selected $V --names $CK | tail -1)
echo "selected checkpoint: $SEL"
if [[ "$SEL" != "NONE" ]]; then
  $PY pilot_r.py eval $V --names base "$SEL" --split test
fi
$PY pilot_r.py report $V
grep -qE "=> (GO|NO-GO)" pilot_r3_out/report.txt && echo "PILOT R3 COMPLETE"
