#!/bin/bash
#SBATCH --job-name=pilot_r
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=/home/boyil2/xling_pilot/slurm-pilot_r-%j.out
# Pilot R pipeline. Every stage is idempotent: resubmit this script until report.txt shows a GO/NO-GO verdict.
#   sbatch ~/xling_pilot/run_pilot_r.sbatch
# Optional: sbatch --export=ALL,AUTO_RESUBMIT=1 ... resubmits itself (max 8 times) while work remains.
set -euo pipefail
cd /home/boyil2/xling_pilot
export HF_HOME=/data/user_data/boyil2/hf_cache
PY=/data/user_data/boyil2/envs/vllm_pilot/bin/python
CK="step10 step20 step30 step40 step50 step60"

$PY pilot_r.py split
$PY pilot_r.py config
$PY pilot_r.py eval --names base --split dev
$PY pilot_r.py eval --names base --split test
$PY pilot_r.py train --max_steps 60
$PY pilot_r.py eval --names $CK --split dev
SEL=$($PY pilot_r.py selected | tail -1)
echo "selected checkpoint: $SEL"
if [[ "$SEL" != "NONE" ]]; then
  $PY pilot_r.py eval --names "$SEL" --split test
  $PY pilot_r.py report
  $PY pilot_r.py eval --names $CK --split test      # ORACLE only; not used for the decision
fi
$PY pilot_r.py report

if grep -qE "=> (GO|NO-GO)" pilot_r_out/report.txt && ls pilot_r_out/eval/step60_test.jsonl >/dev/null 2>&1; then
  echo "PILOT R COMPLETE"
elif [[ "${AUTO_RESUBMIT:-0}" == "1" ]]; then
  n=$(cat pilot_r_out/.resubmits 2>/dev/null || echo 0)
  if (( n < 8 )); then echo $((n+1)) > pilot_r_out/.resubmits; sbatch --export=ALL,AUTO_RESUBMIT=1 /home/boyil2/xling_pilot/run_pilot_r.sbatch; fi
fi
