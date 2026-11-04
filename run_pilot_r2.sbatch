#!/bin/bash
#SBATCH --job-name=pilot_r2
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=/home/boyil2/xling_pilot/slurm-pilot_r2-%j.out
# Pilot R2 (non-thinking gold COPSD). Every stage is idempotent: resubmit until report.txt shows GO/NO-GO.
#   sbatch ~/xling_pilot/run_pilot_r2.sbatch
# Optional: sbatch --export=ALL,AUTO_RESUBMIT=1 ... resubmits itself (max 8 times) while work remains.
set -euo pipefail
cd /home/boyil2/xling_pilot
export HF_HOME=/data/user_data/boyil2/hf_cache
PY=/data/user_data/boyil2/envs/vllm_pilot/bin/python
V="--variant r2"
CK="step10 step20 step30 step40 step50 step60"

[[ -f pilot_r2_out/split.json ]] || cp pilot_r_out/split.json pilot_r2_out/split.json   # same DEV/TEST split as Pilot R
$PY pilot_r.py config $V
# loss-sign check must exist and be clean before training (losscheck exits non-zero if unclipped KL < 0)
if ! grep -q '"R2_nonthinking_base_step0"' pilot_r2_out/loss_check.json 2>/dev/null; then
  $PY pilot_r.py losscheck $V --label R2_nonthinking_base_step0
fi
$PY pilot_r.py eval $V --names base --split dev
$PY pilot_r.py train $V --max_steps 60
$PY pilot_r.py eval $V --names $CK --split dev
SEL=$($PY pilot_r.py selected $V | tail -1)
echo "selected checkpoint: $SEL"
if [[ "$SEL" != "NONE" ]]; then
  $PY pilot_r.py eval $V --names base "$SEL" --split test
fi
$PY pilot_r.py report $V

if grep -qE "=> (GO|NO-GO)" pilot_r2_out/report.txt; then
  echo "PILOT R2 COMPLETE"
elif [[ "${AUTO_RESUBMIT:-0}" == "1" ]]; then
  n=$(cat pilot_r2_out/.resubmits 2>/dev/null || echo 0)
  if (( n < 8 )); then echo $((n+1)) > pilot_r2_out/.resubmits; sbatch --export=ALL,AUTO_RESUBMIT=1 /home/boyil2/xling_pilot/run_pilot_r2.sbatch; fi
fi
