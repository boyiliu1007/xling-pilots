#!/bin/bash
#SBATCH --job-name=pilot_a2_rest
#SBATCH --partition=neulab-mini
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=babel-w9-28,babel-w9-26,babel-l9-32
#SBATCH --output=/home/boyil2/xling_pilot/slurm-a2-%j.out
# Continue Pilot A2 with the runs that have no JSONL yet, then rebuild the report.
set -euo pipefail
cd /home/boyil2/xling_pilot
export HF_HOME=/data/user_data/boyil2/hf_cache
PY=/data/user_data/boyil2/envs/vllm_pilot/bin/python
RUNS=()
# Backfill run_stats.json for runs finished by the first (interrupted) process, which predates run_stats.json.
$PY - <<'EOF'
import json, re, pathlib
p = pathlib.Path("pilot_a2_out/run_stats.json")
st = json.loads(p.read_text()) if p.exists() else {}
for line in open("a2_full.log"):
    m = re.search(r"INFO (R1_copsd_en|R2_gsm8k_en|R3_copsd_en_thinking|L_copsd_swa|L_copsd_yor): ([\d.]+)s, (\d+) tokens", line)
    if m and m.group(1) not in st:
        st[m.group(1)] = {"generation_s": float(m.group(2)), "generated_tokens": int(m.group(3)),
                          "items": 100 if m.group(1).startswith("R3") else 500,
                          "host": "babel-w9-20", "slurm_job_id": "11037482", "note": "backfilled from a2_full.log"}
p.write_text(json.dumps(st, indent=2)); print("run_stats:", {k: v.get("generation_s") for k, v in st.items()})
EOF
declare -A EXPECT=([R1_copsd_en]=500 [R2_gsm8k_en]=500 [R3_copsd_en_thinking]=100 [L_copsd_swa]=500 [L_copsd_yor]=500)
for r in R1_copsd_en R2_gsm8k_en R3_copsd_en_thinking L_copsd_swa L_copsd_yor; do
  f=pilot_a2_out/$r.jsonl
  if [[ -f $f && $(wc -l < $f) -eq ${EXPECT[$r]} ]]; then continue; fi
  RUNS+=("$r")
done
echo "remaining runs: ${RUNS[*]:-none}"
if (( ${#RUNS[@]} )); then
  $PY pilot_a2.py --runs "${RUNS[@]}"
else
  $PY pilot_a2.py --report_only
fi
