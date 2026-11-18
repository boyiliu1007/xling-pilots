#!/bin/bash
# usage: bench_submit.sh MODEL NAME [REPPEN]  -> one preempt A6000 job running logs/bench_eval.py
M=$1; N=$2; PEN=${3:-}
sbatch --parsable --partition=preempt --qos=preempt_qos --requeue --gres=gpu:A6000:1 --time=8:00:00 --cpus-per-task=8 --mem=64G \
  --exclude=babel-u9-20,babel-w9-28,babel-w9-26,babel-l9-32 --job-name=e2-bench-${N:0:60} --output=logs/%x-%j.out \
  --wrap="cd /home/jeffc2/ANLP_final/xling-pilots && BASE_MODEL=$M REPPEN=$PEN PYTHONPATH=. /data/user_data/jeffc2/venvs/votedistill/bin/python logs/bench_eval.py $N"
