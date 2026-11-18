#!/bin/bash
cd /home/jeffc2/ANLP_final/xling-pilots
export BASE_MODEL=Qwen/Qwen3-4B PY=/data/user_data/jeffc2/venvs/votedistill/bin/python PYTHONPATH=.
for L in eng hau; do for M in base grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s42 grpo__gsm8k-mt3k__exact__votelid-lr5e-6-q16-steps300__s42; do
  echo "$(date) START langeval $M $L"; $PY logs/lang_eval.py $M $L && echo "$(date) DONE $M $L" || echo "$(date) FAILED $M $L"
done; done
