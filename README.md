# Cross-lingual self-distillation on AfriMGSM (Qwen3-1.7B)

Code and first results for *Teaching Low-Resource Languages with a Model's Own English Understanding*:
transfer Qwen3-1.7B's English math reasoning to Swahili (and Yoruba) with crosslingual on-policy self-distillation
([COPSD](https://github.com/cisnlp/COPSD)), replacing COPSD's gold English reference with the model's own
English majority vote.

Two tools:

| Tool | What it does | Proposal |
|---|---|---|
| `python -m votedistill.consensus` | Samples n answers per question, majority-votes, and scores the vote against gold | Teacher reliability (§6.3); the label-free reference for Experiment 1; student-capability signals for Experiments 2–3 |
| `python -m votedistill.copsd` | Single-GPU COPSD trainer, AfriMGSM evaluation with COPSD's own grader, report | Gold-COPSD baseline (§5.1); Experiments 1–3 change what the teacher sees |

All runs so far: one NVIDIA RTX A6000 (48 GB), `Qwen/Qwen3-1.7B` in bf16, seed 42.

## Results so far

### English consensus (earlier non-thinking runs; raw outputs removed)
These votes used non-thinking mode, while the proposal's teacher follows COPSD's thinking mode, so their raw outputs
were removed and only the summary numbers are kept here as motivation. Re-run with
`python -m votedistill.consensus generate --thinking ...` before relying on them; the earlier code is in git history (4e5abab).

AfriMGSM (n = 8, 250 test items): majority-vote accuracy English 0.852, Swahili 0.084, Yoruba 0.040.
On items where the Swahili majority is wrong, the English majority at vote share ≥ 5/8 covers 0.921 of them with
precision 0.910 [0.864, 0.942] (Yoruba: 0.929, 0.906). This used AfriMGSM's human-parallel English (an upper bound)
and included the TEST items, so it is a diagnostic only.

Training data (n = 8):

| Set | maj@8 | coverage at ≥ 5/8 | precision at ≥ 5/8 |
|---|---|---|---|
| COPSD train (500 OpenThoughts olympiad problems) | 0.586 | 0.564 | 0.787 [0.736, 0.831] |
| GSM8K train (500-item sample) | 0.894 | 0.938 | 0.930 [0.903, 0.949] |

**Open issue for Experiment 1:** on COPSD's own training set, about 1 in 5 voted references would be wrong at the
proposal's 5/8 threshold, and even unanimous votes are only 0.889 precise. GSM8K-style problems give a much cleaner
signal but have no Swahili translations here. In thinking mode with 4096 tokens, 71% of samples were truncated on the
COPSD set, so the token budget for thinking-mode voting is also open.

### Gold-COPSD baseline — `results/copsd_gold/` (training), `results/copsd_gold_eval4096/` (evaluation)
60 training steps with the official hyperparameters, thinking mode. Evaluated with the COPSD paper protocol
(thinking, n = 12, T = 1.0, 4096 new tokens) on a fixed split of AfriMGSM Swahili: DEV 50 items for checkpoint
selection, TEST 200 items (`results/splits/`).

| TEST, base → step 20 (DEV-selected) | Δ [95% paired-bootstrap CI] |
|---|---|
| pass@12: 0.150 → 0.245 | **+9.5 [+4.0, +15.5]** |
| maj@12: 0.090 → 0.105 | +1.5 [−2.5, +5.5] |
| mean per-sample accuracy: 0.028 → 0.064 | +3.6 [+2.2, +5.2] |
| answer format rate: 0.23 → 0.81 | +57.4 |

- The base score (15.2 pass@12 on all 250 items) matches COPSD's reported 14.8, so the setup reproduces.
- The gain is mostly the model learning to finish and box its answer. Under majority vote it is not significant.
  **Decide on one headline metric before running the proposal's experiments.**
- One seed, 60 rather than 100 steps, and DEV selection only among steps 20, 40, 60 (compute budget).
  Treat this as a sanity check, not the final baseline.
- The training loss goes negative because of the official per-(token, vocab) clip at 0.05; `loss_check.json` shows
  the unclipped KL is never negative.

## Layout

| Path | Contents |
|---|---|
| `votedistill/answers_numeric.py` | `\boxed{}` / last-number extraction and numeric majority vote (AfriMGSM, GSM8K) |
| `votedistill/answers_math.py` | math-verify checking and equivalence-class voting, mirroring COPSD's grader (olympiad answers) |
| `votedistill/data.py` | AfriMGSM, COPSD training set and GSM8K loaders; AfriMGSM row-alignment check; the DEV/TEST split |
| `votedistill/consensus.py` | Majority-vote sampling and its report |
| `votedistill/copsd/` | `collator.py` (student/teacher prompts: **where teacher-context experiments go**), `loss.py`, `train.py`, `evaluate.py`, `report.py`, `losscheck.py`, `config.py` (official hyperparameters and every deviation) |
| `votedistill/stats.py`, `gpu.py`, `utils.py`, `paths.py` | Wilson and paired-bootstrap intervals; GPU wait and memory polling; I/O; machine-specific paths |
| `slurm/` | `copsd.sbatch` (train + DEV/TEST eval + report) and `consensus.sbatch` |
| `tests/` | Answer extraction and voting tests (`pytest tests`) |
| `results/<run>/` | `report.txt`, `config.json`, per-item JSONL with every generation |

## Running

```bash
pip install -r requirements.txt               # e.g. uv venv --python 3.11 /data/user_data/$USER/venvs/votedistill
git clone https://github.com/cisnlp/COPSD /data/user_data/$USER/ext/COPSD && git -C /data/user_data/$USER/ext/COPSD checkout 4f4410730e9b
# defaults live under /data/user_data/$USER (COPSD_DIR, CKPT_ROOT, HF_HOME); override via env vars, see votedistill/paths.py
# Slurm scripts: submit from the repo root after `mkdir -p logs`
pytest tests

# majority votes (finished runs are skipped)
python -m votedistill.consensus generate --out results/consensus_train_thinking --runs copsd:en gsm8k:en --thinking --max_tokens 4096
python -m votedistill.consensus report --out results/consensus_train_thinking

# COPSD: train, evaluate, select on DEV, report
RUN=copsd_gold MAX_STEPS=100 sbatch slurm/copsd.sbatch
RUN=copsd_noref TRAIN_ARGS="--no_reference" sbatch slurm/copsd.sbatch   # teacher sees the English question only (L1)
python -m votedistill.copsd report --run_dir results/copsd_gold_eval4096
```

Each `results/` directory pins its settings: a COPSD run directory refuses evaluations whose sampling differs from
its `eval_config.json`, so a new protocol needs a new `--run_dir` (pointing `--ckpt_dir` at existing checkpoints).

## Where the proposal's experiments plug in

- **Experiment 1 (voted reference):** pick an agreeing English trace per problem from a `consensus` run, write it into
  each training row's `solution` in `votedistill/copsd/collator.py:prepare_rows`, and train as usual.
- **Experiment 2 (L0–L3):** lives in `votedistill/exp2/` (results in `results/exp2/`; analysis plan fixed in
  `results/exp2/PREREGISTRATION.md`). Levels and teacher contexts are in `votedistill/exp2/levels.py`: L2, L3 and the
  L3-shuffled placebo all fill the same COPSD reference slot (answer only / full trace / another problem's trace).
  ```bash
  PY=/data/user_data/$USER/venvs/votedistill/bin/python
  $PY -m votedistill.exp2 eval --level L0 --split dev test && $PY -m votedistill.exp2 report --level L0
  $PY -m votedistill.exp2 eval --level L0-eng --split dev test && $PY -m votedistill.exp2 report --level L0-eng  # capability groups
  $PY -m votedistill.exp2 screen --name copsd_k2 --contexts L0-same L1 L2-gold L3-gold L3-shuffled-gold         # pre-training D screen
  $PY -m votedistill.exp2 screen-report --name copsd_k2
  LEVEL=L0 sbatch slurm/exp2.sbatch
  ```
  Note: for the African languages COPSD's `language_config.py` sets `teacher_final_instruction = student_instruction`,
  so a teacher with neither English question nor reference has exactly the student's prompt (training it is a no-op).
- **Experiment 3 (adaptive):** per-problem choice of the above, driven by consensus vote shares.

## Models

LoRA adapters of the gold-COPSD run (rank 64, alpha 128, all attention and MLP projections) at steps 10–60 are
attached to the **v1.0 GitHub Release** as `pilot_r_stepXX.tar` (about 279 MB each).

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "pilot_r_step20/adapter")   # after: tar xf pilot_r_step20.tar
```

## Data, models and attribution

- AfriMGSM (`masakhane/afrimgsm`, Apache-2.0); GSM8K (`openai/gsm8k`, MIT); Qwen3-1.7B (Apache-2.0).
- COPSD training data (`yihongLiu/COPSD-AfriMGSM-TrainDataset`) and the COPSD code (https://github.com/cisnlp/COPSD)
  declare no license. `votedistill/copsd/loss.py` is a verbatim copy of COPSD's `generalized_jsd_loss`, which extends TRL's
  Apache-2.0 GKD loss. Several outputs and the released adapters derive from the COPSD training set.
  All credit for COPSD belongs to Liu, Zhao, Hedderich and Schütze, *Crosslingual On-Policy Self-Distillation for
  Multilingual Reasoning* (arXiv:2605.09548).
- The result JSONLs contain benchmark question text and model generations. Do not train on benchmark test items.
