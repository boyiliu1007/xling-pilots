# Cross-lingual self-distillation pilots (Qwen3-1.7B, African languages)

A set of small, pre-registered pilot experiments asking whether a model's **English** reasoning can serve as a teacher signal
for the **same model** on low-resource languages (Swahili, Yoruba, Hindi), and whether gold-reference
[COPSD](https://github.com/cisnlp/COPSD) (Crosslingual On-Policy Self-Distillation) improves Qwen3-1.7B on AfriMGSM Swahili.

Every pilot fixed its decision rule before seeing results. Each `pilot_*_out/report.txt` reports
numerator / denominator / estimate with 95% Wilson intervals, or paired-bootstrap CIs (10,000 resamples) for differences.
The raw per-sample generations behind every number are included.

All runs: one NVIDIA RTX A6000 (48 GB), `Qwen/Qwen3-1.7B` in bf16, seed 42.

## Results at a glance

| Pilot | Question | Pre-registered verdict |
|---|---|---|
| **A** `pilot_out/` | On AfriMGSM items where the swa/yor majority vote is wrong, is the English majority vote (8 samples) right? | **PASS** (swa, yor) |
| **C** `pilot_c_out/` | On Global-MMLU, is the EN–L gap option-order instability that permutation debiasing removes? | **FAIL** (sw, yo, hi) |
| **A2** `pilot_a2_out/` | Is English self-consensus precise on COPSD's actual training distribution? | COPSD set **NOT USABLE**; GSM8K train **USABLE** |
| **R** `pilot_r_out/` | Does gold COPSD (official hyperparameters) improve AfriMGSM-swa pass@12? Thinking mode, 1024-token eval | **NO-GO** |
| **R2** `pilot_r2_out/` | Same, but non-thinking mode for training and eval (2048-token eval) | **NO-GO** |
| **R3** `pilot_r3_out/` | Pilot R checkpoints re-evaluated with the COPSD paper protocol (thinking, 4096 tokens) | **NO-GO** (only on the formatted-accuracy clause) |

### Pilot A: English self-consensus on AfriMGSM (250 items, n = 8)
Majority-vote accuracy (maj@8): en 0.852, swa 0.084, yor 0.040.
On items where the L majority is wrong, the English consensus filtered at vote share ≥ 0.625 has
precision 0.910 [0.864, 0.942] with 0.921 coverage (swa), and 0.906 [0.860, 0.938] with 0.929 coverage (yor).
Caveat: the English rescue rate (0.85) equals overall English accuracy, because swa/yor are near the floor.
This used the dataset's human-parallel English questions, so it is an upper bound.

### Pilot C: MCQ option-order bias (Global-MMLU, 500 items, 4 cyclic shifts)
| | en | sw | yo | hi |
|---|---|---|---|---|
| debiased accuracy | 0.546 | 0.296 | 0.324 | 0.400 |
| order-sensitive items | 0.556 | 0.846 | 0.810 | 0.684 |
| debiasing gain (vs single order) | +0.032 | −0.024 | +0.022 | +0.018 |

Order sensitivity is higher in the target languages, but debiasing closes none of the EN–L gap; the gap is knowledge.
A "teacher view" (L question + English translation) is about as accurate as English.
Deviation: the assistant turn is prefilled with `The correct answer is: **`. Without it, letter-probability mass was 0.07–0.56.

### Pilot A2: English consensus on the training distribution (n = 8)
| Set | maj@8 | coverage @0.625 | precision @0.625 |
|---|---|---|---|
| COPSD train (500 OpenThoughts olympiad problems), non-thinking | 0.586 | 0.564 | 0.787 [0.736, 0.831] |
| COPSD train, thinking, 4096 tokens (100 items; 71% truncated) | 0.440 | 0.230 | 0.957 [0.790, 0.992] |
| GSM8K train (500) | 0.894 | 0.938 | 0.930 [0.903, 0.949] |

### Pilots R / R2 / R3: gold COPSD on AfriMGSM Swahili (DEV 50 / TEST 200 split, pass@12)
| | eval protocol | base TEST pass@12 | DEV-selected ckpt TEST pass@12 | Δ [95% CI] | format rate base → ckpt |
|---|---|---|---|---|---|
| R | thinking, 1024 tok, T = 1.0 | 0.145 | 0.165 (step 60) | +2.0 [−3.5, +7.5] | 0.20 → 0.55 |
| R2 | non-thinking, 2048 tok, T = 0.7 | 0.240 | 0.230 (step 60) | −1.0 [−6.5, +4.5] | 0.75 → 0.73 |
| R3 | thinking, 4096 tok, T = 1.0 | 0.150 | 0.245 (step 20) | **+9.5 [+4.0, +15.5]** | 0.23 → 0.81 |

- R3's base (15.0 on 200 items; 15.2 on all 250) matches COPSD's reported Swahili base for Qwen3-1.7B (14.8).
- R3 fails only the third clause of its rule: accuracy among formatted samples fell 0.120 → 0.079.
  That clause is composition-biased, because the base formats only its easy, short samples.
  Mean sample accuracy rose +3.6 pts [+2.2, +5.2].
- R2: the share of Swahili reasoning roughly halves after training (0.53 → 0.27).
- The official loss (forward KL(teacher‖student) with a per-(token, vocab) clip at 0.05) goes negative during training.
  `pilot_r2_out/loss_check.json` shows this comes entirely from the clip; the unclipped KL is never negative.
- `pilot_r_out/report.txt` also has an ORACLE section (every checkpoint on TEST, not used for decisions).
  Steps 20–40 reach +5 to +6.5 pts at 1024 tokens.

## Models

LoRA adapters (rank 64, alpha 128, all attention and MLP projections, fp32) on top of `Qwen/Qwen3-1.7B`,
for Pilot R (thinking) and Pilot R2 (non-thinking), at steps 10, 20, 30, 40, 50 and 60.
They are attached to the **v1.0 GitHub Release** as `pilot_r_stepXX.tar` and `pilot_r2_stepXX.tar`
(about 279 MB each), because they exceed GitHub's per-file limit.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "pilot_r_step20/adapter")   # after: tar xf pilot_r_step20.tar
```

Prompt formats differ between R and R2/R3. See `eval_prompt()` in `pilot_r.py`.
Optimizer and RNG states for resuming training are not released.

## Layout

| Path | Contents |
|---|---|
| `pilot.py`, `extract.py`, `stats.py` | Pilot A: vLLM generation, `\boxed{}` / numeric extraction, Wilson intervals |
| `pilot_c.py`, `mcq.py` | Pilot C: next-token letter scoring over cyclic option shifts |
| `pilot_a2.py`, `mathcheck.py` | Pilot A2: math-verify equivalence-class voting |
| `pilot_r.py` | Pilots R / R2 / R3: single-GPU COPSD trainer, eval, loss check, report (`--variant r\|r2\|r3`) |
| `run_*.sbatch` | Idempotent SLURM pipelines (resubmit until the report shows a verdict) |
| `test_*.py` | Unit tests (`pytest`) |
| `pilot_*_out/` | `report.txt`, `config.json` (hyperparameters, versions, every deviation), `run.log`, splits, per-item JSONL |

## Reproducing

```bash
pip install -r requirements.txt
git clone https://github.com/cisnlp/COPSD   # pilot_r.py imports its collator, language config and eval helpers
python -m pytest -q test_extract.py test_mcq.py test_mathcheck.py
python pilot.py --langs swa yor --n 8                         # Pilot A
python pilot_c.py                                             # Pilot C
python pilot_a2.py                                            # Pilot A2
sbatch run_pilot_r.sbatch                                     # Pilot R  (also: run_pilot_r2.sbatch, run_pilot_r3.sbatch)
```

Paths to the COPSD checkout, checkpoints and caches are hard-coded near the top of `pilot_r.py` and in the `.sbatch` files
(`/data/user_data/...`, `/home/...`); adjust them for your machine.
`VLLM_USE_FLASHINFER_SAMPLER=0` is set because the FlashInfer sampler needs a JIT toolchain.

## How Pilot R differs from the official COPSD setup

The official run uses TRL 0.26 GOLD + accelerate + vLLM 0.11 on 4 GPUs. Pilot R is a single-GPU reimplementation that
imports COPSD's data collator, prompts and evaluation helpers and copies its loss function.
It keeps the official hyperparameters: lr 5e-6, LoRA 64/128, effective batch 32, temperature 1.1, top_p 0.95, top_k 20,
max completion 2048, fixed teacher, beta 0, clip 0.05. Differences:

- 1 GPU × micro-batch 4 × 8 accumulation steps.
- 60 steps, on the official 100-step linear LR horizon.
- Student weights reach vLLM as a LoRA adapter after every step.
- SDPA attention instead of FlashAttention-2.
- A seeded, HF-like data order.
- Newer library versions.

The full list is under `deviations` in `pilot_r_out/config.json`.

## Data, models and attribution

- AfriMGSM (`masakhane/afrimgsm`, Apache-2.0); Global-MMLU (`CohereLabs/Global-MMLU`, Apache-2.0);
  GSM8K (`openai/gsm8k`, MIT); Qwen3-1.7B (Apache-2.0).
- COPSD training data (`yihongLiu/COPSD-AfriMGSM-TrainDataset`) and the COPSD code (https://github.com/cisnlp/COPSD)
  declare no license. `pilot_r.py` contains a verbatim copy of COPSD's `generalized_jsd_loss`, which extends TRL's
  Apache-2.0 GKD loss. Several outputs and the released adapters derive from the COPSD training set.
  All credit for COPSD belongs to Liu, Zhao, Hedderich and Schütze, *Crosslingual On-Policy Self-Distillation for
  Multilingual Reasoning* (arXiv:2605.09548).
- The result JSONLs contain benchmark question text and model generations.
  Please do not train on benchmark test items.
