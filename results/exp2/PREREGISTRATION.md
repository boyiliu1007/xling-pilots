# Experiment 2 — analysis plan, fixed before any trained level is evaluated

Written 2026-11-12, before any Experiment 2 training run. Changes after this date are listed at the bottom with
their reason, so it stays clear which choices were made before seeing results.

## Question
What does the teacher transfer in crosslingual on-policy self-distillation (Qwen3-1.7B, Swahili), and does the
benefit of each kind of privileged information depend on the student's per-item capability?

## Levels (student always sees only the Swahili question)
| Level | Teacher sees | Role |
|---|---|---|
| L0 | — (no training) | baseline |
| L1 | + English question | value of the English question |
| L2 | + final answer only, in the reference slot | answer leakage (Zou et al. 2026 predict little transfer) |
| L3 | + full English reasoning trace | trace transfer (Experiment 1 when the trace is voted) |
| L3-shuffled | + trace of a different problem (fixed-seed derangement) | placebo for the content of the reference |

L2, L3 and L3-shuffled use the same COPSD reference slot and wording; only the content differs.
Gold variants (-gold) are upper bounds; label-free variants (-vote) use the model's own English majority vote.

## Evaluation (unchanged from the COPSD protocol)
AfriMGSM Swahili, fixed split (`results/splits/`): DEV 50 for checkpoint selection only, TEST 200 for reporting.
Thinking mode, n = 12, T = 1.0, top-p 0.95, 4096 new tokens, vLLM seed 42, COPSD's extraction and grader.

## Metrics
- **Primary: maj@12** on TEST (majority answer correct).
- Secondary: pass@12, mean per-sample accuracy.
- Mechanism: truncation rate, loop rate (an 8-word window repeated ≥ 5 times), format rate, accuracy among finished
  samples, share of Swahili text.
- Intervals: paired bootstrap over items (10,000 resamples, seed 42); with several seeds, report each seed and the
  seed-mean.

## Training and selection
- Same hyperparameters for every level (official COPSD, `votedistill/copsd/config.py`), same steps, same checkpoints
  (every 10 steps).
- Checkpoint selected by DEV **maj@12** (ties → earlier step). (The earlier gold-COPSD run selected by pass@12; it
  will be re-selected by maj@12 for comparison.)
- At least 2 training seeds per trained level.

## Capability groups (`votedistill/exp2/capability.py`)
From the base model only, no gold labels: L0 (Swahili, 12 samples) and L0-eng (English AfriMGSM, 12 samples) of the
same items. English confident = more than half of the 12 English samples give the English majority answer.
- A: English confident, Swahili majority ≠ English majority (the "access" gap; transfer should help most)
- B: English confident, Swahili majority = English majority
- C: English not confident
Group sizes are reported on DEV and TEST from base outputs alone. Hypothesis test: gain over L0 per level within each
group (paired bootstrap within group). With TEST 200, group-level comparisons are exploratory unless a group has
≥ 60 items; Yoruba is the planned extension for power.

## Pre-training screen (`votedistill/exp2/screen.py`)
For every teacher context, on the same student rollouts of the 500 COPSD training problems (2 per problem):
Zou et al.'s D (mean |log p_student − log p_teacher| of the sampled token) and the step-0 COPSD loss. Prediction to
test: the ranking of contexts by D predicts the ranking of TEST maj@12 gains (Spearman over levels; per item later).

## Open decisions (to be resolved before the label-free runs)
1. Training data for voted references: COPSD olympiad set (vote precision 0.79 at ≥ 5/8), machine-translated
   GSM8K questions (vote precision 0.93), or both.
2. Vote threshold for accepting a voted reference.
3. Whether to keep COPSD's right-padded teacher prompt layout in training (faithful to the official code:
   `multilingual_opsd_trainer.py` concatenates right-padded teacher prompts and generations with a pad-only mask and
   no position ids) or score each sequence unpadded. Measured on the screen rollouts (`padcheck_L3-gold.json`): for
   L3-gold, 75% of teacher prompts carry pads (mean 394, max 1378) and the teacher distribution moves by
   KL 0.0104 per token, against an L3-gold teacher signal of fkl 0.030; correct position ids remove ~80% of it.
4. Capability groups as defined above are nearly degenerate on AfriMGSM-swa: TEST A/B/C = 174/15/11 (DEV 42/4/4), and
   within A, 159 of 174 items have no Swahili sample matching the English answer. Only A reaches 60 items, so the
   planned group analysis has no power at the base model; an alternative capability axis is needed.

## Changes after writing
- 2026-11-12, before any result of a trained level (found by reading the code and checking prompts token by token):
  removed the level "L0+instr". COPSD's `language_config.py` ends with
  `for code, cfg in AFRICAN_LANGUAGE_CONFIG.items(): cfg["teacher_final_instruction"] = cfg["student_instruction"]`,
  so for Swahili the teacher prompt without English question and reference is token-identical to the student prompt:
  training it is a no-op, and L1 vs L0 differs only by the English question. The identical context is kept in the
  pre-training screen as a correctness check ("L0-same", D must be ~0).
