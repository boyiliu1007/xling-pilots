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
- 2026-11-13, before any trained level was evaluated (decisions with the user):
  - Padding (open decision 3): two L3-gold runs, `official` layout (faithful to COPSD) and `exact` layout (pads
    stripped before the completion is appended; `copsd/train.py:LAYOUTS`). All other runs use `exact`.
  - Training data (open decision 1): all three, as separate runs of L3-vote: `copsd`, `gsm8k-mt` (500 GSM8K *train*
    questions, NLLB-200 3.3B sentence-level translation, number-preservation QC; `exp2/translate.py`) and `mix`
    (half each). To compare data and not data size, every vote-context dataset is subsampled to the same size
    N = min(accepted copsd, accepted gsm8k-mt), seed 42 (`exp2/trainsets.py`).
  - Vote threshold (open decision 2): the proposal's 5/8 (unique majority), English thinking-mode consensus, n = 8,
    T = 0.6, 16384 tokens. Rejected questions are dropped. Reference = text after `</think>` of the first finished
    sample agreeing with the majority (matches the format of COPSD's gold `solution`, a worked solution, not a trace).
  - Steps: the official 100 (the earlier gold run used 60), checkpoints every 10, DEV maj@12 selection over all ten.
  - Training seed is now a parameter (data order, LoRA init, sampling); first seed 42, second seed after the first
    round. Evaluation sampling seed stays 42 for every model.
  - Capability axis (open decision 4): undecided. Candidate (gold-free, varies across items): number of the 12 L0
    samples that finish (do not hit the 4096-token limit). TEST distribution 0: 38, 1-2: 74, 3-5: 57, 6-12: 31 items;
    L0 pass@12 rises 0.00 / 0.08 / 0.21 / 0.39 across these bins.
- 2026-11-13, after seeing DEV curves but before any TEST result: DEV maj@12 barely moves across checkpoints
  (0.08-0.14 for every run, base 0.12; one item = 0.02), so DEV maj@12 selection is close to random. The primary
  analysis stays as written (DEV maj@12 selection); added a selection-free analysis: every run is also evaluated on
  TEST at its final checkpoint (step 100), and all contrasts are reported for both.
- 2026-11-13 ~15:30, loss-clip ablation (decided with the user after the D screen, before any ablation TEST result):
  runs `<ctx>__copsd__exact__{noclip,clip0.5}__s{42,43}` for L1 and L3-gold, plus L3-shuffled-gold without clip.
  Contrasts in `summary.py` ("no clip: ...", "clip 0.5: ...", "clip effect ..."). Clip diagnostics:
  `exp2/clipstats.py`, results/exp2/screen/*/clipstats.txt.
- 2026-11-13 ~16:00, two-stage gold-free experiment (written before any TEST result of the clip ablation or of these
  runs; known at the time: TEST of the official-clip grid and of 3 L3-vote runs, the clip diagnostics, and the
  *training-rollout* curves of the ablation runs, in which clipped runs learn to finish, boxed 0.28-0.40 at steps
  31-60, and unclipped / clip-0.5 runs do not, 0.09-0.19).
  Hypothesis: the official clip produces COPSD's gain on Swahili (learning to terminate) and blocks the teacher's
  information; stage 1 with the clip teaches termination, stage 2 without it transfers information.
  Everything is gold-free except evaluation: the teacher reference is the English majority vote (L3-vote), the
  self-SFT filter is agreement with that vote, and the **primary analysis uses the fixed final checkpoint (step 100;
  self-SFT alone: step 30)**, since DEV selection uses gold labels (DEV-selected numbers are reported as secondary).
  - Switch step N = 30, by a gold-free rule fixed now: the first checkpoint after which the boxed rate of the clipped
    runs' training rollouts stops rising (L3-vote/L3-gold copsd s42/s43, 10-step windows: 0.20-0.22, 0.32-0.38,
    0.50-0.57, then 0.46-0.54 and lower).
  - Stage 2 continues the stage-1 run (adapter, AdamW state, LR schedule, RNG) to step 100 with the arm's context
    and clip; data copsd (vote-accepted rows for vote contexts), exact layout, seeds 42 and 43.
  - Arms: A (method) stage 1 = clipped L3-vote 30 steps, stage 2 no clip with L3-vote / L1 / L3-shuffled-vote;
    A-oracle the same with L3-gold in both stages; B = plain clipped L3-vote 100 steps (existing runs); C = L3-vote
    without clip from step 0; D stage 1 = vote-filtered self-SFT 30 steps (`exp2/selfsft.py`: base model, 8 samples
    per vote-accepted question with the student prompt and training sampling, keep finished answers equivalent to
    the vote, at most 2 per question; NLL with the COPSD optimizer, batch and schedule), then stage 2 as A (L3-vote);
    D-alone = the self-SFT checkpoint at step 30.
  - Primary contrasts (TEST, step 100, paired bootstrap, per seed and pooled over seeds): (1) A L3-vote - A placebo,
    (2) A L3-vote - A L1, (3) A L3-vote - B, (4) A L3-vote - A-oracle; secondary: A - C, D - A, D-alone - B.
    Metrics as before (maj@12 primary, pass@12, mean_acc), plus the mechanism table (truncation, loops, accuracy of
    finished answers, distinct-word degeneration measure, language).
  - Degeneration measure added to the mechanism table (calibrated on existing step-100 outputs before these runs):
    truncated and (verbatim loop or distinct-word share of the last 300 words < 0.3); the verbatim loop detector alone
    misses the word salad of L1 runs.
