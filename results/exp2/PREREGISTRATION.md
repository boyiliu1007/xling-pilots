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
- 2026-11-13 ~21:00, deviation (decided with the user after seeing the step-100 TEST results of arm A, A-oracle and
  part of the clip ablation): arm D stage 2 (`L3-vote__copsd__exact__s1sft30-noclip__s{42,43}`) cancelled before it
  started. Reason: unclipped training was harmful in every completed run (no clip vs clip, L3-gold: pass@12 -17.2;
  two-stage vs plain COPSD: -9.8), and arm A's stage 2 undid stage 1, so the arm could not answer its question and
  the compute was better spent elsewhere. D-alone (self-SFT, step 30) is kept. Contrast "stage 1: self-SFT - clipped
  COPSD (then no clip)" is therefore not reported.
- 2026-11-13 ~23:45, mechanism follow-ups (before any result of these runs; known: the two-stage and clip-ablation
  TEST results, the token-level mechanism diagnostic `exp2/mechanism.py`, teacher generation `exp2/teachergen.py`).
  Hypothesis: the official clip acts as an entropy regularizer (clipped loss = m x forward KL to the teacher with its
  strongly preferred tokens removed; clipped checkpoints: student entropy +0.70 nats, loop escape x12, p(</think>)
  x70-100 on the base model's text), and this, not the teacher's information, produces COPSD's Swahili gain.
  1. Think in English (copsd/styles.py; Swahili question, English instruction and COPSD's English thinking prefix):
     base-model levels L0-enthink-sw (answer in Swahili), L0-enthink-en, L0-selftrans-sw (translate the question
     first); COPSD with English thinking for student, teacher and evaluation: L1 and L3-vote (official clip) and
     L3-vote without clip, s42/s43. Prediction: the teacher's information can transfer only when the student's text
     is coherent, i.e. L3-vote > L1 here although not with Swahili thinking.
  2. Translation ability of the model itself (exp2/translation.py), before any translate-own-reasoning pipeline.
  3. Teacher-free controls on the official Swahili prompt, s42/s43, no clip: hot-self distillation (teacher = the
     base model on the student's prompt at temperature 1.5 or 2.5) and unlikelihood on verbatim 8-token loops; and
     decoding-only levels (sampling temperature 1.3 / 1.5, repetition penalty 1.1 / 1.3). Prediction: if the clip
     is an entropy regularizer, hot-self (and perhaps higher temperature) reaches COPSD's pass@12 gain over L0
     (+7 to +12 at step 100) without any teacher information. Primary comparison: step-100 TEST, each control - L0
     and each control - clipped COPSD L3-gold (contrasts in summary.py).
- 2026-11-14 ~01:15, decoding x training follow-up. Post hoc: motivated by an observed result, L0-reppen1.1 (base
  model, repetition penalty 1.1, no training) TEST pass@12 0.285 / mean_acc 0.081 / truncation 0.018, above every
  COPSD run (best step-100 mean_acc 0.052). Also known at this point: L0-reppen1.3 and L0-temp1.3/1.5 DEV only,
  L0-selftrans-sw TEST (pass@12 0.24, but answers mostly English), translation ability (EN->SW unusable), self-SFT s42.
  1. Dose curve: base-model levels L0-reppen1.05, L0-reppen1.2 (with 1.1 and 1.3 already queued).
  2. Does COPSD stack with the penalty? `exp2 decode-eval`: TEST at step 100 of L3-gold__copsd__exact and
     L3-vote__copsd__exact (s42/s43) and L3-gold__copsd__official (s42/s43), sampled with repetition penalty 1.1.
     Comparisons: vs the same checkpoint with default decoding, and vs the base model with penalty 1.1 (L0-reppen1.1).
     The second is the one that matters: does training add anything once decoding already prevents loops?
     No prediction committed (post hoc); reported descriptively, all runs listed, no selection.
  Caveat for every repetition-penalty result: vLLM's penalty also counts prompt tokens, and 1.1 was chosen a priori
  but the follow-up exists because it worked; the dose curve is the check against a lucky single setting.
- 2026-11-14 ~01:40, bigger models (requested by the user; no result of these runs known). Base models of the same
  family, no training, same TEST/DEV items, prompt and sampling as L0 (n 12, T 1.0, 4096 tokens): Qwen3-4B and
  Qwen3-8B at default decoding and with repetition penalty 1.1 (levels L0-qwen3-{4b,8b}[-reppen1.1]). Questions:
  is the Swahili looping specific to 1.7B, does the penalty still help a bigger model, and how does Qwen3-1.7B
  with the penalty compare with a 2.4x / 4.7x bigger model. Expectation (not a test): bigger models loop less and
  gain less from the penalty. Descriptive; no checkpoint selection involved.
- 2026-11-14 ~10:00, Qwen3-4B and penalty-rollout self-distillation (requested by the user: a gold-free, on-policy
  method that keeps Swahili output while breaking loops; written before any result of these runs). Known: all
  Qwen3-1.7B results above, base Qwen3-4B/8B with and without repetition penalty 1.1.
  Hypothesis: the teacher's information has not transferred in any setting because the teacher scores degenerate
  student text; if the student's own rollouts are coherent, it can. Method "rp" (penalty-rollout self-distillation):
  COPSD's trainer unchanged except that the student's on-policy rollouts are sampled with repetition penalty 1.1
  (vLLM; evaluation decoding is separate). Teacher = the same model with its *own* English majority-vote solution
  (L3-vote; gold-free). Second candidate "penalty-decoded self-SFT" (selfsft-vote ... __rp): the model's own Swahili
  answers sampled with penalty 1.1, kept if finished, in Swahili, and equal to its English majority vote; 30 SFT steps.
  Grid, seeds 42/43, 100 steps (self-SFT 30), official hyperparameters otherwise:
    Qwen3-4B: L3-vote, L1, L3-vote no clip (default rollouts); L3-vote rp, L1 rp, L3-vote rp-noclip; hot-self τ1.5
              (teacher-free control); penalty-decoded self-SFT. Qwen3-4B's voted references come from its own English
              thinking consensus (same settings as the 1.7B consensus: 500 COPSD items, n 8, T 0.6, 16384 tokens,
              vote threshold 5/8), run in 5 shards (items[i::5]) and merged.
    Qwen3-1.7B: L3-vote rp, L1 rp, L3-vote rp-noclip, penalty-decoded self-SFT (default-rollout runs exist).
  Evaluation (selection-free, no DEV sweep): TEST at the final checkpoint with default decoding and with repetition
  penalty 1.1 (`exp2 eval-final`); existing 1.7B runs L1, L3-vote noclip and hot-self τ1.5 also get the penalty
  evaluation. 4B training runs on RTX PRO 6000 (does not fit an A6000 next to vLLM); all evaluation on A6000.
  Primary outcomes (paired bootstrap, per seed), for each model:
    P1 beats the training-free baseline: rp L3-vote with penalty decoding - base + penalty 1.1, and rp L3-vote with
       default decoding - base + penalty 1.1 (loop-breaking internalised); on maj@12, pass@12, mean_acc.
    P2 Swahili kept: answer-language share of rp L3-vote >= that of base + penalty 1.1 (descriptive, with CI).
    P3 transfer: rp L3-vote - rp L1 (information effect when rollouts are coherent).
    P4 clip: rp L3-vote - rp L3-vote no clip.
  Predictions: P3 > 0 on Qwen3-4B (and smaller or none on 1.7B); P1 >= 0; P4 smaller than with default rollouts
  (with coherent rollouts the clip's entropy effect matters less). A method "works" only if P1 > 0 on mean_acc or
  maj@12 with the CI excluding zero in both seeds and P2 holds.
- 2026-11-14 ~15:10, implementation note (before any Qwen3-4B voted run started): Qwen3-4B has only a COPSD
  consensus (no GSM8K one), so its vote contexts are not size-equalised with GSM8K: 487 voted COPSD rows (vote share
  >= 5/8; the 1.7B voted runs had 442 after equalisation). Qwen3-1.7B rows unchanged (verified identical qids).
  4B consensus: maj@8 0.882 vs gold, coverage at 5/8 0.974, precision 0.893 (1.7B thinking-mode: 0.889 precision).
- 2026-11-14 ~20:30, self-SFT filter ablation (agreed with the user; written before any of these runs started).
  Known: every result above. COPSD with the voted reference does not transfer (P3 failed on both models, by ~18
  maj@12 points on 4B); penalty-decoded self-SFT is the only method that breaks loops and keeps Swahili, and with
  penalty decoding it ties base + penalty 1.1. Open question: is the English majority vote what makes self-SFT work,
  or would any finished Swahili answer do? That decides whether the result supports the proposal (the model's own
  English understanding teaches its Swahili) or is plain penalty-decoded self-training.
  Arms (selfsft.FILTERS): one shared sample pool per model (the existing penalty-1.1 generation, 8 samples per
  question over the vote-accepted COPSD questions); a sample is eligible if it finished, has an answer after
  </think> and is in Swahili. It is kept if its answer equals:
    vote      the English majority vote (the method; gold-free)
    swvote    the majority of the question's own eligible Swahili samples (>= 2 agreeing, strictly more than any
              other answer; gold-free, no English)
    nofilter  anything (no answer check)
    gold      the gold answer (oracle upper bound; analysis only, not a candidate method)
  at most 2 per question, then subsampled (seed 1234) to the smallest arm (gold on both models): 208 rows on 1.7B,
  305 on 4B. Run names selfsft-vote__copsd__exact__rp-<filter>__s{42,43}; training and evaluation exactly as the
  existing penalty-decoded self-SFT (30 SFT steps; TEST at step 30 with default decoding and with penalty 1.1).
  The vote arm is re-run at the equalised size (the existing runs used 228 / 308 rows).
  Known before training (descriptive, computed with gold on the kept training answers; not an outcome): share of
  kept answers that are correct: 1.7B vote 0.865, swvote 0.250, nofilter 0.178; 4B vote 0.902, swvote 0.443,
  nofilter 0.239. Kept sets cover different questions (1.7B 135 / 158 / 191 / 125; 4B 181 / 204 / 256 / 180).
  Primary outcomes (TEST, default decoding, mean_acc and maj@12, paired bootstrap per seed, each model):
    F1 English vote vs Swahili self-vote: vote - swvote      (does English understanding add over self-consistency?)
    F2 English vote vs no filter:         vote - nofilter    (does any answer filter matter?)
    F3 gap to the oracle:                 gold - vote        (how much a perfect verifier would add)
  Secondary: the same with penalty-1.1 decoding; truncation and Swahili share per arm.
  Predictions: F1 > 0 and F2 > 0 on mean_acc (the filter's precision gap is large); F3 small (|F3| < F2).
  The proposal's claim is supported if F1 > 0 with the CI excluding zero in both seeds on at least one model.
  If F1 ~ 0 and F2 ~ 0, the self-SFT gain is loop-breaking from penalty-decoded data, not English knowledge, and is
  reported as such. Next step (iteration / more questions to beat base + penalty) is decided after these results.
- 2026-11-14 ~22:30, large self-SFT filter ablation (agreed with the user: larger set, more steps, a bigger model;
  written before any of these runs or their data exist). Known: the small ablation above: F1, F2, F3 all ~0 on
  both models (gold-correct training answers = 18%-correct ones). Question: is answer correctness irrelevant only
  because 30 steps on 200-300 examples teach form (finish, no loops) but not reasoning?
  Data: gsm8k-mt3k = a 3000-question GSM8K *train* sample (data.gsm8k_items n 3000, seed 42; may overlap the
  500-question gsm8k-mt sample), NLLB-200 3.3B translation with the same QC; English thinking-mode consensus per
  model on the 3000 English questions (n 8, T 0.6, 16384 tokens, threshold 5/8, as before; sharded and merged).
  Sample pool per model: 8 Swahili samples per QC-passing question with repetition penalty 1.1 (COPSD training
  sampling, 2048 tokens), over *all* questions (the gold / no-filter / Swahili-vote arms are not restricted to
  questions the English vote accepted). Arms as before (vote / swvote / nofilter / gold, at most 2 per question,
  equalised to the smallest arm, seed 1234). Training: 100 SFT steps (batch 32; LR schedule over 100 steps as
  before), seeds 42/43. Models: Qwen3-1.7B (A6000) and Qwen3-8B (RTX PRO 6000; results/exp2_qwen3-8b; its base
  L0 / L0-reppen1.1 evaluations copied from results/exp2/L0-qwen3-8b[-reppen1.1]). Qwen3-4B is not repeated (cost).
  Run names selfsft-vote__gsm8k-mt3k__exact__rp-<filter>__s{42,43}. Evaluation: TEST at step 100, default decoding
  and penalty 1.1 (eval-final), as before.
  Primary outcomes, same as before: F1 vote - swvote, F2 vote - nofilter, F3 gold - vote (mean_acc and maj@12,
  default decoding, paired bootstrap per seed, each model); plus G = gold - nofilter (does correctness matter at all).
  Predictions: G > 0 at this scale on at least one model (if correctness ever matters, it should with ~10x the
  training examples); F1, F2 > 0 if G > 0. If G ~ 0 on both models, the conclusion is that penalty-decoded
  self-SFT teaches form, not reasoning, at the scale this project can afford, and the write-up says so.
- 2026-11-14 ~23:45, round 2 of self-SFT (user asked to keep experimenting overnight; written before any round-1
  result of the large ablation exists). Question: does on-policy iteration (sample -> filter -> train, again from
  the improved model) beat base + penalty 1.1 (P1, never met so far), and does the filter start to matter once the
  model is better (in round 2 the model's own Swahili answers are more often finished and correct)?
  Method: for each large round-1 run selfsft-vote__gsm8k-mt3k__exact__rp-<f>__sN, run ..rp-<f>-r2__sN: sample
  its step-100 checkpoint with exactly the round-1 pool's questions, prompts, sampling (penalty 1.1) and seeds;
  keep with the same filter f (at most 2 per question); subsample (seed 1234) to the round-1 equalised size of
  that model (fewer if fewer kept, reported); train the round-1 adapter for 100 more SFT steps with a fresh
  optimizer and LR schedule. Arms: Qwen3-1.7B all four filters; Qwen3-8B vote, nofilter, gold (cost); seeds 42/43.
  Evaluation as before (TEST, step 100 of round 2, default and penalty-1.1 decoding).
  Outcomes: R1 = round 2 - round 1 (same filter, seed); P1 = round 2 - base + penalty 1.1 (default decoding and
  penalty decoding); F1/F2/F3/G as before within round 2. Predictions: R1 > 0 for vote and gold, ~0 for nofilter
  (iterating on unchecked answers should not keep improving reasoning); P1 still <= 0 with default decoding.
- 2026-11-15 ~11:30, round 3 of self-SFT on Qwen3-8B (user asked to test how far the English-vote advantage keeps
  growing; written after the round-2 results, before any round-3 run or pool exists). Known: Qwen3-8B round 2
  (commit 02015b0) gave the first non-null filter effect: F2 vote - nofilter +2.3 / +2.5 and G gold - nofilter
  +3.0 / +2.5 mean_acc (CIs exclude 0, both seeds), vote ~ gold; R1 +2.6..+4.3 for vote / gold, +1.1..+2.5 for
  nofilter; round-2 vote still -1.6 / -1.8 vs base + penalty 1.1 (n.s.). Qwen3-1.7B round 2: filter null.
  Method: identical to round 2, one round on: ..rp-<f>-r3__sN samples ..rp-<f>-r2__sN step 100 with the round-1
  pool's questions, prompts, sampling (penalty 1.1) and seeds; same filter, at most 2 per question; subsample (seed
  1234) to the round-1 equalised size (3166; fewer if fewer kept, reported); train the round-2 adapter 100 more SFT
  steps, fresh optimizer and LR schedule. Arms: vote, nofilter, gold; seeds 42/43; Qwen3-8B only (1.7B was null).
  Evaluation as before. Outcomes: F2(r3) = vote - nofilter and its change from round 2 (F2(r3) - F2(r2), descriptive,
  per seed); G(r3); F3(r3) = gold - vote; R2 = round 3 - round 2 per filter; P1(r3) = vote round 3 - base +
  penalty 1.1. Predictions: F2(r3) >= F2(r2) (the advantage keeps growing: nofilter compounds its own errors);
  F3(r3) ~ 0; R2 > 0 for vote, smaller than R1 (diminishing returns); P1(r3) for vote reaches ~0 or above.
  "The advantage keeps growing" is supported if F2(r3) > F2(r2) in both seeds with F2(r3) CIs excluding 0.
  If F2(r3) <= F2(r2), the effect is a one-step separation that saturates, and the write-up says so.
