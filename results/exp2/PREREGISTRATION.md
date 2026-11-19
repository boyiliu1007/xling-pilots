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
  (commit 53c6b4d) gave the first non-null filter effect: F2 vote - nofilter +2.3 / +2.5 and G gold - nofilter
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
- 2026-11-15 ~15:45, post hoc check of the Qwen3-8B round-2 filter effect (after seeing round 2 and the round-3
  sampling pool, before any round-3 TEST result). Prompted by language drift: the whole-output English share rises
  from ~10% (round 1) to 16-22% (round 2), mostly English reasoning with a Swahili final answer, and usable
  Swahili samples in the next pool fall 15361 -> 13627 -> 8354. Recomputed per item: share of the 12 samples that are
  correct AND in Swahili (answer language; and whole-output language). Result (s42 / s43, paired bootstrap):
  F2 vote - nofilter +2.3 / +2.5 (any language) -> +1.4 / +1.5 (answer in Swahili, CIs include 0) -> +1.4 / +0.3
  (whole output in Swahili); G +3.0 / +2.5 -> +1.8 / +0.9 -> +1.8 / +0.2; F3 ~0 throughout. R1 (round 2 - round 1)
  for vote +3.5 / +2.6 -> +2.4 / +2.3 -> -0.3 / -2.0. Reading: about 40% of the round-2 vote advantage, and all of
  the round-2 iteration gain in fully-Swahili outputs, comes from more English in the output. Round 3 is reported on
  both metrics (any language, and correct with the whole output in Swahili).
- 2026-11-15 ~17:00, RL with verifiable rewards (user: "find something that works when the model is very bad at
  the language; gold as privileged information is fine"; start on Qwen3-1.7B, move to 4B if 1.7B is too small).
  Written before any run of this kind exists. Known: everything above, including the language drift of iterated
  self-SFT and the Swahili-only re-analysis of the 8B round-2 effect (no significant filter effect remains).
  Model choice (decided now by a rule computed on existing pools, before training): reward signal on the 3000
  GSM8K-MT questions, 8 penalty-1.1 samples per question: Qwen3-1.7B 5.3% of samples correct and Swahili, only 17%
  of questions with mixed outcomes (0 < correct < 8); Qwen3-8B 30.6% / 60%. 1.7B is too sparse for group-relative
  RL, so the experiment runs on Qwen3-4B (between the two; base TEST mean_acc 0.144).
  Method (votedistill/exp2/grpo.py): on-policy GRPO, one optimizer step per batch (ratio 1, no clipping, no KL),
  advantage = reward - group mean (no std division), loss normalised by a constant (rollouts x 2048; Dr. GRPO).
  8 questions x 8 rollouts per step, 200 steps (1600 questions, one pass), student Swahili prompt (COPSD style,
  thinking), T 1.0, top-p 1.0, no repetition penalty, 2048 tokens; LoRA r 64 / alpha 128 as before, lr 1e-5
  constant, AdamW, grad clip 1.0; checkpoints every 25 steps. Run names grpo__gsm8k-mt3k__exact__<reward>__s{42,43}.
  Rewards: gold (correct answer, any language; ceiling), goldsw (correct AND whole output Swahili by
  copsd.evaluate.language_of; the main arm), and later, once Qwen3-4B's English consensus on gsm8k-mt3k exists,
  votesw (answer = own English majority vote AND whole output Swahili; gold-free) and goldswv (goldsw + 0.5 x votesw).
  Evaluation (selection-free): TEST at step 200, default decoding and penalty 1.1 (eval-final). Both metrics are
  primary from now on: mean_acc (any language) and Swahili accuracy = share of the 12 samples that are correct AND
  whose whole output is Swahili; maj@12 and pass@12 secondary.
  Primary contrasts (paired bootstrap, per seed): goldsw - base + penalty 1.1 (Swahili accuracy; does anything
  beat the decoding baseline on Swahili?), gold - base + penalty 1.1 (mean_acc), goldsw - gold (Swahili accuracy
  and mean_acc: cost/benefit of the language reward). Predictions: gold raises mean_acc above base + penalty but
  drifts toward English (Cross-lingual Collapse, arXiv 2506.05850); goldsw keeps Swahili >= 0.9 and beats base +
  penalty on Swahili accuracy; goldsw - gold < 0 on mean_acc by a few points. "Something works" = goldsw beats base
  + penalty 1.1 on Swahili accuracy with the CI excluding 0 in both seeds.

## 2026-11-15 ~18:45 — GRPO amendment: higher learning-rate arm (decided before any GRPO evaluation)

Observed (training log only, `goldsw` s42 at lr 1e-5, steps 1–26): batch reward flat at about 0.17
(0.180 / 0.172 / 0.167 over steps 1–10 / 11–20 / 21–26), whole-output Swahili share 0.95 → 0.90, about 3 of 8
questions per step carrying signal. With Adam the step size is set by the learning rate, not by the loss
normaliser, and 1e-5 is at the low end for LoRA RL (LoRA's best learning rate is typically about 10× full fine-tuning's).

Added arm: `grpo__gsm8k-mt3k__exact__goldsw-lr5e-5__s42`, identical except learning rate 5e-5 (variant suffix
`-lr<value>`; plain names keep 1e-5). Rule fixed now: if the lr-1e-5 `goldsw` run's mean reward over steps 41–50
is not above its steps 1–10 mean (0.180), the lr-1e-5 run is stopped and the remaining arms (gold, s43, votesw,
goldswv) move to the learning rate that learns; the lr-1e-5 checkpoints that exist are still evaluated and reported.
Both learning rates are reported; the success criterion is unchanged and is judged on the arm(s) that complete.

### ~19:05 — step-50 rule applied
`goldsw` lr 1e-5 s42: steps 41–50 mean reward 0.141 vs steps 1–10 0.180 (0.172 / 0.173 / 0.144 in between) →
stopped at step 51. Its step-50 checkpoint is evaluated (TEST, default and penalty-1.1 decoding) as
`EXP2_FINAL_STEP=50`. The other lr-1e-5 arms (gold s42, gold s43, goldsw s43) are held, not yet moved: they move to
lr 5e-5 only if the lr-5e-5 run learns by the same rule (steps 41–50 mean reward above its steps 1–10 mean).

### ~19:50 — lr 5e-5 passes the step-50 rule; remaining arms moved to lr 5e-5
`goldsw` lr 5e-5 s42: steps 41–50 mean reward 0.208 vs steps 1–10 0.167 → passes. (Both learning rates use the same
seed-42 question schedule, so steps 41–50 are the same questions: 0.208 at 5e-5 vs 0.141 at 1e-5.) The held lr-1e-5
arms are cancelled and resubmitted as `gold-lr5e-5` s42, `gold-lr5e-5` s43 and `goldsw-lr5e-5` s43. Success criterion
unchanged, judged on the lr-5e-5 arms.

## 2026-11-16 ~00:00 — GRPO amendment: larger batch (decided before any GRPO test evaluation)

Observed (training log only, `goldsw` lr 5e-5 s42, 200 steps): batch reward about 0.19 over steps 1–75, then
0.11–0.16; Swahili share 0.95 → 0.99; steps 151–175 unstable (finished 0.67, length 1370). Only about 2.5–3 of the
8 questions per step had mixed rewards, so each update came from about 20 informative samples.

Added arm: `grpo__gsm8k-mt3k__exact__goldsw-lr5e-5-q32-steps150__s42` (and s43): 32 questions per step (about 10
informative per update), 150 steps (4800 question visits, about 1.6 passes over the 3000 questions), lr 5e-5;
everything else unchanged. Variant suffixes `-q<n>` and `-steps<n>`; plain names keep their earlier settings.
Evaluated like the other arms (TEST of step 150, default and penalty-1.1 decoding). Success criterion unchanged.

## 2026-11-16 ~00:30 — GRPO amendment: the `goldsw` reward was hacked; hack-resistant reward `goldswr`

Result (TEST, `goldsw` lr 5e-5 s42, step 200; paired bootstrap over items, 95% CI):
- default decoding: mean_acc 12.5 vs base 14.4 (−1.9 [−5.0, +1.2]); majority 14.5 vs 26.5.
- penalty 1.1: mean_acc 13.6 vs base+penalty 23.2 (−9.6 [−13.0, −6.5]); Swahili accuracy 12.8 vs 18.0 (−5.1 [−8.2, −2.2]).
- Reward hacking: the policy passes `language_of` (Swahili vs English function-word COUNTS) by repeating one Swahili
  function word: "jibu" is 51% of all words in its test outputs, surrounded by pseudo-words ("toto", "totali",
  "soali"). Median distinct Swahili function words per correct output: 3 (base 12, base+penalty 20).

Change: reward `goldswr` = correct AND `robust_sw` (at least 8 DISTINCT Swahili function words, more distinct Swahili
than English ones, and no word above 35% of all words). On existing TEST outputs it keeps 92% / 91% of base /
base+penalty `goldsw` successes and 0% of the hacked run's. Runs: `goldswr-lr5e-5-q32-steps150` s42 and s43.
The q32 `goldsw` runs (0 steps logged) and the queued `goldsw-lr5e-5` s43 are cancelled. `gold-lr5e-5` s42/s43
(language-free reward, cannot hack the check) stay as the control.
Evaluation: Swahili accuracy is reported with BOTH `language_of` and `robust_sw`; for GRPO arms the `robust_sw`
version is primary, since `language_of` is the quantity that was gamed.

### 2026-11-16 ~08:50 — goldswr s42 training finished; second hack suspected; exploratory step-50 evaluation
Training log, 25-step blocks: correct 0.197 / 0.242 / 0.216 / 0.167 / 0.178 / 0.207; language_of Swahili share
0.92 / 0.99 / 0.99 / 0.87 / 0.95 / 0.57. In the last block the reward (correct AND robust_sw, 0.165) exceeds correct AND
language_of-Swahili (0.145), so robust_sw accepts outputs language_of calls English: suspected code-switching (English
text with at least 8 distinct Swahili function words). The preregistered evaluation stays TEST of step 150. Added,
EXPLORATORY and chosen after seeing the training log: TEST of step 50 (training-log peak), job 11173016.

## 2026-11-16 ~10:30 — NEW EXPERIMENT (replaces the GRPO arms above): self-play transfer with a privileged English reference

The earlier GRPO arms are cancelled (user decision). Reason: two reward hacks of word-list language checks, and
no accuracy gain. The design follows SP3F (Sutawika, Swamy, Wu, Neubig, ACL 2026, "Gained in Translation"): RL with
gold-answer + language-fidelity rewards plus a judge that sees an English reference as privileged information. The
change that makes it test THIS project's claim: in the gold-free arms the privileged reference is the model's OWN
English solution (majority vote of 12 English thinking-mode samples), not a gold one.

Arms (Qwen3-4B, LoRA r64, thinking mode, gsm8k-mt3k Swahili questions, 8 rollouts per question, 32 questions per
step, 300 steps, constant lr 5e-6, temperature 1.0, Dr. GRPO advantages, one update per batch, no KL):
- A `goldlid`:     gold-correct + lid_ok                                       (gold ceiling, answer only)
- B `goldlidref`:  gold-correct + lid_ok + ref_f1 vs the gold English GSM8K solution   (SP3F-like, gold privilege)
- C `votelid`:     equals own English vote + lid_ok      (vote-accepted questions only; gold-free)
- D `votelidref`:  equals own English vote + lid_ok + ref_f1 vs own English voted answer text  (THE PROPOSAL)
lid_ok = at least 70% of the prose is Swahili by lingua (all languages; calibrated: 88% of base TEST outputs pass, 9%
of the "jibu"-hacked outputs). ref_f1 = F1 of the number sets in the rollout and the reference (judge-free stand-in
for SP3F's pairwise judge; an LLM judge is a later step only if D or B shows something).
Deviations from SP3F: LoRA instead of full fine-tuning (lr 5e-6 ≈ 10× their 5e-7), no SFT stage (the
translated-reasoning SFT is outside supervision; the gold-free arms must not use it), 300 steps instead of 500, one
update per batch (so no PPO clipping), number overlap instead of an LLM judge. One seed (42) first; seed 43 after.

Primary metric: TEST (AfriMGSM swa) accuracy of outputs that pass lid_ok ("Swahili accuracy, LID"), step 300, default
decoding; also any-language mean_acc and the penalty-1.1 decoding, vs base and base+penalty, paired bootstrap.
Contrasts: D − C (does own English reasoning help beyond its answer), D − B and C − A (cost of no gold),
B − A (does the reference help with gold), each arm − base. Success for the proposal: D > base on Swahili accuracy
(LID) with the CI excluding 0. Rollout texts are saved every 10 steps and read for hacking at step 50.

### 2026-11-16 ~20:35 — amendment: 16 questions per step (before any result)
Arm D at 32 questions/step took about 6.5 min/step (3 steps run, then aborted and deleted): with the continuous
ref_f1 term every group carries signal, so all 256 rollouts are back-propagated, and 300 steps would take ~33 h.
All arms now use 16 questions per step (128 rollouts), 300 steps, lr 5e-6 (same number and size of updates; 16
informative groups per update, more than any earlier run): `<reward>-lr5e-6-q16-steps300`. Compute: D and C on the
interactive node's A6000 (one after the other), A and B on the shared preempt queue (they start whenever it allows;
moved to the interactive node if D finishes first). Everything else as in the ~10:30 entry.

### 2026-11-17 ~00:20 — exploratory step-150 DEV check of arm D (before any evaluation of it)
Step-50 training logs: language pass and finish rates rise, accuracy flat (0.13-0.20 among finished outputs); ~2/3 of
groups have all 8 rollouts wrong, so most of the signal is form (lid_ok, ref_f1). To decide early whether the method
learns reasoning, arm D's step-150 checkpoint is evaluated on DEV (50 items x 12 samples), default decoding and
penalty 1.1, into results/exp2_qwen3-4b/devcheck/ (training paused ~1 h, then resumed from step 150). Compared with
base and base+penalty on DEV (L0 / L0-reppen1.1 base_dev.jsonl). Reading, fixed now: "promising" = Swahili accuracy
(LID) under default decoding above base+penalty's, and above base with the paired CI excluding 0. DEV is small, so
this only steers effort (whether C, A, B and seed 43 are worth the GPU time); it changes nothing in the primary
analysis, and D still trains to step 300 regardless.

### 2026-11-17 ~03:40 — result of the step-150 DEV check (arm D)
DEV, 50 items x 12, Swahili accuracy (LID) / any-language mean_acc / LID pass / truncated:
base 12.5 / 14.0 / 89.2 / 26.5; base+penalty 13.5 / 23.0 / 66.0 / 5.2; D step 150 22.8 / 23.3 / 96.8 / 1.2;
D step 150 + penalty 23.2 / 24.2 / 92.2 / 0.8. D − base, Swahili accuracy +10.3 [+5.8, +15.2]; D − base+penalty
+9.3 [+4.7, +14.3] -> "promising" by the rule above. But any-language mean_acc only ties base+penalty (+0.3
[−4.0, +5.2]): so far D reaches the accuracy that penalty decoding reaches, while staying in Swahili, rather
than reasoning better. C, A and B proceed as planned.

### 2026-11-17 ~09:30 — TEST results, arms A and B (s42, step 300); compute note for C
TEST, 200 items x 12; Swahili accuracy (LID) / any-language mean_acc / LID pass / truncated:
base 11.9 / 14.4 / 85.8 / 31.0; base+penalty 14.8 / 23.2 / 63.2 / 6.6;
A goldlid 23.5 / 24.5 / 96.9 / 6.0; B goldlidref 26.1 / 27.6 / 96.0 / 1.7.
B − base+penalty: Swahili accuracy +11.2 [+8.3, +14.3], mean_acc +4.4 [+1.7, +7.1]. A − base+penalty: +8.7 [+6.3,
+11.2], mean_acc +1.3 [−1.1, +3.8]. B − A: +2.6 [+0.8, +4.5], mean_acc +3.1 [+1.3, +4.9]. Penalty decoding:
A+pen 24.9 / 25.9, B+pen 25.9 / 27.9.
Compute: preempt RTX PRO 6000 ran 300 steps in ~4 h (local A6000 ~13 h), so arm C moved from the interactive node to
preempt (train 11186698, eval 11186699); D finishes and is evaluated locally.

### 2026-11-17 ~12:45 — TEST result, arm D (votelidref, s42, step 300, default decoding)
Session loss: interactive job 11180064 ended 09:51 at step 288; D resumed from step-275 on babel-u9-32.
D: Swahili accuracy (LID) 25.2 / mean_acc 26.1 / LID pass 98.0 / truncated 2.1.
D − base: +13.4 [+10.4, +16.5] (preregistered success criterion met), mean_acc +11.7 [+8.8, +14.6].
D − base+penalty: +10.4 [+7.9, +13.0], mean_acc +2.8 [+0.4, +5.3]. D − B (gold): −0.8 [−2.7, +1.0], mean_acc −1.5
[−3.3, +0.2]. D − A: +1.8 [−0.1, +3.5]. Penalty decoding and arm C pending.
D, penalty 1.1 decoding: 25.7 / 27.0 / 96.3 / 0.7. D+pen − base+pen: +10.8 [+8.3, +13.5], mean_acc +3.8 [+1.4, +6.3];
D+pen − B+pen: −0.2 [−2.2, +1.7], mean_acc −0.9 [−2.9, +1.0].

### 2026-11-17 ~15:00 — TEST result, arm C (votelid, s42, step 300, default decoding)
C: Swahili accuracy (LID) 25.0 / mean_acc 25.6 / LID pass 98.1 / truncated 7.0.
C − base +13.1 [+10.2, +16.1]; C − base+penalty +10.1 [+7.7, +12.7], mean_acc +2.3 [+0.0, +4.7]; C − A +1.5
[−0.2, +3.1]. D − C: +0.3 [−1.3, +1.9], mean_acc +0.5 [−1.1, +2.1] — the own-English-solution reference adds
nothing measurable on top of the own-English-vote answer reward (unlike the gold reference: B − A +3.1 [+1.3, +4.9]),
though it lowers truncation (2.1 vs 7.0). C penalty decoding pending.

### 2026-11-17 ~15:30 — follow-up: two controls for "does the model's own ENGLISH ability teach it?" (before any result)
C and D gain ~+13 Swahili accuracy over base, but C's reward has two parts (agree with own English vote, lid_ok). Two
controls, same settings as C (vote-accepted questions, lr 5e-6, q16, 300 steps, s42, preempt RTX PRO 6000):
- E `lidonly`: reward = lid_ok only. Tests whether the gain is just staying in Swahili / finishing.
- F `selfvotelid`: reward = agrees with the group's own Swahili majority answer (TTRL-style, most common answer among
  the 8 finished samples, needs >= 2 votes) + lid_ok. Tests whether English is needed or any self-consistency works.
Primary: TEST Swahili accuracy (LID), default decoding, step 300; also mean_acc. Contrasts C − E and C − F, paired
bootstrap. Reading: English ability teaches = C > E and C > F, CIs excluding 0. If C ≈ F, self-consistency suffices;
if C ≈ E, the gain is form (language + finishing). Rollout texts read at step 50 for hacking (E may collapse to
short Swahili with no math).

### 2026-11-17 ~22:00 — follow-up batch (before any result): prior-work reward, offline vs online, transfer
Related work found 2026-11-17 (PASMR, arXiv 2601.17671; cross-lingual self-consistency, arXiv 2606.01464) already
rewards target-language answers by agreement with the model's own English answer, without a language reward.
Three additions, all s42, Qwen3-4B, gsm8k-mt3k:
- G `grpo__..__voteonly-lr5e-6-q16-steps300`: reward = agrees with own English vote only (no lid_ok), C's
  questions and settings. Prediction: any-language mean_acc up, Swahili accuracy (LID) below C. Contrast C − G.
- RFT `selfsft-vote__gsm8k-mt3k__exact__lidvote__s42`: C's reward applied OFFLINE. Base model samples 8 Swahili
  answers per question once (self-SFT generation: T 1.1, top-p 0.95, top-k 20, 2048 tokens); keep those agreeing
  with the English vote AND lid_ok (<= 2 per question); SFT 100 steps (COPSD optimizer, LoRA r64, lr 5e-6, 32/step).
  Contrast C − RFT: same signal, online vs offline.
- Transfer and forgetting: base, C and D (step 300) on AfriMGSM TEST in English (eng) and Hausa (hau), default
  decoding, any-language mean_acc, paired bootstrap vs base. Exploratory.
Evaluation as before (TEST, final checkpoint, default + penalty 1.1 decoding for G and RFT).

### 2026-11-17 ~22:40 — two new arms built on D (before any result)
Diagnosis of D (its rollouts): ref_f1 is mostly copying and correctness — the question's own numbers alone score
0.44 against the own English solution (30% of the reference's numbers are in the question), and rollouts with the
vote answer score 0.89 vs 0.63 without. About 2/3 of groups have no rollout matching the vote, so they carry no
answer signal. Two arms, settings as C/D (vote-accepted questions, lr 5e-6, q16, 300 steps, s42, preempt):
- H `votelidstep`: C + step_f1 = F1 between the numbers the rollout introduces (not in the Swahili question) and
  the own English solution's intermediate results (its numbers not in the question). Process credit that copying
  the question cannot earn and number spam lowers. Contrast H − C (does own English reasoning help when the credit
  targets intermediate steps?) and H − D.
- I `votelidrefanc`: D + self-translated anchors. For each group where no rollout matches the English vote, the
  current policy translates its own English solution into Swahili (non-thinking, greedy, <= 1536 tokens); the
  anchor = that translation as the thinking + "**Jibu:** \boxed{vote}". It joins the group as a 9th sample only if
  it keeps the vote answer, passes lid_ok and fits 2048 tokens; advantage over all 9; no importance weighting
  (one update per batch). Anchors tried/accepted are logged per step; texts of rejected anchors are saved.
  Contrast I − D (do self-translated demonstrations rescue all-wrong groups?).
Primary metric and evaluation as before. Rollout texts read at step 50 for hacking (I: is the policy learning to
copy translated solutions verbatim? H: number padding?).

### 2026-11-17 ~22:10 — TEST result, control E (lidonly, s42, step 300)
E: Swahili accuracy (LID) 9.4 / mean_acc 9.4 / LID pass 99.9 / truncated 11.8; penalty decoding 8.8 / 9.0.
C − E: +15.6 [+12.2, +19.2], mean_acc +16.2 [+12.7, +19.8]. E − base: −2.5 [−4.3, −0.7], mean_acc −5.0 [−6.9, −3.2].
Reading (as preregistered): the language reward alone does not produce C's gain; it lowers accuracy below the
untrained model. F (Swahili self-vote) pending.

### 2026-11-18 ~00:50 — TEST results: control F (selfvotelid) and English / Hausa transfer
F (Swahili self-vote + lid_ok): Swahili accuracy (LID) 17.8 / mean_acc 19.0 / truncated 0.7; penalty 18.2 / 19.5.
C − F: +7.2 [+4.6, +9.9], mean_acc +6.6 [+4.0, +9.3]. With C − E (+15.6) both preregistered conditions for "the
model's own English ability teaches" hold (C > E and C > F, CIs excluding 0).
AfriMGSM TEST mean_acc, default decoding, vs base (paired bootstrap): English base 94.9, D 95.5 (+0.6 [−0.1, +1.4]),
C 95.5 (+0.5 [−0.1, +1.3]) — no forgetting. Hausa (never trained) base 3.6, D 5.3 (+1.8 [+0.6, +3.0]), C 5.7
(+2.1 [+1.0, +3.4]) — small transfer.

### 2026-11-18 ~01:20 — follow-ups (before any result): longer training, and Swahili + Hausa
1. Longer training of the method (D). Training accuracy was still rising at step 300 (blocks of 50: 0.15, 0.16,
   0.20, 0.21, 0.22, 0.24). `grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps600__s42` continues D exactly:
   step-300 checkpoint (adapter, optimizer state) copied; constant LR and per-step sampling seeds, and the 600-step
   question order starts with the 300-step order (verified). TEST at steps 450 and 600 (default + penalty 1.1).
   Reading: D600 − D300 and D450 − D300 on Swahili accuracy (LID); descriptive (more steps chosen post hoc).
2. One English teacher for two languages (design entry to follow once the Hausa data and language check exist).

### 2026-11-18 ~11:40 — TEST results: G, offline RFT, H, and D at 450 / 600 steps (s42)
Swahili accuracy (LID) / mean_acc / LID pass, default decoding (paired bootstrap, 200 items):
- G `voteonly` (PASMR-style, no language term): 10.0 / 32.2 / 28.0. C − G: Swahili accuracy +15.0 [+11.5, +18.5],
  mean_acc −6.6 [−9.9, −3.4]. Without a language term the policy answers in English (72% of outputs): any-language
  accuracy rises, Swahili accuracy falls below base+penalty. The language check is what keeps the gain in Swahili.
- Offline RFT `selfsft-vote__gsm8k-mt3k__exact__lidvote` (C's reward as a filter, 100 SFT steps): 13.0 / 15.3 / 87.6
  (truncated 26.0). RFT − base +1.2 [−0.1, +2.5]; C − RFT +11.9 [+9.3, +14.7]. Same signal offline ≈ nothing: the
  on-policy delivery is what works.
- H `votelidstep`: 27.7 / 28.3 / 98.2. H − C +2.8 [+0.8, +4.8]; H − D +2.5 [+0.7, +4.3]. Credit on the own English
  solution's intermediate numbers helps where D's whole-reference F1 did not (D − C +0.3 n.s.).
- D continued: step 450 29.4 / 30.0; step 600 32.5 / 33.0 / 98.8. D600 − D300 +7.2 [+4.5, +10.0]; D600 − D450
  +3.1 [+1.1, +5.1]; D600 − base+pen +17.7 [+14.1, +21.2], mean_acc +9.7 [+6.7, +12.8]; D600 − B (gold, 300
  steps, unequal compute) +6.4 [+3.9, +9.0]. Still rising at 600.
- I `votelidrefanc`: the eval job was cancelled (its dependency was the crashed first attempt); resubmitted 11200188.

### 2026-11-18 ~11:40 — design (before training): Swahili + Zulu, one English teacher
Second language Zulu, not Hausa (lingua has no Hausa) and not Yoruba (NLLB writes numbers as words; only 155/3000
Yoruba translations pass the number QC vs 1131 Zulu). Language check calibration: lingua labels 16% of QC-passed
Zulu questions <70% Zulu, mostly as Xhosa; for Zulu, chunks labelled ZULU or XHOSA count (questions then pass 98.5%;
Swahili check unchanged). Base Qwen3-4B on AfriMGSM Zulu TEST: mean_acc 2.5, Zulu accuracy (LID) 2.2, LID pass 91.
Run `grpo__gsm8k-mt3k-zul__exact__votelidref-lr5e-6-q32-steps300__s42`: D's reward (vote_ok + lid_ok in the
question's language + ref_f1 vs the own English vote solution), vote-accepted questions of both languages (2752
Swahili + 1109 Zulu, same English vote per qid), uniform sampling, q32, 300 steps (≈6900 Swahili question draws ≈
D at ~430 steps). Evaluation: Swahili TEST (compare to D450 as the closest compute match, and to D300), Zulu TEST
(vs base, Zulu accuracy by LID), Hausa (unseen) and English (forgetting). Reading: Zulu − base > 0 with CI excluding
0 = the recipe works on a second, much weaker language; Swahili ≥ D450 − 2 = no interference.

### 2026-11-18 ~12:00 — main-conference plan (before any of these results)
Goal: turn the single-seed Qwen3-4B Swahili result into a paper. All settings as D (lr 5e-6, q16, 300 steps, LoRA
r64, Dr. GRPO) unless stated; primary metric Swahili accuracy (LID) on AfriMGSM TEST; paired bootstrap over items,
seeds averaged per item where a run has several seeds; Holm correction over the paper's main contrasts (listed below).
Wave 1 (existing code, launched now):
1. Seeds 43 and 44 for A goldlid, B goldlidref, C votelid, D votelidref, E lidonly, F selfvotelid, G voteonly,
   H votelidstep, and offline RFT (selfsft lidvote). Main claims are re-tested on 3 seeds.
2. Equal compute: B, G and H continued to 600 steps from their step-300 checkpoints (as D600). The claim "gold-free
   ≥ gold" is judged at equal steps (D600 vs B600); "language check needed" at G600 vs D600/C.
3. Model scale: Qwen3-1.7B and Qwen3-8B, s42, runs B, C, D, F, G (own English vote from each model's own consensus
   file). Prediction: D − base > 0 and C − G > 0 on Swahili accuracy for both sizes.
Wave 2 (code to be written, design entries before launch): more training languages (single-language runs, lingua-
supported: Zulu, Yoruba with number words allowed, Shona, Sotho); baselines (translate-test with NLLB, SFT on
NLLB-translated gold solutions, prompting "think in English, answer in Swahili", PASMR-style SFT warm-up + vote
reward); second test sets (MGSM Swahili, MSVAMP Swahili); language-check ablation (Swahili-vs-English detector,
word-list check, thresholds 0.5/0.9) with hacking rates; vote-corruption (10/20/40% of vote answers replaced by a
wrong answer); a non-Qwen model; a blinded native-speaker rating sheet of 100 outputs; full TEST curve of D600.
Main contrasts (Holm): D − base, D − base+pen (any-language), D600 − B600, C − G, C − E, C − F, C − RFT, H − D.

### 2026-11-18 ~12:30 — wave 2a design (before any result): language-check ablation, vote noise, more languages
All on Qwen3-4B, D's reward and settings (votelidref, lr 5e-6, q16, 300 steps), s42; evaluation metric unchanged
(Swahili accuracy by lingua over all languages ≥ 0.7), so a check the policy games shows up as low LID pass / low
Swahili accuracy. Reported per run: Swahili accuracy, any-language accuracy, LID pass, the share of training rollouts
that pass the training check but fail the evaluation check ("hack rate").
- Language check: `-lid0.5`, `-lid0.9` (threshold), `-detsw2` (lingua limited to Swahili vs English), `-detwl` (the
  word-list check robust_sw used by goldswr). Prediction: detsw2 and detwl have higher hack rates and lower Swahili
  accuracy than D; thresholds change little.
- Vote noise: `-cor0.1`, `-cor0.2`, `-cor0.4` — that fraction of vote-accepted questions (seed 1234) get the vote
  answer and own solution of another question with a different answer (vote accuracy 94.9% → 85.5/76.0/57.0% approx).
  Reading: Swahili accuracy as a function of vote accuracy; where it falls to C − (C − E)/2 is the usable limit.
- Single-language runs (no Swahili): `gsm8k-mt3k-<lang>only` for Zulu (QC'd 1109 vote-accepted), Yoruba, Shona, Sotho,
  each graded by the same English votes; lingua check in that language (Zulu also accepts Xhosa). Yoruba QC relaxed:
  NLLB writes numbers as words, so only digit-numbers absent from the English question fail (2710/3000 pass; risk: a
  mistranslated number makes the vote wrong for that question — noted, not filtered). Shona and Sotho translated now,
  LID calibrated on base outputs before training. Per language: D and G (no language check), s42; evaluated on that
  language's AfriMGSM TEST (Swahili-style LID accuracy), Swahili and English.

### 2026-11-18 ~14:10 — Shona and Sotho calibration (before training)
NLLB QC: Shona 1343/3000, Sotho 1313/3000. Base Qwen3-4B AfriMGSM TEST: Shona mean_acc 2.4 (LID pass 62.4), Sotho 2.3
(59.7). lingua passes 98.3% of QC'd Shona questions; Sotho only 85% (misses labelled Tswana), so for Sotho chunks
labelled SOTHO or TSWANA count (as Zulu/Xhosa). Runs `gsm8k-mt3k-{sna,sot}only` × {votelidref, voteonly}, s42, as
wave 2a. Also logged (descriptive): D TEST curve every 50 steps is monotone within noise (300: 25.2, 450: 29.4,
600: 32.5); MSVAMP Swahili (1000 items, 4 samples): D 62.3 vs base+pen 41.3 Swahili accuracy (+20.9 [+19.1, +22.8]),
any-language +6.7 [+5.0, +8.5]; B (gold) 59.6.

## 2026-11-18 15:30 — MSVAMP Swahili results (1000 items, maj-free mean of 4, s42)

| run | Swahili acc | any-language acc | LID pass |
|---|---|---|---|
| base + rep. penalty 1.1 | 41.3 | 55.8 | 74.9 |
| gold + gold solution (B), 300 | 59.6 | 60.7 | 98.5 |
| vote + Swahili check (C), 300 | 55.9 | 56.3 | 99.5 |
| vote, no Swahili check (G), 300 | 24.0 | 66.7 | 36.3 |
| our method (D), 300 | 62.3 | 62.5 | 99.5 |
| D + step credit (H), 300 | 59.6 | 59.7 | 99.6 |
| our method (D), 600 | 65.1 | 65.3 | 99.7 |

Paired bootstrap (items), Swahili acc: D − base+pen +20.9 [19.1, 22.8]; D600 − base+pen +23.8 [21.7, 26.1] (any-language +9.5 [7.4, 11.6]); C − G +31.9 [29.8, 33.9]; D − B +2.7 [1.0, 4.3]; D600 − D300 +2.9 [1.2, 4.5]; **H − D −2.7 [−4.2, −1.1]** — the step-credit gain on AfriMGSM (+2.5) does NOT replicate on MSVAMP; claim 3 (step credit) is downgraded to "mixed" pending seeds. G has the highest any-language accuracy (66.7) but answers mostly in English, as on AfriMGSM.

## 2026-11-18 ~23:30 — EXPLORATORY: why does the own-English-vote reward match gold? (logs/why_vote.py, CPU only)

- The labels are almost gold. English thinking-mode vote (Qwen3-4B, n=8): accepts 2783/2805 GSM8K-mt3k questions
  (99.2%) and equals gold on 96.6% of them; on the COPSD olympiad set it accepts 487/500, 89.9% equal gold. Selection
  is not the mechanism (B also trains 99.3% on vote-accepted questions).
- Same answer signal: share of groups where the answer reward varies is 32.7/32.5/32.0% (B) vs 32.8/31.9/32.0% (D)
  for steps 1–100/101–300/301–600; Swahili gold accuracy of rollouts is 16.8→28.7 (B) vs 17.4→28.4 (D).
- Wrong votes are resisted: on the 3.4% vote-wrong questions D rewards only 7–12% of rollouts (vs 18–29% on
  vote-right ones); gold accuracy there is 4–7%, i.e. these are hard questions.
- The own-solution reference is NOT a better shaping signal than the gold rationale: in mixed groups, the
  highest-ref_f1 rollout is gold-correct 78.7/82.6/85.4% (D) vs 80.1/86.1/88.0% (B).
- TEST step 300 (s42): A 23.5, B 26.1, C 25.0, D 25.2 Swahili acc — D ≈ B at 300; D > B appears at 600 (32.5 vs
  30.1) and on MSVAMP (+2.7 [1.0, 4.3]).
Reading: "self-vote beats gold" is the wrong claim. The defensible claim is that the model's English competence
(96.6% vote accuracy vs ~17% Swahili accuracy at step 1) is a near-gold label source, so gold answers, gold
rationales and translated SFT (SP3F's supervision) are replaceable; the Swahili check is what makes it work.
Implication for a round-2 vote: on GSM8K at most 3.4% of labels can change, so round 2 vs D600 is expected null.

## 2026-11-18 ~23:50 — PREREGISTERED: self-translated training questions (no external MT system)

User decisions tonight: (1) cancel ~39 non-essential jobs (1.7B, step-credit seeds/600, no-check 600, secondary-control
seeds, lid0.5, cor0.1, Yoruba/Shona/Sotho); kept: seeds 43/44 of B/C/D/G, 8B B/C/D/G, Zulu-only D/G, Swahili+Zulu,
lid0.9/detsw2/detwl, cor0.2/0.4. (2) Round-2 re-voting dropped: GSM8K votes are already 96.6% gold, and the vote errors
on harder data sit on olympiad problems needing 4k–9k+ tokens (COPSD: 88.2% maj-correct; errors concentrated at median
length ≥ 3k tokens) against our 2,048-token rollout cap.

Design: Qwen3-4B base (non-thinking, greedy) translates the same 3000 gsm8k-mt3k English questions into Swahili
(translate.self_translate; prompt in its SELF_PROMPT). QC = the NLLB QC (numbers, length ratio, loops) + lid_fraction
>= 0.7. Train our method D (`grpo__gsm8k-st3k__exact__votelidref-lr5e-6-q16-steps300__s42`) on the QC-passing,
vote-accepted questions, everything else identical to D300 (same English votes, same hyperparameters, s42). Evaluate
TEST step 300, default decoding.
Primary contrast: D(self-translated) − D(NLLB), Swahili acc (LID) on AfriMGSM TEST, paired bootstrap; also vs base+pen.
Claim if the CI of D(self) − base+pen excludes 0 and D(self) − D(NLLB) CI includes 0 or is positive: the method needs
only unlabeled English questions and the model itself (no gold answers, no gold rationales, no MT system, no judge,
no SFT stage) — the separation from SP3F. Report translation QC pass rate vs NLLB (2805/3000).

## 2026-11-19 — PREREGISTERED (Person C): SDRRL baseline, translated self-distillation SFT (before any SDRRL run)

Purpose: the supervised, offline baseline of the proposal (SDRRL, Zhang et al. 2024) and the main-conference plan's
"SFT on NLLB-translated solutions" slot, on the same questions and the same English votes as our method (D).
Data: the 2783 vote-accepted gsm8k-mt3k questions (QC-passing NLLB Swahili question; Qwen3-4B English thinking-mode
vote, share >= 5/8; vote = gold on 2687, 96.6%). Reference per question = the shortest finished English sample that
agrees with the vote (thinking + final answer). NLLB-200-3.3B (beam 4) translates it sentence by sentence with math
masked; the translation is used only if all math survives, it reads as Swahili (function-word check) and its mean
sentence CometKiwi (wmt22-cometkiwi-da) is >= 0.8. SFT set per question: {English, Swahili} question x {English
reference, Swahili reference if accepted}; questions code-switched with FreeDict (p = 0.15 per dictionary word); plus
200 MAFAND-MT sentence pairs as translation and completion instructions. No gold answers, gold rationales or judge.
Training: Qwen3-4B, LoRA r64 / alpha 128 on all projections (as every run), cross-entropy on the assistant turn, AdamW
lr 5e-5, linear schedule with 10% warmup, batch 32, 4 epochs, examples over 4096 tokens dropped, s42
(`sdrrl__gsm8k-mt3k__exact__s42`; seeds 43/44 only if s42 is informative). Compute is not matched to D: SDRRL's
recipe sets the step count (~1300 steps of SFT vs 300 RL steps); both are reported in the write-up.
Evaluation as every Qwen3-4B run: AfriMGSM TEST (200 x 12), final checkpoint only, default decoding and repetition
penalty 1.1; primary metric Swahili accuracy (LID >= 0.7, lingua over all languages), also any-language mean_acc;
paired bootstrap over items (logs/sp3f_compare.py). Reported with it: translation acceptance rate and CometKiwi
distribution (build_report.txt), share of TEST outputs that think in English.
Contrasts: SDRRL − base+pen (does translated SFT beat penalty decoding?); D − SDRRL (on-policy RL with the model's
own votes vs the MT-based SFT recipe, same votes); C − SDRRL; SDRRL − RFT (two offline SFTs: own Swahili samples vs
translated English traces).
Reading: D − SDRRL with the CI excluding 0 in D's favour = the method beats the standard translate-and-fine-tune
baseline without needing an MT system for the solutions. SDRRL >= D = the write-up must state what RL adds over
translated SFT. Note: the tgt_en subset trains Swahili questions -> English answers, so SDRRL may lose Swahili accuracy
(LID) while gaining any-language accuracy; both are reported and the primary metric stays Swahili accuracy.
Prediction: <Person C fills this in before the first SDRRL run starts>
