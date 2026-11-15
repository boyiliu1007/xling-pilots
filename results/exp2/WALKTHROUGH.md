# Experiment 2 walkthrough: what we ran, what we found, where it stands

Branch `jeff/exp2-levels` (built on `refactor-votedistill`). Status as of 2026-11-15.
Every design decision was written into [`PREREGISTRATION.md`](PREREGISTRATION.md) before the runs it concerns, and
every number below is in a committed `SUMMARY.txt` (`results/exp2/`, `results/exp2_qwen3-4b/`,
`results/exp2_qwen3-8b/`).

## TL;DR

1. **COPSD's teacher information never transferred.** With Qwen3-1.7B on Swahili, a teacher shown the gold English
   solution does no better than one shown another problem's solution (placebo). With the model's own English
   majority vote as the reference, Qwen3-4B gets *worse* (−15 maj@12 vs the teacher with only the English question).
2. **Almost all of the gains we saw are "stop looping".** The base model spends its token budget on repetition
   loops in Swahili (77% of 1.7B samples are truncated). A repetition penalty of 1.1 at decoding, with no training,
   beats every COPSD run.
3. **Our gold-free method:** penalty-decoded self-SFT. The model's own Swahili answers are sampled with that penalty,
   filtered by its own English majority vote, and used to fine-tune it. This teaches the model to stop looping and
   keeps it answering in Swahili, but in a single round it only ties (1.7B, 4B) or trails (8B) the
   decoding-only baseline.
4. **Does the English vote matter (the proposal's claim)?** For one round of self-SFT it did not: training on answers
   that are 7–41% correct worked as well as training on 98–100% correct ones (1.7B, 4B, 8B; small and large data).
   **First positive evidence:** in a *second* round on Qwen3-8B, the English-vote filter beats no filter by
   +2.3 / +2.5 mean accuracy points (both seeds, CIs exclude 0) and matches the gold-answer oracle.
5. **Running now:** round 3 on Qwen3-8B, to check whether that advantage grows, holds, or was a one-off.

## Setup (common to everything)

- **Task:** AfriMGSM Swahili, with a fixed split: DEV 50 items (checkpoint selection only) and TEST 200 items (all
  reported numbers).
- **Evaluation:** COPSD's protocol. Thinking mode, 12 samples per question at temperature 1.0 and top-p 0.95, up to
  4096 new tokens, and COPSD's own answer grader.
- **Metrics:**
  - **maj@12:** the majority answer is correct. This is the preregistered primary metric.
  - **pass@12:** any of the 12 samples is correct.
  - **mean_acc:** the average per-sample accuracy. It is the least noisy, so we lean on it.
  - **Mechanism:** truncation rate, loop rate, accuracy among finished samples, and the share of the answer written in
    Swahili.
- **Statistics:** paired bootstrap over the 200 items (10,000 resamples). Two training seeds (42, 43), always
  reported separately. A "significant" result below means the 95% CI excludes 0 in both seeds.
- **Training:** COPSD's official hyperparameters (LoRA rank 64, 100 steps, batch 32). After we found that DEV
  selection is close to random (one DEV item = 2 points), the primary numbers use the **final checkpoint (step 100)**.
- **Gold labels** are used only for evaluation and for oracle arms that give an upper bound. No candidate method uses
  them.

**Base models (no training), TEST, default decoding:**

| Model | maj@12 | mean_acc | truncated | + repetition penalty 1.1: maj@12 / mean_acc |
|---|---|---|---|---|
| Qwen3-1.7B | 0.090 | 0.028 | 77% | 0.105 / 0.081 |
| Qwen3-4B | 0.265 | 0.144 | 31% | 0.340 / 0.232 |
| Qwen3-8B | 0.510 | 0.285 | 47% | 0.595 / 0.445 |

That last column is the **"base + penalty" baseline**. It is the bar any trained method has to beat, and it turned
out to be the hardest comparison in the project.

## Phase 1: what does the COPSD teacher transfer? (Qwen3-1.7B)

COPSD trains a student that sees only the Swahili question toward a teacher (the same model) that sees extra
context. We varied that context:

| Level | Teacher sees |
|---|---|
| L0 | no training (baseline) |
| L1 | + the English question |
| L2-gold | + the final answer only |
| L3-gold | + the full English solution |
| L3-shuffled | + the solution of a *different* problem (placebo) |
| L3-vote | + the model's own English majority-vote solution (gold-free; this is the proposal's Experiment 1) |

We also varied the training data (COPSD's olympiad set, machine-translated GSM8K, or a mix) and the prompt layout
(COPSD's padded layout vs. an exact one).

**Result (step 100, change vs L0):** every level gains about +5 to +12 pass@12 and +1 to +2 mean_acc, with
maj@12 ≈ 0. **L3-shuffled equals L3-gold**, so the content of the reference does not matter. L3-vote is no better
than L3-gold. The gain is that the model learns to finish and box an answer.

## Phase 2: why does COPSD help at all? (mechanism)

- **The 0.05 loss clip:** COPSD clips the per-(token, vocab) loss at 0.05, and that clip acts as an entropy
  regulariser. It removes the teacher's strongly preferred tokens, raises the student's entropy and lets it escape
  loops.
  - Removing the clip hurts (L3-gold pass@12 −17).
  - A two-stage design (clip first, then no clip) undid itself. The stage-2 arm that trains on self-SFT data was
    cancelled for that reason (preregistered deviation).
- **Teacher-free controls get the same gain.** "Hot-self" distillation (teacher = the model itself at temperature
  1.5–2.5, no extra information) and an unlikelihood loss on loops both match COPSD.
- **Decoding alone does better.** A repetition penalty of 1.1 gives +5.3 mean_acc and +13.5 pass@12 over L0, and
  truncation drops from 77% to 2%. That beats every COPSD run. The dose curve peaks at 1.1: 1.05 and 1.2 help less,
  and 1.3 hurts.
- **Thinking in English** (Swahili question, English reasoning) also helps without training (+4.7 mean_acc). COPSD on
  top of it adds nothing.
- **Translation:** the model cannot translate its own reasoning into Swahili usably (EN→SW unusable by chrF++), so a
  "think in English, translate" pipeline was ruled out.

## Phase 3: bigger models and a gold-free, on-policy method (Qwen3-4B, then 8B)

Hypothesis: the teacher's information can't transfer because the teacher is scoring degenerate, looping student
text. We tested two ways to make the student's text coherent:

- **`rp` (penalty-rollout COPSD):** COPSD unchanged, except that the student's training rollouts are sampled with
  repetition penalty 1.1.
- **PD-SFT (penalty-decoded self-SFT):** sample the model's own Swahili answers with penalty 1.1, keep those that
  finished, are in Swahili and agree with its own English majority vote, then fine-tune on them.

**Result (Qwen3-4B):**
- COPSD with the voted reference fails outright. L3-vote maj@12 is 0.185 on both seeds, against about 0.33 for L1:
  the transfer prediction (P3) fails.
- PD-SFT is the only method that breaks loops while keeping Swahili. It ties base + penalty.

## Phase 4: does the English vote matter? (the filter ablation)

This is the key question for the proposal. Is it the model's *English understanding* (via the vote) that teaches
Swahili, or would any finished Swahili answer do?

**Arms:** one shared sample pool, four filters (`selfsft.FILTERS`). Each arm keeps at most 2 samples per question and
is subsampled to the size of the smallest arm, so data size is equal across arms.

| Arm | Keeps a Swahili sample if its answer equals |
|---|---|
| `vote` | the model's English majority vote (**our method**, gold-free) |
| `swvote` | the majority of its own Swahili samples (gold-free, no English) |
| `nofilter` | anything |
| `gold` | the gold answer (oracle upper bound, analysis only) |

**Preregistered contrasts** (mean_acc and maj@12, per seed):

| Contrast | Definition | Question it answers |
|---|---|---|
| F1 | vote − swvote | Does English add over Swahili self-consistency? |
| F2 | vote − nofilter | Does any filter matter? |
| F3 | gold − vote | How much would a perfect verifier add? |
| G | gold − nofilter | Does answer correctness matter at all? |

**Round 1 (each arm trained once from the base model):**

| Setting | Rows per arm | Share of training answers correct (vote / nofilter / gold) | Result |
|---|---|---|---|
| Small: COPSD olympiad set, 30 steps, 1.7B and 4B | 208 / 305 | 0.87–0.90 / 0.18–0.24 / 1.0 | F1, F2, F3 ≈ 0 |
| Large: 3000 GSM8K-MT questions, 100 steps, 1.7B | 751 | 0.97 / 0.07 / 1.0 | F1, F2, F3, G ≈ 0 (all within ±1) |
| Large: 3000 GSM8K-MT questions, 100 steps, 8B | 3166 | 0.98 / 0.41 / 1.0 | F1, F2, F3, G within ±2; seeds disagree in sign |

- **One round teaches form, not reasoning.** Every arm gains about the same:
  - 1.7B: +4 to +5 mean_acc;
  - 8B: +10 to +12 mean_acc;
  - truncation falls from 47% to 11% on 8B.
  On 8B, accuracy among *finished* answers actually drops, from 0.53 to 0.45.
- **8B still loses to the decoding baseline.** Trained models are 4–5 mean_acc *below* base + penalty, and sampling
  them with the penalty brings them back level, but not above.

**Round 2 (`-r2`):** resample from each arm's round-1 checkpoint with the same filter, then train 100 more steps.

| Contrast (mean_acc, s42 / s43) | 1.7B | 8B |
|---|---|---|
| Round 2 − round 1, vote | +2.1 / +1.3 | **+3.5 / +2.6** |
| Round 2 − round 1, nofilter | +1.9 / +1.6 | +2.5 / +1.1 |
| F2 = vote − nofilter | +0.3 / 0.0 | **+2.3 / +2.5** (CIs exclude 0) |
| G = gold − nofilter | −0.3 / +0.3 | **+3.0 / +2.5** (CIs exclude 0) |
| F3 = gold − vote | ≈ 0 | ≈ 0 (0.428 / 0.430 vs 0.435 / 0.430) |
| vote − base + penalty | +1.0 / +0.7 (tie) | −1.8 / −1.6 (not significant) |

**Reading:**
- **1.7B:** round 2 removes the remaining loops (truncation 0.20 → 0.02) and edges level with base + penalty, but the
  filter still does not matter.
- **8B:** after round 2, correctness starts to matter. The English vote beats no filter and is as good as gold. This
  held even though the filtered arms had slightly *fewer* rows (about 3000 vs 3166), because fewer of the model's
  own samples survive the vote.

## Current verdict

| Claim | Status |
|---|---|
| COPSD transfers English knowledge to Swahili through the teacher's reference | **Not supported.** The placebo equals gold on 1.7B, and the voted reference hurts on 4B. |
| Gains on Swahili mostly come from fixing degenerate output (loops, truncation) | **Supported**, consistently across sizes. Decoding alone does most of it. |
| Penalty-decoded self-SFT is a gold-free, on-policy way to internalise that and keep Swahili | **Supported**, but it ties (1.7B, 4B) or trails (8B) the decoding-only baseline. |
| The model's own English understanding (the vote filter) improves Swahili reasoning | **Not in one round. Emerging at 8B in round 2** (+2.5 mean_acc, equal to gold). One model, two seeds; round 3 is the replication. |

## Status and what's running

- **Round 3 on Qwen3-8B** (`rp-{vote,nofilter,gold}-r3__s{42,43}`) was submitted around midday 2026-11-15 and
  preregistered beforehand (commit `4091901`).
  - "The advantage keeps growing" requires F2 in round 3 to exceed F2 in round 2 on both seeds, with CIs excluding 0.
  - Otherwise the write-up reports a one-step separation that saturates.
  - A watchdog commits the results automatically when they finish.
- **Honest expectation:** F2 holds at about +2 to +3, and the vote arm ties base + penalty rather than beating it.
  Strict "keeps growing" is maybe 30%.
- **If it saturates, the next lever is the vote's coverage, not more rounds.** The vote keeps the same ~1700 easier
  questions every round. Accepting more or harder questions, or weighting them, would put new information into
  training.

## Discussion points

1. **The decoding baseline.** Do we frame the paper around beating base + repetition penalty? It is a cheap,
   training-free baseline that every trained method so far fails to beat clearly.
2. **Reporting the negative results.** Is "COPSD's Swahili gain is loop-breaking, not information transfer" (placebo
   = gold, teacher-free controls match, decoding beats it) a contribution on its own?
3. **The round-2 effect on 8B.** Is it the core positive result? Should round 3, a third seed, or Yoruba come first?
4. **Caveats to keep in mind:**
   - vLLM's repetition penalty also counts prompt tokens.
   - The 1.1 setting was chosen a priori, but the follow-ups exist because it worked; the dose curve is the check.
   - GSM8K-MT is a train sample and may overlap the earlier 500-question sample. It does not overlap AfriMGSM TEST.
   - Results cover one language and two seeds.

## How to navigate the code

- **Run names:** `<context>__<data>__<layout>__[<variant>__]s<seed>`, parsed in `votedistill/exp2/runs.py`.
  - Example: `selfsft-vote__gsm8k-mt3k__exact__rp-vote-r2__s42` = self-SFT, 3000-question GSM8K-MT, exact layout,
    penalty-decoded pool, English-vote filter, round 2, seed 42.
- **Key files:**

  | File | Contents |
  |---|---|
  | `votedistill/exp2/levels.py` | teacher contexts |
  | `votedistill/exp2/selfsft.py` | self-SFT: sample pool, filters, rounds, training |
  | `votedistill/exp2/trainsets.py` | datasets and votes |
  | `votedistill/exp2/translate.py` | NLLB GSM8K→Swahili translation |
  | `votedistill/consensus.py` | English majority vote |
  | `votedistill/exp2/summary.py` | every contrast in the SUMMARY files |
  | `votedistill/copsd/` | the COPSD trainer and evaluation |

- **Model choice:** results for a non-default model use `BASE_MODEL=Qwen/Qwen3-8B` (or `-4B`) and go to
  `results/exp2_qwen3-8b/` (or `-4b`).
- **Commands:**
  - Run one experiment: `NAME=<run name> sbatch slurm/exp2_run.sbatch` (preempt partition, requeues and resumes).
  - Rebuild a summary: `[BASE_MODEL=...] python -m votedistill.exp2 summary`.
  - More commands are in `README.md` → "Experiment 2".
- **Not in git:** checkpoints (`/data/user_data/jeffc2/votedistill_ckpt/exp2*`; only step-100 adapters are kept)
  and large generation pools.
