# Szu-Ju's experiments on Experiment 2 (branch `szujuc/understanding-uopsd`)

> Results and verdicts: [`README.md`](README.md). This file is the plan as written before training; it is left unchanged.

Written 2026-11-16, before any of the runs below were trained. Model Qwen3-1.7B, AfriMGSM Swahili, Experiment 2's
DEV 50 / TEST 200 split, COPSD's evaluation protocol (12 samples, T 1.0, top-p 0.95, 4096 tokens, thinking mode).
Every number is reported at the **final checkpoint (step 100)**; there is no DEV selection (Experiment 2 found DEV
selection close to random). Gold labels are used only for evaluation.

## Why

Experiment 2's base-model rows already separate *reading* from *reasoning* (TEST, mean_acc / maj@12):

| Question -> thinking | mean_acc | maj@12 | boxed | truncated |
|---|---|---|---|---|
| English -> English (L0-eng) | 0.886 | 0.915 | 97% | 4% |
| Swahili -> English (L0-enthink-en) | 0.072 | 0.090 | 87% | 17% |
| Swahili -> Swahili (L0) | 0.028 | 0.090 | 23% | 77% |
| Swahili -> Swahili, penalty 1.1 (L0-reppen1.1) | 0.081 | 0.105 | 98% | 2% |

With English reasoning and clean output, accuracy on Swahili questions is still 0.072: the model misreads the question.
COPSD gains so far are loop-breaking (Experiment 2). No trained checkpoint has been evaluated on English yet.

## Q1. What does training do to English, and to reading? (`xeval`, all runs below + Experiment 2 adapters if shared)

Every checkpoint is evaluated in four views (`votedistill/xfer/xeval.py`), each paired with its base level:
`swsw` (vs L0), `swen` (vs L0-enthink-en), `enen` (vs L0-eng), `swsw-rp` (vs L0-reppen1.1). Reported per view: deltas
with paired-bootstrap CIs, maj@12 fixed/broken, plus the **comprehension gap** (enen − swen mean_acc, per item) and
the **transfer set** (base solves from English but not Swahili→English) fixed share.

## Q2. U-OPSD baseline (proposal 5.3) — `uopsd__gsm8k-mt__clip__s{42,43}`

Student = English question (English side of Experiment 2's GSM8K train sample); teacher = same frozen model + the
longest finished English rollout agreeing with the 8-sample majority (vote share >= 1/2; 461/466 questions, vote
correct on 433/461); loss only on rollouts whose answer disagrees with the vote. Official COPSD hyperparameters and
clip, exact layout. Baseline, so no pass bar: it answers "does improving English alone move Swahili?"
Expectation: enen within ±2 (already 0.886), swsw ≈ 0 vs L0.

## Q3. Reading-comprehension OPSD — `rc__gsm8k-mt__first256__noclip__s{42,43}`

Student = Swahili question, thinks and answers in English (style en-think-en); teacher = context L1 (+ English question,
no solution; label-free). Loss only on the first 256 completion tokens (where the problem is restated; clipstats:
45–59% of the KL is in the first 10% of a rollout), no clip.
**Pass:** swen mean_acc − L0-enthink-en > 0 with the 95% CI excluding 0 in both seeds, AND enen drop no worse than −2
points. Secondary: comprehension gap shrinks; transfer-set fixed share.
Closest existing run: `L1__copsd__exact__enthink__s{42,43}` (full sequence, with clip, en-think-sw, COPSD data):
step 100 mean_acc 0.065 / 0.065 vs its base 0.075 — no gain.

## Q4. What does the clip do? (one seed each, exploratory)

- `rc__gsm8k-mt__first256__clip__s42` vs Q3 s42: does the official clip remove the reading signal?
- `rc__gsm8k-mt__all__noclip__s42` vs Q3 s42: does restricting to the first 256 tokens matter?
- `uopsd__gsm8k-mt__noclip__s42` vs Q2 s42: does the clip matter when the student does not loop (English rollouts
  are ~4% truncated)?
Existing evidence (Experiment 2, 1.7B): with English thinking, clip and no clip end equal (L3-vote enthink step 100
0.062 / 0.052 vs enthink-noclip 0.057 / 0.057), and clipstats shows the clipped gradient anti-aligned with the KL
gradient (cos −0.18) and lowering the teacher's top-1 token at 2–12% of positions.
A one-seed result here is reported as exploratory; it needs a second seed before it is a finding.

## Runs

`NAME=<run> sbatch slurm/xfer_run.sbatch` (train 100 steps, then the four views). Smoke test: seed s1, 10 steps.
