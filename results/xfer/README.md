# xfer — Szu-Ju's experiments on branch `szujuc/understanding-uopsd`

Status as of 2026-11-17. Every result below is **exploratory**: at most two seeds per arm, and the only
pre-registered bar (Q3) is checked against `PLAN.md`, which was written before any run was trained.

**The question.** Experiment 2 (`results/exp2/WALKTHROUGH.md`) found that COPSD's gains on Swahili come from breaking
repetition loops rather than from cross-lingual transfer. This branch asks *what is actually missing*: reasoning or
reading? Can on-policy self-distillation teach the model to read Swahili?

**Short answer.** The Swahili gap is a **reading** gap. Qwen3-1.7B solves 89% of the English questions but only 7% of
the same questions in Swahili, even when it thinks in English. Self-distillation, whether on English only (U-OPSD) or
on reading (RC-OPSD), moves Swahili by at most ~2 points. No trained 1.7B checkpoint beats the untrained model with
repetition penalty 1.1. External translation (NLLB) recovers +58 points.

All numbers: AfriMGSM Swahili TEST (200 items), COPSD evaluation protocol (thinking mode, 12 samples, T 1.0,
top-p 0.95, 4096 new tokens, COPSD grader). Δ = trained − base, in points, paired bootstrap 95% CI over items. Final
checkpoint (step 100), no DEV selection.

## 1. Reading study (base models, no training): `reading/`

Code `votedistill/xfer/reading.py`, full report `reading/report.txt`. The model thinks and answers in English; only
the question text changes.

| mean_acc | English (human) | Swahili | own translation | NLLB-3.3B translation | Qwen3-8B translation | chrF++ of own translation |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.885 | 0.076 | 0.054 | 0.653 | 0.506 | 28.4 |
| Qwen3-4B | 0.958 | 0.278 | 0.254 | 0.720 | 0.563 | 46.5 |
| Qwen3-8B | 0.943 | 0.529 | 0.552 | 0.715 | — | 57.1 (NLLB 67.7) |

Gap decomposition (mean_acc points, 95% CI):

| | 1.7B | 4B | 8B |
|---|---|---|---|
| en − sw (total reading gap) | +81.0 [+76.8, +85.0] | +68.0 [+62.9, +73.0] | +41.4 [+35.6, +47.2] |
| self − sw (explicit − implicit reading) | −2.2 [−5.2, +0.8] | −2.4 [−6.9, +2.1] | +2.4 [−2.7, +7.1] |
| nllb − sw (external MT helps by) | +57.7 [+51.6, +63.7] | +44.2 [+37.7, +50.7] | +18.7 [+12.6, +24.8] |
| en − nllb (lost in MT itself) | +23.2 [+17.8, +29.0] | +23.8 [+18.3, +29.3] | +22.7 [+17.4, +28.1] |

- **Reasoning is not the bottleneck.** With the human English question every model scores 0.89–0.96.
- **Writing out a translation does not help.** Explicit − implicit ≈ 0 at every scale (all CIs include 0; maj@12 is
  even negative for 1.7B and 4B). Even on items where the 4B's own translation is faithful (n = 64), solving the
  translation (0.435) is no better than solving the Swahili (0.408). The model does not hold reading knowledge that
  a translation step could unlock.
- **The gap closes with scale** (81 → 68 → 41 points), and a better reader helps: NLLB +58/+44/+19, and the 8B's
  translation +45 for 1.7B and +31 for 4B.
- Caveat: with this translation prompt, 1.7B copies the Swahili back on 5% of items. chrF++ stays low (≈30) even on
  its genuine English attempts.

## 2. Self-distillation runs: `runs/` (training), `xeval/` (evaluation), `xeval/REPORT.txt`

Each checkpoint is evaluated in four views (`votedistill/xfer/xeval.py`), each against its own base:
`swsw` Swahili question → Swahili thinking (base L0) · `swen` Swahili question → English thinking (L0-enthink-en) ·
`enen` English → English (L0-eng) · `swsw-rp` swsw with repetition penalty 1.1 (L0-reppen1.1, the strongest
untrained baseline). Base mean_acc for 1.7B: swsw 0.028, swen 0.072, enen 0.886, swsw-rp 0.081.

| Run (Qwen3-1.7B unless noted) | swsw | swen | enen | swsw-rp | comprehension gap* | transfer set fixed (swen)** |
|---|---|---|---|---|---|---|
| **Q2 U-OPSD** `uopsd__gsm8k-mt__clip__s42` | −0.3 [−1.0, +0.4] | −1.8 [−3.1, −0.5] | **−2.9 [−4.5, −1.2]** | −1.7 [−3.0, −0.5] | −1.1 [−3.3, +1.0] | 7/165 (4%) |
| `uopsd__gsm8k-mt__clip__s43` | +0.0 [−0.7, +0.8] | −1.0 [−2.2, +0.2] | −1.9 [−3.3, −0.5] | −2.0 [−3.5, −0.5] | −1.0 [−2.8, +0.9] | 6/165 (4%) |
| **Q3 RC-OPSD** `rc__gsm8k-mt__first256__noclip__s42` | +1.2 [+0.1, +2.4] | +1.5 [−0.3, +3.2] | −0.3 [−1.2, +0.7] | **−3.0 [−4.8, −1.3]** | −1.8 [−3.8, +0.2] | 13/165 (8%) |
| `rc__gsm8k-mt__first256__noclip__s43` | +0.4 [−0.6, +1.4] | +1.5 [−0.2, +3.2] | +0.1 [−1.1, +1.4] | −2.0 [−3.8, −0.3] | −1.4 [−3.4, +0.6] | 20/165 (12%) |
| Q4 RC + clip `rc__gsm8k-mt__first256__clip__s42` | +1.8 [+0.6, +3.0] | +1.9 [+0.4, +3.4] | −0.9 [−2.0, +0.3] | −1.6 [−3.2, −0.1] | −2.8 [−4.6, −0.9] | 19/165 (12%) |
| Q4 RC all tokens `rc__gsm8k-mt__all__noclip__s42` | +0.6 [−0.3, +1.5] | +2.2 [+0.3, +4.1] | +0.9 [−0.2, +2.0] | −2.4 [−4.1, −0.8] | −1.3 [−3.5, +0.8] | 16/165 (10%) |
| Q4 U-OPSD no clip `uopsd__gsm8k-mt__noclip__s42` | +0.5 [−0.4, +1.3] | +0.4 [−0.7, +1.5] | −1.5 [−3.2, +0.2] | −2.0 [−3.4, −0.7] | −1.9 [−4.0, +0.2] | 7/165 (4%) |
| **Qwen3-4B** RC `rc__gsm8k-mt__first256__noclip__s42` (`../xfer_qwen3-4b/`) | **+2.9 [+1.0, +4.9]** | +2.0 [−0.4, +4.4] | −1.2 [−2.4, −0.2] | −0.9 [−3.5, +1.6] (maj +6.0 [0, +12]) | **−3.2 [−5.8, −0.6]** | 19/124 (15%) |

\* enen − swen mean_acc per item; negative = reads Swahili better relative to English.
\** Items the base gets right from English but wrong from Swahili→English (majority vote); share fixed after training.
4B bases: swsw 0.152, swen 0.272, enen 0.951, swsw-rp 0.233.

**Verdicts**

- **Q2 (U-OPSD, English only) — training on English does not move Swahili.** swen and swsw stay at 0 or go
  negative, and the official clip costs 2–3 points of English (`enen`), the one view where the model was already
  strong. The no-clip run keeps English within noise but still does not help Swahili.
- **Q3 (RC-OPSD, pre-registered bar) — FAIL.** Bar: swen Δ > 0 with the CI excluding 0 in both seeds, and enen
  drop ≥ −2. swen is +1.5 in both seeds, but both CIs include 0 ([−0.3, +3.2], [−0.2, +3.2]). enen is fine. The
  direction is consistent, and RC fixes 2–3× more transfer-set items than U-OPSD (8–12% vs 4%), but the effect is
  small.
- **No 1.7B run beats the untrained repetition-penalty baseline.** swsw-rp drops by 1.6–3.0 points in every
  run, so trained + penalty (≈0.05–0.065) ends below base + penalty (0.081). Part of the swsw gain is again
  loop-breaking: swsw truncation falls from 77% to 49–60% in the RC runs.
- **Q4 (clip, one seed, exploratory).** RC with the clip has the largest 1.7B swsw/swen gains, but its training
  rollouts re-enter loops (truncation 48% → 78%, boxed 61% → 33% over training; no-clip runs stay flat). Restricting
  the loss to the first 256 tokens is not what helps: all-token RC gives a similar swen gain (+2.2). For U-OPSD, only the clipped
  runs have an English drop whose CI excludes 0.
- **Qwen3-4B RC (one seed) is the only run where reading improves measurably:** swsw +2.9, comprehension gap −3.2,
  both CIs excluding 0, at a cost of −1.2 English. It still ends below the 4B's own repetition-penalty base on
  mean_acc (0.181 vs 0.233), though its maj@12 rises +6.0 with the penalty. It needs a second seed before it counts as
  a finding.

Training-log signal (`runs/*/train_loss.csv`, first 10 → last 10 steps; truncated / boxed rollouts):
U-OPSD 0.10 → 0.10 / 0.93 → 0.92 (clip), 0.06 → 0.03 / 0.95 → 0.98 (no clip);
RC first256 no clip 0.49 → 0.56 and 0.54 → 0.55; RC clip **0.48 → 0.78 / 0.61 → 0.33**; RC all 0.52 → 0.52;
4B RC 0.60 → 0.67 / 0.44 → 0.38.

## 3. Setup of each method

| | Student sees | Teacher (same frozen model) sees | Loss | Data |
|---|---|---|---|---|
| U-OPSD (`uopsd.py`, proposal §5.3) | English question | + the longest finished English rollout agreeing with the 8-sample majority (vote share ≥ 1/2; 461/466 questions, vote correct on 433) | only on rollouts that disagree with the vote | English side of Experiment 2's GSM8K-MT train sample |
| RC-OPSD (`rc.py`) | Swahili question, thinks in English | + the English question (no solution, label-free) | first 256 completion tokens (`first256`) or all (`all`) | GSM8K-MT Swahili/English pairs |

Official COPSD hyperparameters, 100 steps, LoRA. `clip` = COPSD's per-(token, vocab) JSD clip at 0.05;
`noclip` = none. Checkpoints: `/data/user_data/szujuc/votedistill_ckpt/xfer{,_qwen3-4b}/<run>/step-100/adapter`.

## 4. Reproduce

```bash
PY=/data/user_data/$USER/venvs/votedistill/bin/python
NAME=rc__gsm8k-mt__first256__noclip__s42 sbatch slurm/xfer_run.sbatch        # train 100 steps + 4 views
BASE_MODEL=Qwen/Qwen3-4B NAME=... sbatch slurm/xfer_run.sbatch               # 4B: results go to results/xfer_qwen3-4b
LABEL=<name> CKPT=<dir with step-100/> sbatch slurm/xfer_run.sbatch         # evaluate an existing adapter
MODEL=Qwen/Qwen3-8B sbatch slurm/xfer_reading.sbatch                         # reading study (run 8B first: x8b needs it)
$PY -m votedistill.xfer report                                               # -> xeval/REPORT.txt
$PY -m votedistill.xfer reading-report                                       # -> reading/report.txt
```
Run names: `<method>__<data>__[<tokens>__]<clip>__s<seed>`. All jobs are idempotent: requeue a preempted job.

## 5. Not done yet

- Second seed for the Q4 arms and for the 4B RC run.
- English evaluation of Jeff's Experiment 2 COPSD adapters (permission denied on `/data/user_data/jeffc2`).
- Proposal §6.2 LLM-judge trace labels and GPU-hour / token cost; §6.4 per-token KL dump (top-divergence tokens).

## Files

| Path | Contents |
|---|---|
| `PLAN.md` | Pre-registration for Q1–Q4, written before training |
| `reading/<model>/` | `trans_self.jsonl` (own translations), `solve_<cond>.jsonl` (12 samples per item), `config.json` |
| `reading/report.txt` | Translation quality, solve accuracy, gap decomposition, faithful/unfaithful split |
| `runs/<run>/` | `config.json`, `train_loss.csv`, `train_rows.jsonl` (training questions), logs |
| `xeval/<run>__<view>/` | `eval/step100_test.jsonl` (every generation), `checkpoint.txt`, `eval_config.json` |
| `xeval/REPORT.txt` | Every 1.7B run, all views; `../xfer_qwen3-4b/xeval/REPORT.txt` for 4B |
