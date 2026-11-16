"""Szu-Ju's experiments on top of Experiment 2 (branch szujuc/understanding-uopsd).

Two questions Experiment 2 left open:
  1. Does training help or hurt English, and does it help the model *understand* a Swahili question (rather than only
     finish and box an answer)? `xeval` evaluates any LoRA checkpoint in three views of the same 200 TEST items:
       swsw  Swahili question, Swahili thinking (COPSD protocol)            base = results/exp2/L0
       swen  Swahili question, English thinking and answer                    base = results/exp2/L0-enthink-en
       enen  English question, English thinking (English AfriMGSM)            base = results/exp2/L0-eng
     swen isolates question comprehension: the reasoning language is English, where the model is strong
     (base enen 0.886 vs swen 0.072 mean_acc). The enen view measures what training does to English.
  2. U-OPSD baseline (Li et al. 2026b; proposal 5.3): single-language unsupervised OPSD on the *English* questions.
     Student = English question; teacher = the same frozen model + the longest English rollout agreeing with the
     majority vote (questions with vote share >= 1/2); loss only on student rollouts whose answer disagrees with the
     vote. If improving English alone moved Swahili, cross-lingual training would be unnecessary.

Results go to results/xfer/, checkpoints to CKPT_ROOT/xfer/.
"""
