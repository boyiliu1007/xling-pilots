"""Experiment 2: capability-conditioned analysis of privileged information (proposal §3.2).

What the teacher sees at each level (the student always sees only the Swahili question):
  L0  no teacher, no training: the frozen base model (the Swahili-only baseline)
  L1  Swahili question + English question
  L2  L1 + majority-voted English final answer            (not implemented yet)
  L3  L2 + an agreeing English reasoning trace             (not implemented yet)

Every level is evaluated with the shared COPSD protocol (votedistill.copsd.evaluate) on the fixed AfriMGSM-swa
DEV/TEST split, and its outputs go to results/exp2/<level>/.
"""
