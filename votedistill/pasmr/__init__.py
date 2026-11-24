"""PASMR baseline (Zhao et al., 2026, "Align to the Pivot", arXiv 2601.17671), adapted to Experiment 2.

Stage 1, PAM (pivot-aligned mapping): SFT from the Swahili question to "<translation_start> English question
<translation_end>" followed by the Swahili solution (votedistill/pasmr/data.py; trainer: votedistill/sdrrl/train.py).
Stage 2, SRL (self-feedback RL): starting from PAM, the policy answers the Swahili question with its pivot; the pivot
English question is fed back to the current policy for an English answer; reward 1 if the two answers agree, 0.1 if
not, 0 on a format error; REINFORCE++ with a per-token KL penalty to PAM (votedistill/pasmr/srl.py).
SFT and RL use disjoint halves of the training questions, as in the paper. PREREGISTRATION 2026-11-22 PASMR entry.
"""

PIVOT_OPEN, PIVOT_CLOSE = "<translation_start>", "<translation_end>"   # the delimiters of the authors' code


def data_dir(data):
    """Shared PASMR data folder (split, PAM SFT set) per base model: results/exp2[_<model>]/pasmr/<data>."""
    from ..exp2.levels import EXP2_RESULTS
    return EXP2_RESULTS / "pasmr" / data
