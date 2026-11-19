"""Label-free SDRRL baseline (Zhang et al., 2024): offline SFT on the model's own voted English traces,
machine-translated into the target language, plus code-switched questions and a little natural parallel text."""


def exp2_data_dir(data):
    """Shared SDRRL data folder of an Experiment 2 training set (translations, QE scores, sft_data.jsonl), per base
    model: results/exp2[_<model>]/sdrrl/<data>. Imported lazily: the CometKiwi environment loads this package too."""
    from ..exp2.levels import EXP2_RESULTS
    return EXP2_RESULTS / "sdrrl" / data


def exp2_votes(data):
    """The base model's English thinking-mode consensus file for an Experiment 2 training set."""
    from ..exp2.trainsets import CONSENSUS, CONSENSUS_FILE
    return CONSENSUS / CONSENSUS_FILE[data]
