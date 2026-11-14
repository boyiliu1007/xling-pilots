"""Prompt styles: which language the model thinks and answers in, for both evaluation and training.

`copsd` is COPSD's official Swahili prompt (Swahili instruction, thinking pre-started in Swahili). The other styles
keep the Swahili question but let the model think in English, where it reasons well (Experiment 2 finding: in
Swahili the base model and the teacher loop from ~3% of the way in). Instructions are in English (the model's
control language); the thinking prefix follows COPSD's own English one ("By request, I will start thinking in
English.", multilingual_grpo_train.py). A style overrides the Swahili entries of COPSD's LANGUAGE_CONFIG (training
collator) and evaluate_math.LANGUAGE_PROMPTS (evaluation) in place; apply_style("copsd") restores the originals.
"""
from .collator import import_copsd, import_copsd_eval

EN_PREFIX = "By request, I will start thinking in English."
STYLES = {
    "copsd": None,
    # think in English, final answer (explanation after </think>) in Swahili
    "en-think-sw": dict(
        instruction="Please reason step by step in English. Then write your final answer in Swahili, and put the "
                    "final answer within \\boxed{}.",
        think_prefix=EN_PREFIX),
    # think and answer in English (only the question is Swahili)
    "en-think-en": dict(
        instruction="Please reason step by step in English, write your final answer in English, and put the final "
                    "answer within \\boxed{}.",
        think_prefix=EN_PREFIX),
    # translate the question first, then think in English, answer in Swahili
    "selftrans-sw": dict(
        instruction="First translate the question into English. Then reason step by step in English. Finally write "
                    "your final answer in Swahili, and put the final answer within \\boxed{}.",
        think_prefix="By request, I will first translate the question into English, then think in English.\n\n"
                     "English translation of the question:"),
}
EN_TRANSITION = ("After reading the English reference solution above, make sure you truly understand the logic behind "
                 "each step—do not copy it or merely paraphrase it. Now, using your own words and independent "
                 "reasoning, solve the original problem.")
_ORIG = {}


def apply_style(style="copsd"):
    if style not in STYLES:
        raise SystemExit(f"unknown prompt style {style}; known: {list(STYLES)}")
    import_copsd()
    import language_config as LC
    EM = import_copsd_eval()
    cfg, ev = LC.LANGUAGE_CONFIG["SWA"], EM.LANGUAGE_PROMPTS["SWA"]
    if not _ORIG:
        _ORIG.update(cfg={k: cfg[k] for k in ("student_instruction", "teacher_final_instruction", "think_prefix",
                                              "transition_prompt")},
                     ev={k: ev[k] for k in ("instruction", "think_prefix")})
    cfg.update(_ORIG["cfg"])
    ev.update(_ORIG["ev"])
    s = STYLES[style]
    if s:
        cfg.update(student_instruction=s["instruction"], teacher_final_instruction=s["instruction"],
                   think_prefix=s["think_prefix"], transition_prompt=EN_TRANSITION)
        ev.update(instruction=s["instruction"], think_prefix=s["think_prefix"])
    return style
