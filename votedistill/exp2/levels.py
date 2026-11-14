"""Teacher-context levels of Experiment 2 and where their outputs live.

A teacher context is (include_problem_en, include_reference, reference source). Every level with a reference uses
the official COPSD collator's reference slot and wording, so L2, L3 and L3-shuffled differ only in what the slot
contains:
  gold           the released COPSD English reference solution
  gold_answer    "The final answer is \\boxed{<gold Answer>}." (answer only, no reasoning)
  shuffled_gold  the gold solution of a different problem (fixed-seed derangement): placebo for reference content
  vote           the model's own English majority-vote solution (exp2/trainsets.py; needs a consensus run)
  vote_answer    "The final answer is \\boxed{<majority answer>}."
  shuffled_vote  the voted solution of a different problem
"""
import random

from ..paths import CKPT_ROOT, MODEL, RESULTS

# Qwen3-1.7B (the main model) keeps results/exp2; another BASE_MODEL gets its own tree, e.g. results/exp2_qwen3-4b
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MODEL_TAG = "" if MODEL == DEFAULT_MODEL else "_" + MODEL.split("/")[-1].lower()
EXP2_RESULTS = RESULTS / f"exp2{MODEL_TAG}"
EXP2_CKPT = CKPT_ROOT / f"exp2{MODEL_TAG}"
SHUFFLE_SEED = 42

# Evaluation-level definitions (results/exp2/<level>/).
LEVELS = {
    "L0": {"trains": False, "teacher": "none (frozen base model, no training)"},
    # Base model on the row-aligned ENGLISH AfriMGSM questions: the label-free English side of the capability groups.
    "L0-eng": {"trains": False, "lang": "eng", "teacher": "none (frozen base model on English AfriMGSM; capability signal)"},
    # Prompt-only baselines on the Swahili questions (copsd/styles.py): think in English instead of Swahili.
    "L0-enthink-sw": {"trains": False, "style": "en-think-sw", "teacher": "none; base model thinks in English, answers in Swahili"},
    "L0-enthink-en": {"trains": False, "style": "en-think-en", "teacher": "none; base model thinks and answers in English"},
    "L0-selftrans-sw": {"trains": False, "style": "selftrans-sw",
                        "teacher": "none; base model translates the question to English, thinks in English, answers in Swahili"},
    # Decoding-only controls on the official Swahili prompt (no training): more randomness / repetition penalty.
    "L0-temp1.3": {"trains": False, "sampling": {"temperature": 1.3}, "teacher": "none; sampling temperature 1.3"},
    "L0-temp1.5": {"trains": False, "sampling": {"temperature": 1.5}, "teacher": "none; sampling temperature 1.5"},
    "L0-reppen1.1": {"trains": False, "sampling": {"repetition_penalty": 1.1}, "teacher": "none; repetition penalty 1.1"},
    "L0-reppen1.05": {"trains": False, "sampling": {"repetition_penalty": 1.05}, "teacher": "none; repetition penalty 1.05"},
    "L0-reppen1.2": {"trains": False, "sampling": {"repetition_penalty": 1.2}, "teacher": "none; repetition penalty 1.2"},
    "L0-qwen3-4b": {"trains": False, "sampling": {"model": "Qwen/Qwen3-4B"}, "teacher": "none; bigger model Qwen3-4B"},
    "L0-qwen3-4b-reppen1.1": {"trains": False, "sampling": {"model": "Qwen/Qwen3-4B", "repetition_penalty": 1.1}, "teacher": "none; Qwen3-4B, repetition penalty 1.1"},
    "L0-qwen3-8b": {"trains": False, "sampling": {"model": "Qwen/Qwen3-8B"}, "teacher": "none; bigger model Qwen3-8B"},
    "L0-qwen3-8b-reppen1.1": {"trains": False, "sampling": {"model": "Qwen/Qwen3-8B", "repetition_penalty": 1.1}, "teacher": "none; Qwen3-8B, repetition penalty 1.1"},
    "L0-reppen1.3": {"trains": False, "sampling": {"repetition_penalty": 1.3}, "teacher": "none; repetition penalty 1.3"},
    "L1": {"trains": True, "teacher": "Swahili question + English question",
           "include_problem_en": True, "include_reference_solution_en": False},
}

# Teacher contexts used by the pre-training screen (and later by training).
TEACHER_CONTEXTS = {
    # For Swahili, COPSD's language_config overwrites teacher_final_instruction with student_instruction, so this
    # teacher prompt is token-identical to the student prompt: training is a no-op; in the screen D must be ~0.
    "L0-same": dict(problem_en=False, reference=None, note="same prompt as the student (correctness check)"),
    "L1": dict(problem_en=True, reference=None, note="+ English question"),
    "L2-gold": dict(problem_en=True, reference="gold_answer", note="+ gold final answer only"),
    "L3-gold": dict(problem_en=True, reference="gold", note="+ gold English solution (= gold COPSD)"),
    "L3-shuffled-gold": dict(problem_en=True, reference="shuffled_gold",
                             note="+ gold English solution of a different problem (placebo)"),
    "L2-vote": dict(problem_en=True, reference="vote_answer", note="+ English majority answer only"),
    "L3-vote": dict(problem_en=True, reference="vote", note="+ the model's own English majority-vote solution"),
    "L3-shuffled-vote": dict(problem_en=True, reference="shuffled_vote",
                             note="+ voted English solution of a different problem (placebo)"),
}

# Earlier base-model evaluation with the same protocol (git 4e5abab, another user's environment). L0 is checked
# against it item by item to confirm that this environment reproduces the setup.
REFERENCE_BASE = RESULTS / "copsd_gold_eval4096" / "eval"


def run_dir(level):
    if level not in LEVELS:
        raise SystemExit(f"unknown level {level}; known: {', '.join(LEVELS)}")
    return EXP2_RESULTS / level


def derangement(n, seed=SHUFFLE_SEED):
    """Permutation of range(n) without fixed points: a random order, each element mapped to the next one (a cycle)."""
    if n < 2:
        raise ValueError("a derangement needs n >= 2")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    perm = [0] * n
    for i in range(n):
        perm[order[i]] = order[(i + 1) % n]
    return perm


def answer_only(answer):
    return f"The final answer is \\boxed{{{answer}}}."


def teacher_rows(rows, context):
    """Rows (already through copsd.collator.prepare_rows) with `solution` set to this context's reference.
    Returns (rows, collator_flags)."""
    spec = TEACHER_CONTEXTS[context]
    ref = spec["reference"]
    flags = dict(include_problem_en=spec["problem_en"], include_reference_solution_en=ref is not None)
    if ref is None or ref == "gold":
        return [dict(r) for r in rows], flags
    if ref == "gold_answer":
        return [{**r, "solution": answer_only(r["Answer"])} for r in rows], flags
    if ref == "shuffled_gold":
        perm = derangement(len(rows))
        return [{**r, "solution": rows[perm[i]]["solution"]} for i, r in enumerate(rows)], flags
    if ref == "vote":
        return [{**r, "solution": r["vote_solution"]} for r in rows], flags
    if ref == "vote_answer":
        return [{**r, "solution": answer_only(r["vote_answer"])} for r in rows], flags
    if ref == "shuffled_vote":
        perm = derangement(len(rows))
        return [{**r, "solution": rows[perm[i]]["vote_solution"]} for i, r in enumerate(rows)], flags
    raise ValueError(f"reference source {ref} is not implemented")
