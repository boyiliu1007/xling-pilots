"""Teacher-context levels of Experiment 2 and where their outputs live.

A teacher context is (include_problem_en, include_reference, reference source). Every level with a reference uses
the official COPSD collator's reference slot and wording, so L2, L3 and L3-shuffled differ only in what the slot
contains:
  gold           the released COPSD English reference solution
  gold_answer    "The final answer is \\boxed{<gold Answer>}." (answer only, no reasoning)
  shuffled_gold  the gold solution of a different problem (fixed-seed derangement): placebo for reference content
Label-free variants (vote, vote_answer, shuffled_vote) need an English thinking-mode consensus run first.
"""
import random

from ..paths import CKPT_ROOT, RESULTS

EXP2_RESULTS = RESULTS / "exp2"
EXP2_CKPT = CKPT_ROOT / "exp2"
SHUFFLE_SEED = 42

# Evaluation-level definitions (results/exp2/<level>/).
LEVELS = {
    "L0": {"trains": False, "teacher": "none (frozen base model, no training)"},
    # Base model on the row-aligned ENGLISH AfriMGSM questions: the label-free English side of the capability groups.
    "L0-eng": {"trains": False, "lang": "eng", "teacher": "none (frozen base model on English AfriMGSM; capability signal)"},
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
    raise ValueError(f"reference source {ref} is not implemented")
