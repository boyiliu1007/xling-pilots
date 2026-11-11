"""Student / teacher prompts, built with the official COPSD data collator.

The student sees only the target-language question. The teacher (the same frozen base model) additionally sees
privileged information: the English question and an English reference solution. Changing what the teacher sees
(a majority-voted reference instead of gold, answer-only, no reference) is the main lever of the proposal's
experiments. The collator reads the reference from each row's `solution` field, so a voted pseudo-reference
is a replacement of `solution` in `prepare_rows`; an answer-only teacher needs a new prompt path in the collator.
"""
import sys

from ..paths import COPSD_DIR


def import_copsd():
    """Make the COPSD checkout importable (multilingual_data_collator, language_config, evaluate_math)."""
    for p in (COPSD_DIR, COPSD_DIR / "african_langs_eval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if not (COPSD_DIR / "multilingual_data_collator.py").exists():
        raise SystemExit(f"COPSD checkout not found at {COPSD_DIR}; clone https://github.com/cisnlp/COPSD and set COPSD_DIR")


def prepare_rows(rows, lang):
    """Map released COPSD training rows to the columns the official collator expects."""
    code = lang.upper()
    out = []
    for r in rows:
        r = dict(r)
        # The released data stores the translation as `problem_translated`; the official script expects `problem_<lang>`.
        r[f"problem_{lang}"] = r["problem_translated"]
        r["target_lang"] = code
        r["problem_en"] = r["problem"]
        out.append(r)
    return out


def make_collator(tok, max_length, include_problem_en=True, include_reference_solution_en=True):
    """Official collator, thinking-mode student. Defaults reproduce gold COPSD (teacher sees English question + solution)."""
    import_copsd()
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    return MultilingualSelfDistillationDataCollator(tok, max_length=max_length, reason_first=False,
                                                    student_enable_thinking=True,
                                                    include_problem_en=include_problem_en,
                                                    include_reference_solution_en=include_reference_solution_en)
