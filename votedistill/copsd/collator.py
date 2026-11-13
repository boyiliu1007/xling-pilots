"""Student / teacher prompts, built with the official COPSD data collator.

The student sees only the target-language question. The teacher (the same frozen base model) additionally sees
privileged information: the English question and an English reference solution. Changing what the teacher sees
(a majority-voted reference instead of gold, answer-only, no reference) is the main lever of the proposal's
experiments. The collator reads the reference from each row's `solution` field, so a voted pseudo-reference
is a replacement of `solution` in `prepare_rows`; an answer-only teacher needs a new prompt path in the collator.
"""
import os
import subprocess
import sys

from ..paths import COPSD_COMMIT, COPSD_DIR

_CACHE_ENV = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "VLLM_CACHE_ROOT")


def import_copsd():
    """Make the COPSD checkout importable (multilingual_data_collator, language_config, evaluate_math)."""
    if not (COPSD_DIR / "multilingual_data_collator.py").exists():
        raise SystemExit(f"COPSD checkout not found at {COPSD_DIR}; clone https://github.com/cisnlp/COPSD and set COPSD_DIR")
    head = subprocess.run(["git", "-C", str(COPSD_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not head.startswith(COPSD_COMMIT):
        raise SystemExit(f"COPSD checkout at {COPSD_DIR} is at {head or '?'}, expected {COPSD_COMMIT}")
    for p in (COPSD_DIR, COPSD_DIR / "african_langs_eval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def import_copsd_eval():
    """Import COPSD's african_langs_eval/evaluate_math.py without its import-time side effect: the module assigns
    HF_HOME, HF_*_CACHE and VLLM_CACHE_ROOT to the literal placeholder "YOUR PATH/..." (a cwd-relative directory).
    The HF libraries are imported first so their cache constants are already fixed, and the environment is restored."""
    import_copsd()
    import datasets, huggingface_hub, transformers, vllm  # noqa: E401,F401  freeze cache paths before the override
    saved = {k: os.environ.get(k) for k in _CACHE_ENV}
    import evaluate_math
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return evaluate_math


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
