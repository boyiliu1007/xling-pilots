"""TEST evaluation of the base model or a run's final checkpoint on another AfriMGSM language (default decoding),
into results/exp2_qwen3-4b/langeval/<model>__<lang>/ (PREREGISTRATION 2026-11-17 ~22:00).
usage: python logs/lang_eval.py <run name | base> <lang>"""
import sys
from votedistill.copsd.config import EVAL
from votedistill.copsd.evaluate import evaluate
from votedistill.exp2.runs import EXP2_RESULTS, final_step, paths
from votedistill.utils import setup_logging

name, lang = sys.argv[1], sys.argv[2]
out = EXP2_RESULTS / "langeval" / f"{name}__{lang}"
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "logs" / "eval.log")
if name == "base":
    evaluate(out, ["base"], "test", dict(EVAL), None, lang=lang)
else:
    evaluate(out, [f"step{final_step(name)}"], "test", dict(EVAL), paths(name)[1], lang=lang)
print(out)
