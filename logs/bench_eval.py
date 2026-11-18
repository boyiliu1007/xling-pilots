"""MSVAMP Swahili (1000 items, 4 samples, otherwise the TEST sampling config) for the base model or a run's final
checkpoint, into <EXP2_RESULTS>/bench/<name>__msvamp/ (PREREGISTRATION 2026-11-18 wave 2).
usage: python logs/bench_eval.py <run name | base> [step]"""
import os, sys
from votedistill.copsd.config import EVAL
from votedistill.copsd.evaluate import evaluate
from votedistill.exp2.runs import EXP2_RESULTS, final_step, paths
from votedistill.utils import setup_logging

name = sys.argv[1]
pen = os.environ.get("REPPEN")   # e.g. 1.1: the repetition-penalty baseline decoding
out = EXP2_RESULTS / "bench" / (f"{name}__msvamp" + (f"__reppen{pen}" if pen else ""))
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "logs" / "eval.log")
cfg = {**EVAL, "val_n": 4, **({"repetition_penalty": float(pen)} if pen else {})}
if name == "base":
    evaluate(out, ["base"], "test", cfg, None, bench="msvamp")
else:
    st = int(sys.argv[2]) if len(sys.argv) > 2 else final_step(name)
    evaluate(out, [f"step{st}"], "test", cfg, paths(name)[1], bench="msvamp")
print(out)
