"""Experiment 2 training runs: one run = (teacher context, training set, prompt layout, seed).

Outputs results/exp2/runs/<name>/ (config.json, train_loss.csv, train_rows.jsonl, eval/, report.txt);
LoRA checkpoints under CKPT_ROOT/exp2/<name>/. Training is the shared COPSD trainer (copsd/train.py) with the official
hyperparameters; only the teacher's rows, the data and the layout change. Evaluation is the COPSD protocol on the
fixed DEV/TEST split; the checkpoint is selected by DEV maj@12 (PREREGISTRATION.md). The base model's evaluation is
L0 (results/exp2/L0, same protocol), copied into each run so the run's report is self-contained.
"""
import logging
import shutil

from ..utils import write_jsonl
from .levels import EXP2_CKPT, EXP2_RESULTS, TEACHER_CONTEXTS, run_dir as level_dir, teacher_rows
from .trainsets import DATASETS, training_rows

log = logging.getLogger("votedistill.exp2.runs")
RUNS = EXP2_RESULTS / "runs"
SELECT_METRIC = "maj_correct"
MAX_STEPS = 100


def run_name(context, data, layout, seed):
    return f"{context}__{data}__{layout}__s{seed}"


def parse_name(name):
    context, data, layout, s = name.split("__")
    if context not in TEACHER_CONTEXTS or data not in DATASETS:
        raise SystemExit(f"bad run name {name}")
    return context, data, layout, int(s[1:])


def paths(name):
    return RUNS / name, EXP2_CKPT / name


def train_run(name, max_steps=MAX_STEPS):
    from ..copsd.train import train
    context, data, layout, seed = parse_name(name)
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    rows, desc = training_rows(data, context)
    trows, flags = teacher_rows(rows, context)
    write_jsonl(rd / "train_rows.jsonl", [{k: r.get(k) for k in ("dataset", "qid", "problem_swa", "problem_en", "solution",
                                                                 "vote_share")} for r in trows])
    log.info(f"{name}: {len(trows)} rows {desc}; collator flags {flags}")
    train(rd, ck, "swa", max_steps, include_problem_en=flags["include_problem_en"],
          include_reference_solution_en=flags["include_reference_solution_en"], seed=seed, layout=layout, rows=trows,
          data_desc=desc, reference_desc=f"{context}: {TEACHER_CONTEXTS[context]['note']}")


def copy_base(name):
    rd, _ = paths(name)
    (rd / "eval").mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        src = level_dir("L0") / "eval" / f"base_{split}.jsonl"
        if not (rd / "eval" / src.name).exists():
            shutil.copy(src, rd / "eval" / src.name)


def eval_run(name, split, max_steps=MAX_STEPS):
    """DEV: every checkpoint. TEST: the DEV-selected checkpoint only. Returns the selected name (or None)."""
    from ..copsd.config import EVAL, SAVE_EVERY
    from ..copsd.evaluate import evaluate, select_checkpoint
    rd, ck = paths(name)
    copy_base(name)
    steps = [f"step{s}" for s in range(SAVE_EVERY, max_steps + 1, SAVE_EVERY)]
    if split == "dev":
        evaluate(rd, steps, "dev", dict(EVAL), ck)
        return select_checkpoint(rd, SELECT_METRIC, steps)
    sel = select_checkpoint(rd, SELECT_METRIC, steps)
    if sel is None:
        raise SystemExit(f"{name}: DEV evaluations incomplete, cannot select")
    evaluate(rd, [sel], "test", dict(EVAL), ck)
    return sel


def report_run(name):
    from ..copsd.report import build_report
    return build_report(paths(name)[0], SELECT_METRIC)
