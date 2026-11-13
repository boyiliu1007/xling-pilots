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
from ..copsd.config import OFFICIAL
from .levels import EXP2_CKPT, EXP2_RESULTS, TEACHER_CONTEXTS, run_dir as level_dir, teacher_rows
from .trainsets import DATASETS, training_rows

log = logging.getLogger("votedistill.exp2.runs")
RUNS = EXP2_RESULTS / "runs"
SELECT_METRIC = "maj_correct"
MAX_STEPS = 100
OFFICIAL_CLIP = OFFICIAL["jsd_token_clip"]


CLIP_VARIANTS = {"noclip": None, "clip0.2": 0.2, "clip0.5": 0.5}   # loss clip ablations; absent = official 0.05


def run_name(context, data, layout, seed, clip_variant=None):
    return "__".join([context, data, layout] + ([clip_variant] if clip_variant else []) + [f"s{seed}"])


def parse_name(name):
    """<context>__<data>__<layout>[__<clip variant>]__s<seed> -> (context, data, layout, seed, clip)."""
    parts = name.split("__")
    if len(parts) == 4:
        (context, data, layout, s), clip = parts, OFFICIAL_CLIP
    elif len(parts) == 5 and parts[3] in CLIP_VARIANTS:
        context, data, layout, v, s = parts
        clip = CLIP_VARIANTS[v]
    else:
        raise SystemExit(f"bad run name {name}")
    if context not in TEACHER_CONTEXTS or data not in DATASETS:
        raise SystemExit(f"bad run name {name}")
    return context, data, layout, int(s[1:]), clip


def paths(name):
    return RUNS / name, EXP2_CKPT / name


def train_run(name, max_steps=MAX_STEPS):
    from ..copsd.train import train
    context, data, layout, seed, clip = parse_name(name)
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    rows, desc = training_rows(data, context)
    trows, flags = teacher_rows(rows, context)
    write_jsonl(rd / "train_rows.jsonl", [{k: r.get(k) for k in ("dataset", "qid", "problem_swa", "problem_en", "solution",
                                                                 "vote_share")} for r in trows])
    log.info(f"{name}: {len(trows)} rows {desc}; collator flags {flags}")
    train(rd, ck, "swa", max_steps, include_problem_en=flags["include_problem_en"],
          include_reference_solution_en=flags["include_reference_solution_en"], seed=seed, layout=layout, rows=trows,
          data_desc=desc, reference_desc=f"{context}: {TEACHER_CONTEXTS[context]['note']}", clip=clip)


def copy_base(name):
    rd, _ = paths(name)
    (rd / "eval").mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        src = level_dir("L0") / "eval" / f"base_{split}.jsonl"
        if not (rd / "eval" / src.name).exists():
            shutil.copy(src, rd / "eval" / src.name)


def eval_run(name, split, max_steps=MAX_STEPS):
    """DEV: every checkpoint. TEST: the DEV-selected checkpoint and the final one (selection-free comparison). Returns the selected name (or None)."""
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
    final = f"step{max_steps}"
    evaluate(rd, list(dict.fromkeys([sel, final])), "test", dict(EVAL), ck)   # DEV pick + the final checkpoint
    return sel


def report_run(name):
    from ..copsd.report import build_report
    return build_report(paths(name)[0], SELECT_METRIC)
