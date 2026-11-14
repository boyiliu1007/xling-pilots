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
# Two-stage runs, variant "<init>-<clip>": start from step N of a stage-1 run with the same data, layout and seed
# (adapter, optimizer, LR schedule and RNG state are carried over), then continue to step 100 with this run's teacher
# context and clip. N = 30: the first checkpoint after the boxed rate of the clipped runs' *training* rollouts stops
# rising (gold-free rule; PREREGISTRATION.md 2026-11-13 two-stage entry).
INITS = {"s1vote30": ("L3-vote", 30), "s1gold30": ("L3-gold", 30), "s1sft30": ("selfsft-vote", 30)}
SFT_CONTEXT = "selfsft-vote"   # stage-1 alternative: SFT on the model's own vote-consistent Swahili answers (selfsft.py)
SFT_STEPS = 30


STYLE_VARIANTS = {"enthink": "en-think-sw"}
# Teacher-free controls (context L0-same: the teacher prompt is the student's): "hot<τ>" = distil towards the base
# model's own distribution at temperature τ (entropy-raising, no clip); "ul" = unlikelihood on verbatim loops.
OBJECTIVE_VARIANTS = {"hot1.5": ("copsd", 1.5), "hot2.5": ("copsd", 2.5), "ul": ("unlikelihood", None)}


# Penalty-rollout prefix (PREREGISTRATION 2026-11-14 penalty-rollout entry): "rp" or "rp-<variant>" = the student's
# training rollouts are sampled with repetition penalty ROLLOUT_PEN (default decoding at evaluation unless stated).
ROLLOUT_PREFIX, ROLLOUT_PEN = "rp", 1.1


def _strip_rp(v):
    head, _, rest = v.partition("-")
    return (True, rest) if head == ROLLOUT_PREFIX else (False, v)


def rollout_pen_of(name):
    parts = name.split("__")
    return ROLLOUT_PEN if len(parts) == 5 and _strip_rp(parts[3])[0] else 1.0


def objective_of(name):
    """(objective, teacher_temp) for the trainer."""
    parts = name.split("__")
    if len(parts) == 5 and _strip_rp(parts[3])[1] in OBJECTIVE_VARIANTS:
        return OBJECTIVE_VARIANTS[_strip_rp(parts[3])[1]]
    return "copsd", None   # prompt style (copsd/styles.py) for student, teacher and evaluation


def style_of(name):
    parts = name.split("__")
    if len(parts) == 5:
        head = _strip_rp(parts[3])[1].split("-")[0]
        if head in STYLE_VARIANTS:
            return STYLE_VARIANTS[head]
    return "copsd"


def parse_variant(v):
    """-> (init (source context, step) or None, clip). A leading style token (e.g. "enthink", "enthink-noclip")
    selects the prompt style; it is stripped here and read by style_of. So is a leading "rp" (rollout_pen_of)."""
    rp, v = _strip_rp(v)
    if rp and not v:
        return None, OFFICIAL_CLIP
    if v in OBJECTIVE_VARIANTS:
        return None, None
    head, _, rest = v.partition("-")
    if head in STYLE_VARIANTS:
        return (None, OFFICIAL_CLIP) if not rest else parse_variant(rest)
    if v in CLIP_VARIANTS:
        return None, CLIP_VARIANTS[v]
    init, _, clip = v.partition("-")
    if init in INITS and clip in CLIP_VARIANTS:
        return INITS[init], CLIP_VARIANTS[clip]
    raise SystemExit(f"bad run variant {v}")


def run_name(context, data, layout, seed, clip_variant=None):
    return "__".join([context, data, layout] + ([clip_variant] if clip_variant else []) + [f"s{seed}"])


def parse_name(name):
    """<context>__<data>__<layout>[__<clip variant>]__s<seed> -> (context, data, layout, seed, clip)."""
    parts = name.split("__")
    if len(parts) == 4:
        (context, data, layout, s), clip = parts, OFFICIAL_CLIP
    elif len(parts) == 5:
        context, data, layout, v, s = parts
        clip = parse_variant(v)[1]
    else:
        raise SystemExit(f"bad run name {name}")
    if (context not in TEACHER_CONTEXTS and context != SFT_CONTEXT) or data not in DATASETS:
        raise SystemExit(f"bad run name {name}")
    return context, data, layout, int(s[1:]), clip


def init_of(name):
    """(stage-1 run name, step) for a two-stage run, else None."""
    parts = name.split("__")
    if len(parts) != 5:
        return None
    init = parse_variant(parts[3])[0]
    if init is None:
        return None
    return "__".join([init[0], parts[1], parts[2], parts[4]]), init[1]


def eval_steps(name, max_steps=MAX_STEPS):
    """Checkpoints this run evaluates: its own (two-stage: from the switch step on; self-SFT: 10..30)."""
    from ..copsd.config import SAVE_EVERY
    if name.split("__")[0] == SFT_CONTEXT:
        return [f"step{s}" for s in range(SAVE_EVERY, SFT_STEPS + 1, SAVE_EVERY)]
    init = init_of(name)
    first = init[1] if init else SAVE_EVERY
    return [f"step{s}" for s in range(first, max_steps + 1, SAVE_EVERY)]


def final_step(name, max_steps=MAX_STEPS):
    return SFT_STEPS if name.split("__")[0] == SFT_CONTEXT else max_steps


def _warm_start(name, rd, ck, desc):
    """Copy the stage-1 checkpoint (and its loss rows and DEV evaluation) into this run before training resumes."""
    from ..copsd.train import latest_ckpt
    src, k = init_of(name)
    s_rd, s_ck = paths(src)
    d = s_ck / f"step-{k}"
    if not (d / "COMPLETE").exists():
        raise SystemExit(f"{name}: stage-1 checkpoint {d} missing")
    if latest_ckpt(ck) is None:
        ck.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, ck / f"step-{k}")
        lines = (s_rd / "train_loss.csv").read_text().splitlines()
        (rd / "train_loss.csv").write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= k]) + "\n")
        (rd / "eval").mkdir(parents=True, exist_ok=True)
        for split in ("dev", "test"):
            f = s_rd / "eval" / f"step{k}_{split}.jsonl"
            if f.exists():
                shutil.copy(f, rd / "eval" / f.name)
        log.info(f"{name}: warm start from {d}")
    return {**desc, "init_from": f"{src} step {k} (adapter, optimizer, LR schedule, RNG); stage-1 loss and DEV/TEST "
                                 f"evaluation of step {k} copied"}


def paths(name):
    return RUNS / name, EXP2_CKPT / name


def train_run(name, max_steps=MAX_STEPS):
    from ..copsd.train import train
    context, data, layout, seed, clip = parse_name(name)
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    if context == SFT_CONTEXT:
        from .selfsft import train_sft
        return train_sft(name, data, layout, seed, rollout_pen_of(name))
    from ..copsd.styles import apply_style
    apply_style(style_of(name))
    rows, desc = training_rows(data, context)
    desc = {**desc, "prompt_style": style_of(name)}
    objective, teacher_temp = objective_of(name)
    if init_of(name):
        desc = _warm_start(name, rd, ck, desc)
    trows, flags = teacher_rows(rows, context)
    write_jsonl(rd / "train_rows.jsonl", [{k: r.get(k) for k in ("dataset", "qid", "problem_swa", "problem_en", "solution",
                                                                 "vote_share")} for r in trows])
    log.info(f"{name}: {len(trows)} rows {desc}; collator flags {flags}")
    train(rd, ck, "swa", max_steps, include_problem_en=flags["include_problem_en"],
          include_reference_solution_en=flags["include_reference_solution_en"], seed=seed, layout=layout, rows=trows,
          data_desc=desc, reference_desc=f"{context}: {TEACHER_CONTEXTS[context]['note']}", clip=clip,
          objective=objective, teacher_temp=teacher_temp, rollout_rep_pen=rollout_pen_of(name))


STYLE_BASE_LEVEL = {"copsd": "L0", "en-think-sw": "L0-enthink-sw"}


def copy_base(name):
    """The base model's evaluation under the run's prompt style (if it exists yet), for the run's own report."""
    rd, _ = paths(name)
    (rd / "eval").mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        src = level_dir(STYLE_BASE_LEVEL[style_of(name)]) / "eval" / f"base_{split}.jsonl"
        if not src.exists():
            continue
        if not (rd / "eval" / src.name).exists():
            shutil.copy(src, rd / "eval" / src.name)


def eval_run(name, split, max_steps=MAX_STEPS):
    """DEV: every checkpoint. TEST: the DEV-selected checkpoint and the final one (selection-free comparison). Returns the selected name (or None)."""
    from ..copsd.config import EVAL, SAVE_EVERY
    from ..copsd.evaluate import evaluate, select_checkpoint
    rd, ck = paths(name)
    style = style_of(name)
    copy_base(name)
    max_steps = final_step(name, max_steps)
    steps = eval_steps(name, max_steps)
    cfg = dict(EVAL, **({"prompt_style": style} if style != "copsd" else {}))
    if split == "dev":
        evaluate(rd, steps, "dev", cfg, ck)
        return select_checkpoint(rd, SELECT_METRIC, steps)
    sel = select_checkpoint(rd, SELECT_METRIC, steps)
    if sel is None:
        raise SystemExit(f"{name}: DEV evaluations incomplete, cannot select")
    final = f"step{max_steps}"
    evaluate(rd, list(dict.fromkeys([sel, final])), "test", cfg, ck)   # DEV pick + the final checkpoint
    return sel


DECODE = EXP2_RESULTS / "decode"


def decode_dir(name, level, step=None):
    return DECODE / f"{name}__{level}"      # eval/step<N>_test.jsonl inside


def decode_eval(name, step, level):
    """TEST of run `name` at checkpoint `step`, sampled with base-model level `level`'s decoding overrides (prompt
    style of the run). Separate directory, so the run's own (default-decoding) evaluations are never touched."""
    import logging, os, platform
    from ..copsd.config import EVAL
    from ..copsd.evaluate import evaluate
    from ..utils import setup_logging
    from .levels import LEVELS
    if "sampling" not in LEVELS[level]:
        raise SystemExit(f"{level} has no decoding overrides")
    _, ck = paths(name)
    out = decode_dir(name, level, step)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "logs" / "eval.log")
    logging.getLogger("votedistill.exp2").info(f"decode-eval {name} step{step} {level} host {platform.node()} "
                                               f"job {os.environ.get('SLURM_JOB_ID')}")
    style = style_of(name)
    cfg = dict(EVAL, **({"prompt_style": style} if style != "copsd" else {}), **LEVELS[level]["sampling"])
    evaluate(out, [f"step{step}"], "test", cfg, ck)
    return out


def eval_final(name, decode_level="L0-reppen1.1"):
    """Selection-free evaluation (Qwen3-4B runs and the penalty-rollout runs; PREREGISTRATION 2026-11-14): TEST of the
    final checkpoint only, with default decoding and with `decode_level`'s decoding. No DEV sweep."""
    # Each evaluation in its own process: a vLLM engine is not released within a process, so a second one would wait
    # for the GPU forever (copsd/gpu.wait_for_gpu sees the first engine as another process).
    import subprocess, sys
    copy_base(name)
    step = final_step(name)
    exp2 = [sys.executable, "-m", "votedistill.exp2"]
    subprocess.run(exp2 + ["eval-run-final", "--name", name], check=True)
    subprocess.run(exp2 + ["decode-eval", "--name", name, "--step", str(step), "--level", decode_level], check=True)
    return decode_dir(name, decode_level)


def eval_run_final(name):
    """TEST of the final checkpoint with default decoding (one vLLM engine; called by eval_final in a subprocess)."""
    from ..copsd.config import EVAL
    from ..copsd.evaluate import evaluate
    rd, ck = paths(name)
    style = style_of(name)
    evaluate(rd, [f"step{final_step(name)}"], "test", dict(EVAL, **({"prompt_style": style} if style != "copsd" else {})), ck)


def report_run(name):
    from ..copsd.report import build_report
    return build_report(paths(name)[0], SELECT_METRIC)
