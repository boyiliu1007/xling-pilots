"""Reading-comprehension OPSD: distil the *reading* of a Swahili question, not the solution.

Motivation (Experiment 2 data, Qwen3-1.7B, AfriMGSM TEST):
  - English question, English thinking: mean_acc 0.886. Swahili question, English thinking: 0.072, with 87% of
    answers boxed. The model reasons fine in English; it misreads the Swahili question.
  - clipstats (copsd_k2): 45-59% of the teacher-student KL lies in the first 10% of each rollout (where the model
    restates the problem), and that is also where the official 0.05 clip binds most (31% of positions).
So: student = Swahili question, thinks and answers in English (style en-think-en); teacher = the same frozen model
also given the English question (context L1, no solution: a gold-free, label-free signal about what the question
says). The loss may be restricted to the first `window` completion tokens, where the problem is restated.

Run name: rc__<data>__<window>__<clip>__s<seed>
  data    gsm8k-mt | copsd          (Experiment 2 training sets; the Swahili question is the student's input)
  window  first256 | all            (loss on the first 256 completion tokens, or on the whole completion)
  clip    clip | noclip             (official per-(token, vocab) 0.05 clip, or none)
"""
import logging

from ..copsd.config import OFFICIAL
from ..utils import write_jsonl
from .uopsd import CKPT, RUNS

log = logging.getLogger("votedistill.xfer.rc")
WINDOWS = {"first256": 256, "all": None}
CLIPS = {"clip": OFFICIAL["jsd_token_clip"], "noclip": None}
STYLE = "en-think-en"
CONTEXT = "L1"


def parse(name):
    p = name.split("__")
    if len(p) != 5 or p[0] != "rc" or p[1] not in ("gsm8k-mt", "copsd") or p[2] not in WINDOWS or p[3] not in CLIPS \
            or not p[4].startswith("s"):
        raise SystemExit(f"bad rc run name {name}: rc__<gsm8k-mt|copsd>__<first256|all>__<clip|noclip>__s<seed>")
    return p[1], WINDOWS[p[2]], CLIPS[p[3]], int(p[4][1:])


def train_rc(name, max_steps=100):
    from ..copsd.styles import apply_style
    from ..copsd.train import train
    from ..exp2.levels import TEACHER_CONTEXTS, teacher_rows
    from ..exp2.trainsets import training_rows
    data, window, clip, seed = parse(name)
    rd, ck = RUNS / name, CKPT / name
    rd.mkdir(parents=True, exist_ok=True)
    apply_style(STYLE)
    rows, desc = training_rows(data, CONTEXT)
    trows, flags = teacher_rows(rows, CONTEXT)
    desc = {**desc, "prompt_style": STYLE, "teacher_context": CONTEXT, "loss_token_limit": window}
    write_jsonl(rd / "train_rows.jsonl", [{k: r.get(k) for k in ("dataset", "qid", "problem_swa", "problem_en")}
                                          for r in trows])
    log.info(f"{name}: {len(trows)} rows {desc}; collator flags {flags}")
    train(rd, ck, "swa", max_steps, include_problem_en=flags["include_problem_en"],
          include_reference_solution_en=flags["include_reference_solution_en"], seed=seed, layout="exact", rows=trows,
          data_desc=desc, reference_desc=f"{CONTEXT}: {TEACHER_CONTEXTS[CONTEXT]['note']}", clip=clip,
          token_limit=window)
