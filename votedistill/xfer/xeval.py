"""Evaluate a LoRA checkpoint in three views of the same AfriMGSM TEST items, and compare each with the base model.

  swsw  Swahili question, COPSD's Swahili prompt and thinking              base results/exp2/L0
  swen  Swahili question, think and answer in English (styles en-think-en)  base results/exp2/L0-enthink-en
  enen  English AfriMGSM question, COPSD's English prompt                   base results/exp2/L0-eng
  swsw-rp  swsw with repetition penalty 1.1 (the decoding baseline)         base results/exp2/L0-reppen1.1

Sampling is exactly the base level's (COPSD protocol: 12 samples, T 1.0, top-p 0.95, 4096 new tokens), so every
step file is paired item-by-item with that level's base_test.jsonl.
Outputs results/xfer/xeval/<label>__<view>/eval/step<N>_test.jsonl.
"""
from pathlib import Path

from ..copsd.config import EVAL
from ..paths import RESULTS
from .uopsd import XFER

XEVAL = XFER / "xeval"
VIEWS = {
    "swsw": dict(lang="swa", base="L0", cfg={}),
    "swen": dict(lang="swa", base="L0-enthink-en", cfg={"prompt_style": "en-think-en"}),
    "enen": dict(lang="eng", base="L0-eng", cfg={}),
    "swsw-rp": dict(lang="swa", base="L0-reppen1.1", cfg={"repetition_penalty": 1.1}),
}


def out_dir(label, view):
    return XEVAL / f"{label}__{view}"


def base_file(view):
    return RESULTS / "exp2" / VIEWS[view]["base"] / "eval" / "base_test.jsonl"


def xeval(label, ckpt_dir, step, view):
    from ..copsd.evaluate import evaluate
    v = VIEWS[view]
    out = out_dir(label, view)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint.txt").write_text(f"{Path(ckpt_dir) / f'step-{step}' / 'adapter'}\n")
    evaluate(out, [f"step{step}"], "test", dict(EVAL, **v["cfg"]), ckpt_dir, lang=v["lang"])
    return out
