"""Can the teacher itself answer in Swahili? Let the base model *generate* from each teacher context's prompt.

COPSD's teacher never generates during training: it only scores the student's text. This measures what its own
next-token distribution would produce if sampled, on the training problems (COPSD set, gold `Answer` known):
accuracy (math-verify, as the COPSD grader), finished (no length cut), closed thinking, verbatim loops, language,
and length. Sampling = the training sampling (T 1.1, top-p 0.95, top-k 20, 2048 tokens), since that is the
distribution the student is distilled towards; a second budget of 4096 tokens at T 1.0 matches the evaluation.
Outputs results/exp2/teachergen/<context>_<budget>.jsonl and report.txt.
"""
import logging
import random
from collections import Counter

import numpy as np

from ..answers_math import equiv, extract_pred, parse_answer
from ..copsd.collator import make_collator, prepare_rows
from ..copsd.config import OFFICIAL
from ..copsd.evaluate import language_of
from ..data import load_copsd_train
from ..paths import MODEL
from ..utils import read_jsonl, write_jsonl
from .levels import EXP2_RESULTS, teacher_rows
from .report import is_repetitive
from .screen import _prompt_ids

log = logging.getLogger("votedistill.exp2.teachergen")
OUT = EXP2_RESULTS / "teachergen"
CONTEXTS = ("L0-same", "L1", "L2-gold", "L3-gold", "L3-shuffled-gold", "L3-vote")
# Vote contexts use only the sampled problems whose English vote was accepted (trainsets.voted_references).
VOTE_CONTEXTS = ("L3-vote",)
BUDGETS = {"train": dict(temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"],
                         max_tokens=OFFICIAL["max_completion_length"]),
           "eval": dict(temperature=1.0, top_p=0.95, top_k=-1, max_tokens=4096)}
# Same budgets with the repetition penalty that removes most loops in the student (L0-reppen1.1): does the teacher,
# once it finishes, actually answer correctly?
BUDGETS.update({f"{b}-rp": {**BUDGETS[b], "repetition_penalty": 1.1} for b in ("train", "eval")})
N_PROBLEMS, N_SAMPLES, SEED = 100, 4, 42


def run(contexts=CONTEXTS, budgets=("train",)):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = prepare_rows(load_copsd_train("swa"), "swa")
    idx = sorted(random.Random(SEED).sample(range(len(rows)), N_PROBLEMS))
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, max_model_len=12288, seed=SEED)
    for c in contexts:
        ix, crows = idx, rows
        if c in VOTE_CONTEXTS:
            from .trainsets import voted_references
            refs = voted_references("copsd")
            crows = [{**r, "vote_solution": refs[i][0] if i in refs else None,
                      "vote_answer": refs[i][1] if i in refs else None} for i, r in enumerate(rows)]
            ix = [i for i in idx if i in refs]
        trows, flags = teacher_rows(crows, c)
        col = make_collator(tok, OFFICIAL["max_length"], **flags)
        key = "student" if c == "L0-same" else "teacher"
        prompts = [_prompt_ids(col, trows[i], key) for i in ix]
        for b in budgets:
            p = OUT / f"{c}_{b}.jsonl"
            if p.exists():
                continue
            sps = [SamplingParams(n=N_SAMPLES, min_p=0.0, seed=SEED * 1_000_003 + i, **BUDGETS[b]) for i in ix]
            outs = llm.generate([{"prompt_token_ids": q} for q in prompts], sps, use_tqdm=False)
            recs = []
            for i, o in zip(ix, outs):
                gold = str(rows[i]["Answer"])
                for j, x in enumerate(o.outputs):
                    s, pp = extract_pred(x.text, thinking=True)
                    lang = language_of(x.text)
                    recs.append({"problem": i, "sample": j, "gold": gold, "pred": s,
                                 "correct": bool(s is not None and equiv(gold, parse_answer(gold), s, pp)),
                                 "finished": x.finish_reason == "stop", "closed_think": "</think>" in x.text,
                                 "loop": is_repetitive(x.text), "lang": lang, "chars": len(x.text), "text": x.text})
            write_jsonl(p, recs)
            log.info(f"{c} {b}: acc {np.mean([r['correct'] for r in recs]):.3f}")
    return report()


def report():
    T = [f"Teacher generation: base model sampled from each context's prompt; {N_PROBLEMS} COPSD training problems x "
         f"{N_SAMPLES} samples; correct = math-verify vs gold Answer", ""]
    for b, cfg in BUDGETS.items():
        T.append(f"## budget {b}: {cfg}")
        T.append(f"  {'context':<18} {'acc':>6} {'pass@4':>7} {'finished':>9} {'closed':>7} {'loop':>6} {'acc|fin':>8} "
                 f"{'sw':>5} {'en':>5} {'median chars':>13}")
        for c in CONTEXTS:
            p = OUT / f"{c}_{b}.jsonl"
            if not p.exists():
                continue
            R = read_jsonl(p)
            by = {}
            for r in R:
                by.setdefault(r["problem"], []).append(r["correct"])
            fin = [r["correct"] for r in R if r["finished"]]
            lang = Counter(r["lang"] for r in R)
            T.append(f"  {c:<18} {np.mean([r['correct'] for r in R]):6.3f} {np.mean([any(v) for v in by.values()]):7.3f} "
                     f"{np.mean([r['finished'] for r in R]):9.3f} {np.mean([r['closed_think'] for r in R]):7.3f} "
                     f"{np.mean([r['loop'] for r in R]):6.3f} {np.mean(fin) if fin else float('nan'):8.3f} "
                     f"{lang['sw'] / len(R):5.2f} {lang['en'] / len(R):5.2f} {np.median([r['chars'] for r in R]):13.0f}")
        T.append("")
    text = "\n".join(T) + "\n"
    (OUT / "report.txt").write_text(text)
    return text
