"""How well does Qwen3-1.7B itself translate between English and Swahili? (Prerequisite for an on-policy pipeline
in which the model translates its own English reasoning into Swahili, with no other model after voting.)

Tasks (greedy, non-thinking chat mode, one instruction prompt; NLLB-200 3.3B as a yardstick on task A only):
  A  AfriMGSM questions (all 250, human-parallel), EN->SW and SW->EN: chrF against the human reference.
  B  the model's own English majority-vote solutions (gold-free references, trainsets.voted_references), EN->SW:
     no reference, so: numbers preserved, \\boxed answer preserved, loops, output language, length ratio, and
     round trip (the model translates its Swahili back to English; chrF against the original English).
chrF++: sacreBLEU's CHRF(word_order=2), corpus level (character 6-grams + word bigrams, beta = 2).
Outputs results/exp2/translation/*.jsonl and report.txt.
"""
import logging
import re
from collections import Counter

import numpy as np

from ..data import _question_numbers, load_afrimgsm
from ..paths import MODEL
from ..utils import read_jsonl, write_jsonl
from .levels import EXP2_RESULTS
from .report import is_repetitive

log = logging.getLogger("votedistill.exp2.translation")
OUT = EXP2_RESULTS / "translation"
LANG = {"en": "English", "sw": "Swahili"}
N_SOLUTIONS = {"gsm8k-mt": 150, "copsd": 100}
MAX_TOKENS = 1536


def chrf(hyps, refs):
    """Corpus-level chrF++ (sacreBLEU)."""
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score([h or "" for h in hyps], [[r or "" for r in refs]]).score


def prompt(tok, text, src, tgt):
    msg = (f"Translate the following {LANG[src]} text into {LANG[tgt]}. Keep all numbers, formulas and LaTeX "
           f"exactly as they are. Output only the {LANG[tgt]} translation, nothing else.\n\n{text}")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def _qwen(llm, tok, texts, src, tgt):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outs = llm.generate([prompt(tok, t, src, tgt) for t in texts], sp, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outs], [o.outputs[0].finish_reason for o in outs]


def run(nllb=True):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from .trainsets import voted_references
    OUT.mkdir(parents=True, exist_ok=True)
    en = [r["question"] for r in load_afrimgsm("en")]
    sw = [r["question"] for r in load_afrimgsm("swa")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.45, max_model_len=6144, seed=42)
    # A: questions
    for src, tgt, xs, refs in (("en", "sw", en, sw), ("sw", "en", sw, en)):
        p = OUT / f"A_questions_{src}2{tgt}_qwen.jsonl"
        if not p.exists():
            hyp, fin = _qwen(llm, tok, xs, src, tgt)
            write_jsonl(p, [{"item": i, "src": x, "ref": r, "hyp": h, "finish": f}
                            for i, (x, r, h, f) in enumerate(zip(xs, refs, hyp, fin))])
    # B: own voted English solutions -> Swahili -> back to English
    for data, n in N_SOLUTIONS.items():
        p = OUT / f"B_solutions_{data}_qwen.jsonl"
        if p.exists():
            continue
        refs = voted_references(data)
        qids = sorted(refs)[:n]
        src = [refs[q][0] for q in qids]
        sw_h, sw_f = _qwen(llm, tok, src, "en", "sw")
        back, back_f = _qwen(llm, tok, sw_h, "sw", "en")
        write_jsonl(p, [{"qid": q, "answer": refs[q][1], "src": s, "hyp": h, "finish": f, "back": b, "back_finish": bf}
                        for q, s, h, f, b, bf in zip(qids, src, sw_h, sw_f, back, back_f)])
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if nllb:
        _nllb(en, sw)
    return report()


def _nllb(en, sw):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from .translate import GEN, MT_MODEL, split_sentences
    codes = {"en": "eng_Latn", "sw": "swh_Latn"}
    for src, tgt, xs, refs in (("en", "sw", en, sw), ("sw", "en", sw, en)):
        p = OUT / f"A_questions_{src}2{tgt}_nllb.jsonl"
        if p.exists():
            continue
        tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang=codes[src])
        model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, dtype=torch.bfloat16).cuda().eval()
        bos = tok.convert_tokens_to_ids(codes[tgt])
        sents = [split_sentences(x) for x in xs]
        flat = [s for ss in sents for s in ss]
        outs = []
        for i in range(0, len(flat), 32):
            enc = tok(flat[i:i + 32], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, forced_bos_token_id=bos, **GEN)
            outs += tok.batch_decode(g, skip_special_tokens=True)
        hyp, k = [], 0
        for ss in sents:
            hyp.append(" ".join(o.strip() for o in outs[k:k + len(ss)]))
            k += len(ss)
        write_jsonl(p, [{"item": i, "src": x, "ref": r, "hyp": h} for i, (x, r, h) in enumerate(zip(xs, refs, hyp))])
        del model


def _nums_ok(a, b):
    return _question_numbers(a) == _question_numbers(b)


def report():
    from ..copsd.evaluate import language_of
    T = ["Translation ability of Qwen3-1.7B (greedy, non-thinking); NLLB-200 3.3B as a yardstick", "",
         "## A. AfriMGSM questions (250, human references)",
         f"  {'direction':<10} {'system':<6} {'chrF++':>6} {'numbers kept':>13} {'loops':>6} {'target lang':>12} {'len ratio':>10}"]
    for d in ("en2sw", "sw2en"):
        tgt = "sw" if d.endswith("sw") else "en"
        for sysname in ("qwen", "nllb"):
            p = OUT / f"A_questions_{d}_{sysname}.jsonl"
            if not p.exists():
                continue
            R = read_jsonl(p)
            T.append(f"  {d:<10} {sysname:<6} {chrf([r['hyp'] for r in R], [r['ref'] for r in R]):6.1f} "
                     f"{np.mean([_nums_ok(r['src'], r['hyp']) for r in R]):13.3f} "
                     f"{np.mean([is_repetitive(r['hyp']) for r in R]):6.3f} "
                     f"{np.mean([language_of(r['hyp']) == tgt for r in R]):12.3f} "
                     f"{np.median([len(r['hyp']) / max(len(r['src']), 1) for r in R]):10.2f}")
    T += ["", "## B. the model's own English voted solutions -> Swahili (no reference) -> back to English",
          f"  {'data':<9} {'n':>4} {'numbers kept':>13} {'boxed kept':>11} {'loops':>6} {'cut off':>8} {'Swahili':>8} "
          f"{'len ratio':>10} {'round-trip chrF':>16}"]
    for data in N_SOLUTIONS:
        p = OUT / f"B_solutions_{data}_qwen.jsonl"
        if not p.exists():
            continue
        R = read_jsonl(p)
        boxed = [("\\boxed{" + r["answer"]) in r["hyp"].replace(" ", "") or r["answer"] in r["hyp"] for r in R]
        T.append(f"  {data:<9} {len(R):4d} {np.mean([_nums_ok(r['src'], r['hyp']) for r in R]):13.3f} "
                 f"{np.mean(boxed):11.3f} {np.mean([is_repetitive(r['hyp']) for r in R]):6.3f} "
                 f"{np.mean([r['finish'] == 'length' for r in R]):8.3f} "
                 f"{np.mean([language_of(r['hyp']) == 'sw' for r in R]):8.3f} "
                 f"{np.median([len(r['hyp']) / max(len(r['src']), 1) for r in R]):10.2f} "
                 f"{chrf([r['back'] for r in R], [r['src'] for r in R]):16.1f}")
    text = "\n".join(T) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text(text)
    return text
