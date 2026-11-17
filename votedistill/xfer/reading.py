"""Explicit vs implicit reading: can the model *translate* the Swahili question, and does solving its own translation
beat solving the Swahili question directly? Base models only (no training), Qwen3-1.7B / 4B / 8B.

Every condition solves AfriMGSM TEST (200 items) with the same prompt (style en-think-en: think and answer in
English) and COPSD's evaluation protocol (12 samples, T 1.0, top-p 0.95, 4096 new tokens, thinking mode, COPSD
grader). Only the question text differs:

  en    human English question                              ceiling: the model's reasoning with perfect reading
  sw    Swahili question                                    implicit reading (understand without writing it down)
  self  the model's own English translation (greedy,        explicit reading: translate, then answer
        non-thinking, results/exp2/translation prompt)
  nllb  NLLB-200 3.3B translation                           an external reader; also the translate-test baseline
  x8b   Qwen3-8B's translation (1.7B and 4B only)           a better reader of the same family, same solver

Gap decomposition per model:  en - sw  =  (en - self) [lost in explicit translation] + (self - sw) [explicit vs
implicit]. self > sw: the model reads better when it writes the translation out. self ~ sw with a poor translation:
the reading knowledge is not there to be elicited.

Per-item translation quality vs the human English question: sentence chrF++ and "numbers kept" (same multiset of
numbers), so solving can be split by faithful vs unfaithful translation.

  python -m votedistill.xfer reading --model Qwen/Qwen3-4B          # translate + solve (idempotent)
  python -m votedistill.xfer reading-report
Outputs results/xfer/reading/<model>/{trans_self.jsonl, solve_<cond>.jsonl}, results/xfer/reading/report.txt.
"""
import logging
import time
from collections import Counter

import numpy as np

from ..copsd.config import EVAL, SEED
from ..data import _question_numbers, load_afrimgsm, load_split
from ..utils import read_jsonl, write_json, write_jsonl
from ..paths import RESULTS

log = logging.getLogger("votedistill.xfer.reading")
OUT = RESULTS / "xfer" / "reading"   # all models in one tree (base models only)
MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
READER = "Qwen/Qwen3-8B"
CONDS = ["en", "sw", "self", "nllb", "x8b"]
NLLB_FILE = RESULTS / "exp2" / "translation" / "A_questions_sw2en_nllb.jsonl"
STYLE = "en-think-en"


def slug(model):
    return model.split("/")[-1].lower()


def mdir(model):
    return OUT / slug(model)


def _trans_prompt(tok, text):
    """Instruction first, the text delimited, and the answer pre-started. The exp2/translation prompt (instruction
    after the text) made Qwen3-1.7B echo the instruction on 20/200 TEST items and copy the Swahili on 25/200, which
    would understate explicit reading."""
    msg = ("Translate the following Swahili math word problem into English. Keep every number exactly as written. "
           "Reply with the English translation only.\n\nSwahili:\n" + text)
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "English:\n"


def translate_self(llm, tok, model):
    p = mdir(model) / "trans_self.jsonl"
    if p.exists():
        return read_jsonl(p)
    from vllm import SamplingParams
    sw = [r["question"] for r in load_afrimgsm("swa")]
    en = [r["question"] for r in load_afrimgsm("en")]
    outs = llm.generate([_trans_prompt(tok, x) for x in sw], SamplingParams(temperature=0.0, max_tokens=1536),
                        use_tqdm=False)
    recs = [{"item": i, "src": s, "ref": r, "hyp": o.outputs[0].text.strip(), "finish": o.outputs[0].finish_reason}
            for i, (s, r, o) in enumerate(zip(sw, en, outs))]
    write_jsonl(p, recs)
    return recs


def questions(model, cond):
    """{item: English-or-Swahili question text} for the condition, or None if its translation is not available."""
    if cond == "en":
        return {i: r["question"] for i, r in enumerate(load_afrimgsm("en"))}
    if cond == "sw":
        return {i: r["question"] for i, r in enumerate(load_afrimgsm("swa"))}
    f = {"self": mdir(model) / "trans_self.jsonl", "nllb": NLLB_FILE, "x8b": mdir(READER) / "trans_self.jsonl"}[cond]
    return {r["item"]: r["hyp"] for r in read_jsonl(f)} if f.exists() else None


def run(model):
    from ..copsd.collator import import_copsd_eval
    from ..copsd.styles import apply_style
    from ..gpu import wait_for_gpu
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    d = mdir(model)
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(EVAL)
    write_json(d / "config.json", {"model": model, "sampling": cfg, "style": STYLE, "conditions": __doc__})
    EM = import_copsd_eval()
    apply_style(STYLE)
    wait_for_gpu()
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=cfg["max_new_tokens"] + 2048,
              seed=SEED)
    translate_self(llm, tok, model)
    rows = load_afrimgsm("swa")
    ids = load_split()["test"]
    sp = SamplingParams(n=cfg["val_n"], temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                        max_tokens=cfg["max_new_tokens"], seed=SEED)
    for cond in CONDS:
        p = d / f"solve_{cond}.jsonl"
        if p.exists() or (cond == "x8b" and model == READER):
            continue
        q = questions(model, cond)
        if q is None:
            log.warning(f"{cond}: translation not available yet (run {READER} first); skipped")
            continue
        prompts = [EM.build_qwen3_prompt(tok, q[i], "SWA", enable_thinking=cfg["enable_thinking"]) for i in ids]
        t0 = time.time()
        res = llm.generate(prompts, sp, use_tqdm=False)
        recs = []
        for i, o in zip(ids, res):
            gt = EM.get_afrimgsm_ground_truth(rows[i])
            texts = [c.text for c in o.outputs]
            preds = [EM.extract_boxed_answer(t) for t in texts]
            corr = [EM.grade_answer(x, gt) for x in preds]
            fp = [x for x in preds if x is not None]
            maj = EM.grade_answer(Counter(fp).most_common(1)[0][0], gt) if fp else False
            recs.append({"item": i, "gold": gt, "question": q[i], "preds": preds, "correct": corr,
                         "truncated": [c.finish_reason == "length" for c in o.outputs],
                         "mean_acc": sum(corr) / len(corr), "maj_correct": maj, "pass_at_n": any(corr),
                         "outputs": texts})
        write_jsonl(p, recs)
        log.info(f"{slug(model)} {cond}: mean_acc {np.mean([r['mean_acc'] for r in recs]):.3f} "
                 f"maj {np.mean([r['maj_correct'] for r in recs]):.3f} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------- report
def _boot(diff, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    m = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return 100 * diff.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def _fmt(t):
    return f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"


def _load(model, cond):
    p = mdir(model) / f"solve_{cond}.jsonl"
    return {r["item"]: r for r in read_jsonl(p)} if p.exists() else None


def trans_quality(recs, ids):
    from sacrebleu.metrics import CHRF
    ch = CHRF(word_order=2)
    by = {r["item"]: r for r in recs}
    out = {}
    for i in ids:
        r = by[i]
        out[i] = {"chrf": ch.sentence_score(r["hyp"] or "", [r["ref"]]).score,
                  "nums": _question_numbers(r["hyp"]) == _question_numbers(r["ref"]),
                  "loop": r.get("finish") == "length"}
        out[i]["fail"] = ("echo" if "Reply with the English" in (r["hyp"] or "") or "Keep all numbers" in (r["hyp"] or "")
                          else "copy" if ch.sentence_score(r["hyp"] or "", [r["src"]]).score > out[i]["chrf"] else "")
    corpus = ch.corpus_score([by[i]["hyp"] or "" for i in ids], [[by[i]["ref"] for i in ids]]).score
    return out, corpus


def report():
    ids = load_split()["test"]
    T = ["Explicit vs implicit reading (base models, AfriMGSM TEST 200, style en-think-en, COPSD protocol: 12 samples,",
         "T 1.0, 4096 tokens). Δ in points, paired bootstrap over items, 95% CI. See votedistill/xfer/reading.py.", ""]
    # translation quality
    T.append("## A. Translation of the Swahili questions into English (TEST 200, vs the human English question)")
    T.append(f"  {'reader':<12} {'chrF++':>7} {'numbers kept':>13} {'cut off':>8} {'echoes prompt':>14} {'copies Swahili':>15}")
    TQ = {}
    for m in MODELS + ["nllb"]:
        f = NLLB_FILE if m == "nllb" else mdir(m) / "trans_self.jsonl"
        if not f.exists():
            continue
        q, c = trans_quality(read_jsonl(f), ids)
        TQ[m] = q
        T.append(f"  {slug(m):<12} {c:7.1f} {np.mean([v['nums'] for v in q.values()]):13.3f} "
                 f"{np.mean([v['loop'] for v in q.values()]):8.3f} {np.mean([v['fail'] == 'echo' for v in q.values()]):14.3f} "
                 f"{np.mean([v['fail'] == 'copy' for v in q.values()]):15.3f}")
    # accuracy table
    T += ["", "## B. Solve accuracy by question text (mean_acc / maj@12 / pass@12 / truncated)"]
    T.append(f"  {'model':<12} " + " ".join(f"{c:>26}" for c in CONDS))
    R = {m: {c: _load(m, c) for c in CONDS} for m in MODELS}
    for m in MODELS:
        cells = []
        for c in CONDS:
            d = R[m][c]
            if d is None:
                cells.append(f"{'-':>26}")
                continue
            v = [d[i] for i in ids]
            cells.append(f"{np.mean([r['mean_acc'] for r in v]):.3f}/{np.mean([r['maj_correct'] for r in v]):.3f}/"
                         f"{np.mean([r['pass_at_n'] for r in v]):.3f}/{np.mean([np.mean(r['truncated']) for r in v]):.2f}"
                         .rjust(26))
        T.append(f"  {slug(m):<12} " + " ".join(cells))
    # decomposition
    T += ["", "## C. Gap decomposition, mean_acc (paired over items)",
          "  en - sw = (en - self) lost in the model's own translation + (self - sw) explicit vs implicit reading"]
    pairs = [("en", "sw", "total reading gap"), ("en", "self", "lost in own translation"),
             ("self", "sw", "EXPLICIT - IMPLICIT"), ("nllb", "sw", "external MT - implicit"),
             ("x8b", "self", "8B's translation - own"), ("en", "nllb", "lost in NLLB translation")]
    for m in MODELS:
        T.append(f"  {slug(m)}")
        for a, b, lab in pairs:
            A, B = R[m][a], R[m][b]
            if A is None or B is None:
                continue
            T.append(f"    {a:>4} - {b:<4} {lab:<26} {_fmt(_boot([A[i]['mean_acc'] - B[i]['mean_acc'] for i in ids]))}"
                     f"   maj {_fmt(_boot([float(A[i]['maj_correct']) - float(B[i]['maj_correct']) for i in ids]))}")
    # item-level: faithful vs unfaithful own translation
    T += ["", "## D. Items split by the model's own translation (faithful = same numbers as the human question and",
          "   sentence chrF++ >= 50). mean_acc per condition; 'sw wins' = items where sw maj is right but self maj wrong."]
    for m in MODELS:
        if m not in TQ or R[m]["self"] is None or R[m]["sw"] is None or R[m]["en"] is None:
            continue
        q = TQ[m]
        groups = {"faithful": [i for i in ids if q[i]["nums"] and q[i]["chrf"] >= 50],
                  "unfaithful": [i for i in ids if not (q[i]["nums"] and q[i]["chrf"] >= 50)]}
        T.append(f"  {slug(m)}")
        for g, its in groups.items():
            if not its:
                continue
            acc = {c: np.mean([R[m][c][i]["mean_acc"] for i in its]) for c in ("en", "sw", "self")}
            sw_wins = sum(R[m]["sw"][i]["maj_correct"] and not R[m]["self"][i]["maj_correct"] for i in its)
            self_wins = sum(R[m]["self"][i]["maj_correct"] and not R[m]["sw"][i]["maj_correct"] for i in its)
            T.append(f"    {g:<11} n={len(its):3d}  en {acc['en']:.3f}  self {acc['self']:.3f}  sw {acc['sw']:.3f}   "
                     f"self wins {self_wins:3d}  sw wins {sw_wins:3d}")
        # chrF bins
        bins = [(0, 30), (30, 50), (50, 70), (70, 101)]
        cells = []
        for lo, hi in bins:
            its = [i for i in ids if lo <= q[i]["chrf"] < hi]
            if its:
                cells.append(f"chrF {lo}-{min(hi, 100)}: n={len(its)} self {np.mean([R[m]['self'][i]['mean_acc'] for i in its]):.3f}"
                             f" sw {np.mean([R[m]['sw'][i]['mean_acc'] for i in its]):.3f}"
                             f" en {np.mean([R[m]['en'][i]['mean_acc'] for i in its]):.3f}")
        T.append("    " + " | ".join(cells))
        # among items the model solves from English (en maj right): how often is it recovered
        solv = [i for i in ids if R[m]["en"][i]["maj_correct"]]
        T.append(f"    items solved from English (n={len(solv)}): recovered by sw {np.mean([R[m]['sw'][i]['maj_correct'] for i in solv]):.3f}"
                 f", by self {np.mean([R[m]['self'][i]['maj_correct'] for i in solv]):.3f}"
                 + (f", by nllb {np.mean([R[m]['nllb'][i]['maj_correct'] for i in solv]):.3f}" if R[m]["nllb"] else ""))
    text = "\n".join(T) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text(text)
    return text
