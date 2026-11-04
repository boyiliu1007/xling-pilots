"""Pilot A: cross-lingual English self-consensus on AfriMGSM with Qwen3-1.7B.

Stages: dataset inspection + alignment -> unit tests -> generation -> report.
  python pilot.py --inspect_only            # Stage 1 only (CPU)
  python pilot.py --langs swa yor --limit 20 --n 8
  python pilot.py --report_only             # rebuild report.txt from existing JSONL
"""
import argparse
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from extract import extract_answer, majority_vote, numeric_equal, parse_number
from stats import fmt, wilson

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-1.7B"
AFRI = "masakhane/afrimgsm"
MGSM = "juletxara/mgsm"
PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
THRESHOLDS = [0.625, 0.75, 0.875, 1.0]
MAX_MISMATCH = 0.05
log = logging.getLogger("pilot")


# ----------------------------------------------------------------------------- data

def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_lang(lang, configs, inspect_lines):
    """Load test split for a language. Returns (rows, source, fields)."""
    from datasets import load_dataset
    if lang == "en":
        cfg = pick(configs["afri"], ["eng", "en", "english"])
        src = (AFRI, cfg) if cfg else (MGSM, "en")
    else:
        if lang not in configs["afri"]:
            raise SystemExit(f"Language {lang!r} not in {AFRI} configs {configs['afri']}")
        src = (AFRI, lang)
    ds = load_dataset(src[0], src[1], split="test")
    cols = ds.column_names
    fields = {
        "question": pick(cols, ["question", "Question", "input", "problem"]),
        "gold": pick(cols, ["answer_number", "answer_num", "target", "numeric_answer", "label"]),
        "id": pick(cols, ["id", "idx", "question_id", "qid", "index", "ID"]),
    }
    rows = [dict(r) for r in ds]
    L = inspect_lines
    L.append(f"\n{'=' * 80}\nLANGUAGE {lang}: {src[0]} / {src[1]} / test  ({len(rows)} rows)")
    L.append(f"columns: {cols}")
    L.append(f"features: {ds.features}")
    L.append(f"detected fields: {fields}")
    L.append("--- one full example row (row 0) ---")
    L.append(json.dumps(rows[0], ensure_ascii=False, indent=2))
    L.append("--- first five examples ---")
    for i, r in enumerate(rows[:5]):
        L.append(f"[{i}] " + json.dumps(r, ensure_ascii=False))
    missing = [k for k in ("question", "gold") if fields[k] is None]
    if missing:
        raise SystemExit(f"{lang}: required field(s) {missing} not found in columns {cols}")
    return rows, src, fields


def norm_gold(v):
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    return parse_number(str(v))


def question_numbers(q):
    return Counter(n for n in (parse_number(t) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", q or "")) if n is not None)


def align(data, targets, lines):
    """Verify alignment of each target language with English. Returns {lang: ok}."""
    en_rows, _, en_f = data["en"]
    ok = {}
    for lang in targets:
        rows, _, f = data[lang]
        L = lines
        L.append(f"\n{'=' * 80}\nALIGNMENT {lang} vs en")
        if f["id"] and en_f["id"]:
            method = f"explicit ID field ({f['id']!r} vs {en_f['id']!r})"
            en_by = {r[en_f["id"]]: r for r in en_rows}
            pairs = [(r, en_by.get(r[f["id"]])) for r in rows]
        else:
            method = ("row order (no ID field in either dataset). AfriMGSM is a row-by-row human translation of "
                      "MGSM/GSM8K-test, but row order is verified here, not assumed: (a) gold answers compared "
                      "item by item, (b) numbers appearing in the question text compared to the English question")
            if len(rows) != len(en_rows):
                L.append(f"ROW COUNT MISMATCH: {len(rows)} vs {len(en_rows)}")
            pairs = list(zip(rows, en_rows))
        L.append(f"method: {method}")
        mism, aligned, overlaps = [], [], []
        for i, (r, e) in enumerate(pairs):
            g = norm_gold(r[f["gold"]])
            ge = norm_gold(e[en_f["gold"]]) if e else None
            qn, en_qn = question_numbers(r[f["question"]]), question_numbers(e[en_f["question"]]) if e else Counter()
            tot = sum((qn | en_qn).values())
            ov = sum((qn & en_qn).values()) / tot if tot else 1.0
            overlaps.append(ov)
            rec = {"idx": i, "gold": g, "en_gold": ge, "num_overlap": round(ov, 3),
                   "q": r[f["question"]][:200], "en_q": (e[en_f["question"]][:200] if e else None)}
            (aligned if numeric_equal(g, ge) else mism).append(rec)
        n = len(pairs)
        rate = len(mism) / n if n else 1.0
        ov = np.array(overlaps)
        L.append(f"gold mismatches: {len(mism)}/{n} = {rate:.4f}")
        L.append(f"question-number multiset overlap (Jaccard) vs English: mean {ov.mean():.3f}, "
                 f"median {np.median(ov):.3f}, items with overlap < 0.5: {(ov < 0.5).sum()}")
        # Shifted-index control: if rows were offset by one, overlap should collapse.
        sh = [sum((question_numbers(pairs[i][0][f['question']]) & question_numbers(pairs[i + 1][1][en_f['question']])).values())
              / max(1, sum((question_numbers(pairs[i][0][f['question']]) | question_numbers(pairs[i + 1][1][en_f['question']])).values()))
              for i in range(n - 1)] if n > 1 else [0]
        L.append(f"control: overlap with the NEXT English row (off-by-one) mean {np.mean(sh):.3f}")
        L.append("first five mismatches:")
        for m in mism[:5]:
            L.append("  " + json.dumps(m, ensure_ascii=False))
        L.append("first five aligned examples:")
        for a in aligned[:5]:
            L.append("  " + json.dumps(a, ensure_ascii=False))
        low = [i for i, o in enumerate(overlaps) if o < 0.5]
        if low:
            L.append(f"low-overlap items (first 5 of {len(low)}):")
            for i in low[:5]:
                L.append(f"  [{i}] {lang}: {pairs[i][0][f['question']][:200]!r}\n       en: {pairs[i][1][en_f['question']][:200]!r}")
        ok[lang] = rate <= MAX_MISMATCH
        L.append(f"RESULT: {'OK' if ok[lang] else 'FAIL'} (threshold {MAX_MISMATCH:.0%})")
    return ok


def stage1(args):
    from datasets import get_dataset_config_names
    lines = [f"Dataset inspection  {datetime.now().isoformat(timespec='seconds')}"]
    configs = {"afri": get_dataset_config_names(AFRI)}
    lines.append(f"{AFRI} configs: {configs['afri']}")
    if not pick(configs["afri"], ["eng", "en", "english"]):
        configs["mgsm"] = get_dataset_config_names(MGSM)
        lines.append(f"{MGSM} configs: {configs['mgsm']}")
    langs = ["en"] + [l for l in args.langs if l != "en"]
    data = {}
    for lang in langs:
        data[lang] = load_lang(lang, configs, lines)
    ok = align(data, langs[1:], lines)
    out = Path(args.output_dir) / "dataset_inspection.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all(ok.values()):
        raise SystemExit(f"Alignment failed for {[l for l, v in ok.items() if not v]}; stopping before inference.")
    return data


# ----------------------------------------------------------------------------- generation

class GpuPeak(threading.Thread):
    """Polls nvidia-smi for peak memory.used of the visible GPU(s) (MiB)."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.peak, self.period, self.stop_ev = 0, period, threading.Event()

    def run(self):
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        while not self.stop_ev.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
                self.peak = max(self.peak, int(o.splitlines()[0]))
            except Exception:
                pass
            self.stop_ev.wait(self.period)


def sampling_cfg(args):
    if args.thinking:
        return dict(temperature=0.6, top_p=0.95, top_k=20)
    return dict(temperature=0.7, top_p=0.8, top_k=20)


def make_record(i, lang, q, gold, outs, n):
    preds = [extract_answer(o) for o in outs]
    maj, counts, share, tie = majority_vote(preds, n)
    correct = [numeric_equal(p, gold) for p in preds]
    return {
        "question_id": i, "language": lang, "question": q, "gold_answer": gold,
        "raw_outputs": outs, "preds": preds, "majority_answer": maj, "vote_counts": counts,
        "vote_share": share, "is_tie": tie,
        "majority_correct": bool(maj is not None and not tie and numeric_equal(maj, gold)),
        "mean_sample_accuracy": sum(correct) / n, "any_sample_correct": any(correct),
        "num_parseable": sum(p is not None for p in preds), "num_unparseable": sum(p is None for p in preds),
    }


def generate(args, data):
    import torch
    # FlashInfer's sampler JIT-compiles on first use and needs ninja/nvcc; use vLLM's PyTorch top-k/top-p sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    sc = sampling_cfg(args)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.n, max_tokens=args.max_tokens, seed=args.seed, **sc)
    log.info(f"vLLM loaded {MODEL}; GPU {torch.cuda.get_device_name(0)}; CUDA_VISIBLE_DEVICES="
             f"{os.environ.get('CUDA_VISIBLE_DEVICES')}; sampling {sp}")
    stats = {}
    for lang, (rows, _, f) in data.items():
        rows = rows[: args.limit] if args.limit else rows
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=r[f["question"]])}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.thinking) for r in rows]
        if lang == "en" or lang == next(iter(data)):
            log.info(f"example prompt ({lang}):\n{prompts[0]}")
        t0 = time.time()
        res = llm.generate(prompts, sp)
        dt = time.time() - t0
        ntok = sum(len(c.token_ids) for r in res for c in r.outputs)
        nlen = sum(c.finish_reason == "length" for r in res for c in r.outputs)
        path = Path(args.output_dir) / f"{lang}.jsonl"
        with open(path, "w") as fh:
            for i, (r, o) in enumerate(zip(rows, res)):
                rec = make_record(i, lang, r[f["question"]], norm_gold(r[f["gold"]]), [c.text for c in o.outputs], args.n)
                rec["finish_reasons"] = [c.finish_reason for c in o.outputs]
                rec["num_generated_tokens"] = [len(c.token_ids) for c in o.outputs]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[lang] = dict(questions=len(rows), seconds=dt, generated_tokens=ntok, truncated_samples=nlen)
        log.info(f"{lang}: {len(rows)} questions, {dt:.1f}s, {ntok} generated tokens, "
                 f"{nlen}/{len(rows) * args.n} samples hit max_tokens -> {path}")
    return stats


# ----------------------------------------------------------------------------- analysis

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def status(r):
    if r["majority_correct"]:
        return "correct"
    if r["majority_answer"] is None:
        return "tie/no-ans"
    return "wrong"


def eligible(r, t):
    return (not r["is_tie"]) and r["majority_answer"] is not None and r["vote_share"] >= t - 1e-9


def lang_metrics(recs, n):
    N = len(recs)
    S = N * n
    return [
        f"  Pass@1 (mean sample acc): {sum(r['mean_sample_accuracy'] * n for r in recs):.0f}/{S} = "
        + fmt(round(sum(r['mean_sample_accuracy'] * n for r in recs)), S).split('= ')[1]
        + "  (Wilson on samples; samples are not independent across a question)",
        f"  Maj@{n} (ties wrong):       " + fmt(sum(r["majority_correct"] for r in recs), N),
        f"  Pass@{n}:                   " + fmt(sum(r["any_sample_correct"] for r in recs), N),
        f"  No-answer rate (samples):  " + fmt(sum(r["num_unparseable"] for r in recs), S),
        f"  No-answer rate (questions, all {n} unparseable): " + fmt(sum(r["num_parseable"] == 0 for r in recs), N),
        f"  Tie rate:                  " + fmt(sum(r["is_tie"] for r in recs), N),
        f"  Mean vote share:           {np.mean([r['vote_share'] for r in recs]):.3f}",
        f"  Samples hit max_tokens:    " + fmt(sum(fr == 'length' for r in recs for fr in r.get('finish_reasons', [])), S),
    ]


def build_report(args, cfg=None, runinfo=None):
    out = Path(args.output_dir)
    en = read_jsonl(out / "en.jsonl")
    n = len(en[0]["preds"])
    targets = [l for l in args.langs if (out / f"{l}.jsonl").exists()]
    data = {l: read_jsonl(out / f"{l}.jsonl") for l in targets}
    ths = sorted(set(THRESHOLDS + [args.threshold]))
    R = []
    R.append("PILOT A — Cross-lingual English self-consensus (Qwen3-1.7B, AfriMGSM)")
    R.append(f"generated {datetime.now().isoformat(timespec='seconds')}")
    R.append("\n## Dataset and alignment")
    insp = out / "dataset_inspection.txt"
    if insp.exists():
        txt = insp.read_text()
        for l in txt.splitlines():
            if l.startswith(("LANGUAGE", "ALIGNMENT", "method:", "gold mismatches", "question-number", "control:",
                             "RESULT", "detected fields", "masakhane", "juletxara")):
                R.append("  " + l)
    R.append(f"  items evaluated: en={len(en)}, " + ", ".join(f"{l}={len(v)}" for l, v in data.items()))
    if cfg:
        R.append("\n## Model and generation config")
        for k in ("model", "dtype", "gpu_memory_utilization", "sampling", "n", "max_tokens", "enable_thinking",
                  "seed", "prompt_template", "versions", "args"):
            R.append(f"  {k}: {json.dumps(cfg.get(k), ensure_ascii=False)}")

    R.append("\n## Per-language metrics  (k/N = point [95% Wilson])")
    for l, recs in [("en", en)] + list(data.items()):
        R.append(f" {l}  (N={len(recs)}, n={n})")
        R += lang_metrics(recs, n)

    R.append("\n## Key metric: English rescue rate")
    en_acc = sum(r["majority_correct"] for r in en)
    decision = {}
    for l, recs in data.items():
        assert len(recs) == len(en)
        W = [i for i, r in enumerate(recs) if not r["majority_correct"]]
        resc = [i for i in W if en[i]["majority_correct"]]
        R.append(f" {l}: L_wrong = {len(W)}/{len(recs)} items (majority wrong, tie, or no answer)")
        R.append(f"   English rescue rate P(EN maj correct | L_wrong): " + fmt(len(resc), len(W)))
        R.append(f"   overall P(EN maj correct) (all items):          " + fmt(en_acc, len(en)))
        R.append(f"   rescued items as share of all items:            " + fmt(len(resc), len(recs)))
        R.append(f"   P(L pass@{n} | L_wrong):                          " + fmt(sum(recs[i]['any_sample_correct'] for i in W), len(W)))
        R.append(f"   P(EN maj correct | L majority correct):         "
                 + fmt(sum(en[i]['majority_correct'] for i, r in enumerate(recs) if r['majority_correct']),
                       sum(r['majority_correct'] for r in recs)))

    R.append("\n## Confidence filtering (eligible = not tie, parseable majority, vote_share >= t; gold only scores precision)")
    for l, recs in data.items():
        W = [i for i, r in enumerate(recs) if not r["majority_correct"]]
        R.append(f" {l}")
        R.append(f"  {'t':>6} | EN on L_wrong: elig  cov(/|L_wrong|)  precision [Wilson]  rescued"
                 f" | EN on all: elig  cov(/N)  precision | L on all: elig  cov(/N)  precision")
        for t in ths:
            ew = [i for i in W if eligible(en[i], t)]
            ewc = sum(en[i]["majority_correct"] for i in ew)
            ea = [i for i in range(len(en)) if eligible(en[i], t)]
            eac = sum(en[i]["majority_correct"] for i in ea)
            la = [i for i in range(len(recs)) if eligible(recs[i], t)]
            lac = sum(recs[i]["majority_correct"] for i in la)
            R.append(f"  {t:>6.3f} | {len(ew):>3}  {fmt(len(ew), len(W))}  prec {fmt(ewc, len(ew))}  rescued {ewc}"
                     f" | {len(ea):>3}  {fmt(len(ea), len(en))}  prec {fmt(eac, len(ea))}"
                     f" | {len(la):>3}  {fmt(len(la), len(recs))}  prec {fmt(lac, len(la))}")
            if abs(t - 0.625) < 1e-9:
                ci = wilson(ewc, len(ew))
                cov = len(ew) / len(W) if W else None
                passed = ci is not None and cov is not None and ci[0] >= 0.60 and cov >= 0.30
                decision[l] = (passed, ewc, len(ew), ci, len(W), cov)

    R.append("\n## Cross-tab: rows = L majority, cols = EN majority")
    cats = ["correct", "wrong", "tie/no-ans"]
    for l, recs in data.items():
        ct = Counter((status(recs[i]), status(en[i])) for i in range(len(en)))
        R.append(f" {l}")
        R.append("  " + "L | EN".rjust(12) + " " +" ".join(f"{c:>11}" for c in cats) + "   total")
        for a in cats:
            R.append(f"  {a:>12} " + " ".join(f"{ct[(a, b)]:>11}" for b in cats) + f"   {sum(ct[(a, b)] for b in cats):>5}")

    R.append(f"\n## Vote-share calibration (accuracy of majority within vote-share bucket; ties/no-answer count wrong)")
    for l, recs in [("en", en)] + list(data.items()):
        R.append(f" {l}")
        buckets = [("<=3/8", lambda k: k <= 3)] + [(f"{k}/8", (lambda kk: lambda k: k == kk)(k)) for k in range(4, 9)]
        if n != 8:
            buckets = [(f"{k}/{n}", (lambda kk: lambda k: k == kk)(k)) for k in range(0, n + 1)]
        for name, fn in buckets:
            sel = [r for r in recs if fn(round(r["vote_share"] * n))]
            R.append(f"  {name:>6}: " + fmt(sum(r["majority_correct"] for r in sel), len(sel)))

    R.append("\n## Pre-registered decision (t = 0.625): PASS iff Wilson LB of EN precision on L_wrong >= 0.60 "
             "AND EN coverage among L_wrong >= 0.30")
    for l, (p, k, m, ci, nw, cov) in decision.items():
        R.append(f"  {l}: {'PASS' if p else 'FAIL'}  precision {fmt(k, m)}  (LB {'N/A' if ci is None else f'{ci[0]:.3f}'}), "
                 f"coverage {m}/{nw} = {'N/A' if cov is None else f'{cov:.3f}'}")

    if runinfo:
        R.append("\n## Runtime and GPU (observed)")
        for k, v in runinfo.items():
            R.append(f"  {k}: {v}")
    R.append("\n## Limitations")
    R += ["  - Single model, single seed, single prompt; n=8 samples; one sampling configuration.",
          "  - Conditional rates on L_wrong use a subset of the 250 items; intervals are wide; items are one benchmark (GSM8K-derived).",
          "  - Pass@1 Wilson interval treats the n samples per question as independent, which understates uncertainty.",
          "  - English questions are the dataset's human-parallel English, an upper bound vs model translation (Pilot B).",
          "  - English MGSM items derive from GSM8K test, which may be contaminated in Qwen3 pretraining.",
          "  - Rescue measures availability of a correct teacher label, not whether distillation would help a student.",
          "  - Answer extraction is heuristic (last \\boxed{}, else last number)."]
    text = "\n".join(R) + "\n"
    (out / "report.txt").write_text(text)
    return text, decision


# ----------------------------------------------------------------------------- main

def versions():
    v = {"python": platform.python_version()}
    for m in ("torch", "vllm", "transformers", "datasets"):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:
            v[m] = f"unavailable ({type(e).__name__})"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["swa", "yor"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.625)
    ap.add_argument("--output_dir", default="pilot_out")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--inspect_only", action="store_true", help="Stage 1 only")
    ap.add_argument("--report_only", action="store_true", help="rebuild report.txt from existing JSONL")
    args = ap.parse_args()
    args.langs = [l for l in args.langs if l != "en"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])

    if args.report_only:
        cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else None
        print(build_report(args, cfg, cfg.get("runtime") if cfg else None)[0])
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    log.info(f"=== pilot start; argv: {sys.argv}; args: {vars(args)}")
    data = stage1(args)
    log.info("Stage 1 (inspection + alignment) passed")
    if args.inspect_only:
        return

    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_extract.py")],
                       capture_output=True, text=True, cwd=HERE)
    log.info("Stage 2 unit tests:\n" + r.stdout[-2000:])
    if r.returncode != 0:
        raise SystemExit("Unit tests failed; not running inference.")

    import torch
    torch.manual_seed(args.seed)
    cfg = {"model": MODEL, "dtype": "bfloat16", "gpu_memory_utilization": 0.9, "sampling": sampling_cfg(args),
           "n": args.n, "max_tokens": args.max_tokens, "enable_thinking": args.thinking, "seed": args.seed,
           "prompt_template": PROMPT, "versions": versions(), "args": vars(args), "argv": sys.argv,
           "hostname": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "env": {"VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0 (set in generate)")}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log.info(f"config: {json.dumps(cfg)}")

    mon = GpuPeak()
    mon.start()
    start = datetime.now()
    t0 = time.time()
    st = generate(args, data)
    wall = time.time() - t0
    end = datetime.now()
    mon.stop_ev.set()
    nq = sum(s["questions"] for s in st.values())
    runinfo = {"start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
               "wall_clock_s (model load + generation)": round(wall, 1),
               "generation_s (sum over languages)": round(sum(s["seconds"] for s in st.values()), 1),
               "questions_total": nq,
               "questions_per_min (generation only)": round(nq / (sum(s["seconds"] for s in st.values()) / 60), 2),
               "generated_tokens_total": sum(s["generated_tokens"] for s in st.values()),
               "peak_gpu_mem_used_MiB (nvidia-smi poll, includes vLLM KV-cache reservation)": mon.peak,
               "per_language": st}
    cfg["runtime"] = runinfo
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log.info(f"runtime: {json.dumps(runinfo)}")
    text, dec = build_report(args, cfg, runinfo)
    print(text)
    log.info("decision: " + json.dumps({l: v[0] for l, v in dec.items()}))


if __name__ == "__main__":
    main()
