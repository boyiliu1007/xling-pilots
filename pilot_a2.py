"""Pilot A2: English self-consensus quality on COPSD's training distribution (OpenThoughts) and GSM8K train.

  python pilot_a2.py --inspect_only
  python pilot_a2.py --limit 10 --output_dir pilot_a2_smoke
  python pilot_a2.py
  python pilot_a2.py --report_only
"""
import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from extract import extract_answer, majority_vote, numeric_equal
from mathcheck import cluster_vote, equiv, extract_pred, is_parseable, parse_answer
from pilot import MODEL, PROMPT, GpuPeak, versions
from pilot_c import wait_for_gpu
from stats import fmt, wilson

HERE = Path(__file__).resolve().parent
COPSD = "yihongLiu/COPSD-AfriMGSM-TrainDataset"
THRESHOLDS = [0.625, 0.75, 0.875, 1.0]
R3_ITEMS = 100
R3_BUDGET_S = 40 * 60
log = logging.getLogger("pilot_a2")


# ----------------------------------------------------------------------------- data

def load_data(args):
    from datasets import load_dataset
    from huggingface_hub import list_repo_files
    L = [f"Pilot A2 dataset inspection  {datetime.now().isoformat(timespec='seconds')}"]
    L.append("COPSD repo: https://github.com/cisnlp/COPSD (commit 4f4410730e9b) contains no data files; its README links the "
             f"released training data as the Hugging Face dataset {COPSD} (one parquet per language).")
    files = list_repo_files(COPSD, repo_type="dataset")
    L.append(f"files: {files}")
    cop = {}
    for lang in ["swa", "yor"]:
        path = f"data/{lang}-00000-of-00001.parquet"
        ds = load_dataset(COPSD, data_files=path, split="train")
        cop[lang] = [dict(r) for r in ds]
        L.append(f"\n{'=' * 80}\nCOPSD {lang}: {path}, format parquet, {len(ds)} rows\ncolumns: {ds.column_names}\nfeatures: {ds.features}")
        for i in (0, 1):
            L.append(f"--- full example row {i} ---\n" + json.dumps(cop[lang][i], ensure_ascii=False, indent=2))
        L.append(f"language values: {dict(Counter(r['language'] for r in cop[lang]))}; "
                 f"source values: {dict(Counter(r['source'] for r in cop[lang]))}")
    same = sum(a["problem"] == b["problem"] and a["Answer"] == b["Answer"] and a["solution"] == b["solution"]
               for a, b in zip(cop["swa"], cop["yor"]))
    L.append(f"\nalignment: no ID column; English problem/solution/Answer identical between swa and yor files row-wise: "
             f"{same}/{len(cop['swa'])} -> question_id = row index")
    if same != len(cop["swa"]):
        raise SystemExit("swa/yor English rows differ; cannot align by row")
    L.append("fields: English question = 'problem'; English reference solution = 'solution'; gold final answer = 'Answer'; "
             "target-language question = 'problem_translated' (language given by 'language'); 'source' = OpenThoughts subset")

    # gold sanity: Answer parseable by math-verify, and agreement with the last \boxed{} of the reference solution
    unp, agree, nobox, dis, consistent = [], 0, 0, [], set()
    for i, r in enumerate(cop["swa"]):
        gp = parse_answer(r["Answer"])
        if not gp:
            unp.append(i)
        s, p = extract_pred(r["solution"])
        if s is None:
            nobox += 1
        elif equiv(r["Answer"], gp, s, p):
            agree += 1
            if gp:
                consistent.add(i)
        else:
            dis.append((i, r["Answer"], s))
    L.append(f"gold 'Answer' unparseable by math-verify: {len(unp)}/{len(cop['swa'])} {unp[:10]}")
    L.append(f"'Answer' vs last \\boxed{{}} of 'solution': equivalent {agree}, solution has no \\boxed {nobox}, "
             f"disagree {len(dis)}; first 5 disagreements: {dis[:5]}")
    L.append(f"'Answer' length (chars): median {np.median([len(r['Answer']) for r in cop['swa']]):.0f}, "
             f"max {max(len(r['Answer']) for r in cop['swa'])}; integer-valued answers: "
             f"{sum(r['Answer'].strip().lstrip('-').isdigit() for r in cop['swa'])}")
    L.append(f"pre-registered (before generation): primary gold = 'Answer' on all items; sensitivity subset 'gold_consistent' = "
             f"'Answer' parseable and equivalent to the solution's last \\boxed{{}}: {len(consistent)} items")

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    idx = sorted(random.Random(args.seed).sample(range(len(gsm)), 500))
    gs = [dict(gsm[i]) for i in idx]
    L.append(f"\n{'=' * 80}\nGSM8K: openai/gsm8k config main split train ({len(gsm)} rows), columns {gsm.column_names}; "
             f"sampled 500 with random.Random({args.seed}).sample, sorted indices; first 10 indices {idx[:10]}")
    L.append("gold = text after '####' in 'answer', commas removed, numeric")
    L.append("--- full example (first sampled) ---\n" + json.dumps(gs[0], ensure_ascii=False, indent=2))
    out = Path(args.output_dir)
    (out / "dataset_inspection.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    def gsm_gold(a):
        return float(a.split("####")[-1].strip().replace(",", ""))
    en = [{"qid": i, "question": r["problem"], "gold": r["Answer"], "source": r["source"],
           "gold_consistent": i in consistent} for i, r in enumerate(cop["swa"])]
    runs = {
        "R1_copsd_en": dict(items=en, mode="math", thinking=False, max_tokens=2048),
        "R2_gsm8k_en": dict(items=[{"qid": int(i), "question": r["question"], "gold": gsm_gold(r["answer"])}
                                   for i, r in zip(idx, gs)], mode="numeric", thinking=False, max_tokens=2048),
        "R3_copsd_en_thinking": dict(items=en[:R3_ITEMS], mode="math", thinking=True, max_tokens=4096),
    }
    for lang in ["swa", "yor"]:
        runs[f"L_copsd_{lang}"] = dict(items=[{"qid": i, "question": r["problem_translated"], "gold": r["Answer"],
                                               "source": r["source"], "gold_consistent": i in consistent}
                                              for i, r in enumerate(cop[lang])],
                                       mode="math", thinking=False, max_tokens=2048)
    return runs


# ----------------------------------------------------------------------------- records

def make_record(run, it, outs, finish, ntoks, mode, thinking, n):
    base = {"run": run, "question_id": it["qid"], "question": it["question"], "gold_answer": it["gold"],
            "source": it.get("source"), "gold_consistent": it.get("gold_consistent"), "raw_outputs": outs, "finish_reasons": finish, "num_generated_tokens": ntoks}
    if mode == "numeric":
        preds = [extract_answer(o) for o in outs]
        maj, counts, share, tie = majority_vote(preds, n)
        corr = [numeric_equal(p, it["gold"]) for p in preds]
        ok = [p is not None for p in preds]
        maj_correct = bool(maj is not None and not tie and numeric_equal(maj, it["gold"]))
        pred_strs = preds
    else:
        gp = parse_answer(it["gold"])
        ex = [extract_pred(o, thinking) for o in outs]
        pred_strs = [s for s, _ in ex]
        parsed = [p for _, p in ex]
        ok = [is_parseable(s, p) for s, p in ex]
        corr = [o_ and equiv(it["gold"], gp, s, p) for o_, (s, p) in zip(ok, ex)]
        rep, counts, share, tie = cluster_vote(pred_strs, parsed, n)
        maj = pred_strs[rep] if rep is not None else None
        maj_correct = bool(rep is not None and equiv(it["gold"], gp, pred_strs[rep], parsed[rep]))
    base.update({
        "preds": pred_strs, "sample_correct": corr, "majority_answer": maj, "vote_counts": counts,
        "vote_share": share, "is_tie": tie, "majority_correct": maj_correct,
        "mean_sample_accuracy": sum(corr) / n, "any_sample_correct": any(corr),
        "num_parseable": sum(ok), "num_unparseable": n - sum(ok),
        "num_truncated": sum(f == "length" for f in finish),
    })
    return base


# ----------------------------------------------------------------------------- generation

def generate(args, runs):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed)
    tok = llm.get_tokenizer()
    log.info(f"vLLM loaded {MODEL} on {torch.cuda.get_device_name(0)}")
    out = Path(args.output_dir)
    stats_path = out / "run_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    # R1+R2 GPU time for the R3 budget rule, including runs finished in an earlier invocation
    gpu_s = sum(stats.get(k, {}).get("generation_s", 0) for k in ("R1_copsd_en", "R2_gsm8k_en") if k not in args.runs)
    order = ["R1_copsd_en", "R2_gsm8k_en", "R3_copsd_en_thinking", "L_copsd_swa", "L_copsd_yor"]
    for name in order:
        if name not in args.runs:
            continue
        cfg = runs[name]
        if name.startswith("R3") and gpu_s > R3_BUDGET_S:
            log.info(f"SKIP {name}: R1+R2 used {gpu_s / 60:.1f} min GPU time (> 40 min)")
            stats[name] = {"skipped": f"R1+R2 generation took {gpu_s / 60:.1f} min > 40 min"}
            stats_path.write_text(json.dumps(stats, indent=2))
            continue
        items = cfg["items"][: args.limit] if args.limit else cfg["items"]
        sc = dict(temperature=0.6, top_p=0.95, top_k=20) if cfg["thinking"] else dict(temperature=0.7, top_p=0.8, top_k=20)
        sp = SamplingParams(n=args.n, max_tokens=cfg["max_tokens"], seed=args.seed, **sc)
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=it["question"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=cfg["thinking"])
                   for it in items]
        log.info(f"{name}: {len(items)} items, sampling {sc}, max_tokens {cfg['max_tokens']}, thinking {cfg['thinking']}; "
                 f"prompt[0]:\n{prompts[0]}")
        t0 = time.time()
        res = llm.generate(prompts, sp)
        dt = time.time() - t0
        if not name.startswith(("R3", "L_")):
            gpu_s += dt
        recs = [make_record(name, it, [c.text for c in o.outputs], [c.finish_reason for c in o.outputs],
                            [len(c.token_ids) for c in o.outputs], cfg["mode"], cfg["thinking"], args.n)
                for it, o in zip(items, res)]
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ntok = sum(sum(r["num_generated_tokens"]) for r in recs)
        stats[name] = {"items": len(items), "generation_s": round(dt, 1), "generated_tokens": ntok,
                       "sampling": sc, "max_tokens": cfg["max_tokens"], "thinking": cfg["thinking"], "mode": cfg["mode"],
                       "finished": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        stats_path.write_text(json.dumps(stats, indent=2))
        log.info(f"{name}: {dt:.1f}s, {ntok} tokens, maj@{args.n} {sum(r['majority_correct'] for r in recs)}/{len(recs)} -> {name}.jsonl")
    del llm
    return stats


# ----------------------------------------------------------------------------- report

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def eligible(r, t):
    return (not r["is_tie"]) and r["majority_answer"] is not None and r["vote_share"] >= t - 1e-9


def run_metrics(name, R, n):
    N, S = len(R), len(R) * n
    T = [f" {name}  (N={N}, n={n})",
         f"   maj@{n} (ties wrong):     " + fmt(sum(r["majority_correct"] for r in R), N),
         f"   pass@1 (samples):         " + fmt(int(round(sum(r["mean_sample_accuracy"] for r in R) * n)), S)
         + "  (Wilson treats samples as independent)",
         f"   pass@{n}:                  " + fmt(sum(r["any_sample_correct"] for r in R), N),
         f"   tie rate:                 " + fmt(sum(r["is_tie"] for r in R), N),
         f"   no-answer (samples):      " + fmt(sum(r["num_unparseable"] for r in R), S),
         f"   no-answer (all {n} unparseable): " + fmt(sum(r["num_parseable"] == 0 for r in R), N),
         f"   truncated (hit max_tokens, samples): " + fmt(sum(r.get("num_truncated", 0) for r in R), S),
         f"   mean vote share:          {np.mean([r['vote_share'] for r in R]):.3f}",
         "   confidence filtering (coverage = eligible / all items; precision = correct among eligible):"]
    for t in THRESHOLDS:
        el = [r for r in R if eligible(r, t)]
        T.append(f"     t={t:.3f}: coverage {fmt(len(el), N)}   precision {fmt(sum(r['majority_correct'] for r in el), len(el))}")
    T.append("   vote-share calibration (majority accuracy by bucket; ties/no-answer count wrong):")
    for name_, fn in [("<=3/8", lambda k: k <= 3)] + [(f"{k}/8", (lambda kk: lambda k: k == kk)(k)) for k in range(4, 9)]:
        sel = [r for r in R if fn(round(r["vote_share"] * n))]
        T.append(f"     {name_:>6}: " + fmt(sum(r["majority_correct"] for r in sel), len(sel)))
    return T


def decision_row(R, t=0.625):
    el = [r for r in R if eligible(r, t)]
    k = sum(r["majority_correct"] for r in el)
    ci = wilson(k, len(el))
    cov = len(el) / len(R)
    ok = ci is not None and ci[0] >= 0.75 and cov >= 0.40
    return ok, k, len(el), ci, cov, len(R)


def build_report(args, cfg=None):
    out = Path(args.output_dir)
    n = args.n
    R = {p.stem: read_jsonl(p) for p in sorted(out.glob("*.jsonl"))}
    T = ["PILOT A2 — English self-consensus quality on the training distribution (Qwen3-1.7B)",
         f"generated {datetime.now().isoformat(timespec='seconds')}",
         "Every rate: k/N = estimate [95% Wilson]. Answer checking: math-verify (COPSD grade_answer logic) for COPSD runs; "
         "Pilot A numeric extraction for GSM8K."]
    insp = out / "dataset_inspection.txt"
    if insp.exists():
        T.append("\n## Data")
        for l in insp.read_text().splitlines():
            if l.startswith(("COPSD repo", "COPSD swa", "COPSD yor", "alignment", "fields:", "gold 'Answer'",
                             "'Answer'", "GSM8K", "gold =", "source values", "language values")):
                T.append("  " + (l if len(l) < 500 else l[:500] + " ..."))
    if cfg:
        T.append("\n## Config")
        for k in ("model", "prompt_template", "seed", "n", "versions", "args", "env"):
            T.append(f"  {k}: {json.dumps(cfg.get(k), ensure_ascii=False)}")
        T.append("  runs: " + json.dumps(cfg.get("runs"), ensure_ascii=False))

    T.append("\n## Per-run metrics")
    for name in ["R1_copsd_en", "R2_gsm8k_en", "R3_copsd_en_thinking", "L_copsd_swa", "L_copsd_yor"]:
        if name in R:
            T += run_metrics(name, R[name], n)
        else:
            T.append(f" {name}: not run" + (f" ({cfg['runs'][name]['skipped']})" if cfg and "skipped" in cfg["runs"].get(name, {}) else ""))

    if "R1_copsd_en" in R and R["R1_copsd_en"][0].get("source") is not None:
        T.append("\n## R1 by OpenThoughts source (maj@8; precision/coverage at t=0.625)")
        for s, c in Counter(r["source"] for r in R["R1_copsd_en"]).most_common():
            sel = [r for r in R["R1_copsd_en"] if r["source"] == s]
            el = [r for r in sel if eligible(r, 0.625)]
            T.append(f"   {s:>16} n={c:>3}: maj {fmt(sum(r['majority_correct'] for r in sel), len(sel))}  "
                     f"cov {fmt(len(el), len(sel))}  prec {fmt(sum(r['majority_correct'] for r in el), len(el))}")

    pa = HERE / "pilot_out" / "en.jsonl"
    if pa.exists():
        A = read_jsonl(pa)
        T.append("\n## Side by side with Pilot A English (AfriMGSM test; recomputed from pilot_out/en.jsonl, identical to pilot_out/report.txt)")
        T.append(f"   {'run':<24} {'maj@8':<30} {'cov@0.625':<30} {'prec@0.625':<30} truncated(samples)")
        for name, RR in [("PilotA_afrimgsm_en", A)] + [(k, R[k]) for k in ("R2_gsm8k_en", "R1_copsd_en", "R3_copsd_en_thinking") if k in R]:
            el = [r for r in RR if eligible(r, 0.625)]
            tr = sum(f == "length" for r in RR for f in r.get("finish_reasons", []))
            T.append(f"   {name:<24} {fmt(sum(r['majority_correct'] for r in RR), len(RR)):<30} {fmt(len(el), len(RR)):<30} "
                     f"{fmt(sum(r['majority_correct'] for r in el), len(el)):<30} {fmt(tr, len(RR) * n)}")

    if "R3_copsd_en_thinking" in R:
        r3 = R["R3_copsd_en_thinking"]
        r1 = {r["question_id"]: r for r in R["R1_copsd_en"]}
        r1s = [r1[r["question_id"]] for r in r3]
        T.append(f"\n## R3 (thinking) vs R1 (non-thinking) on the same {len(r3)} items")
        for nm, RR in (("R1 non-thinking", r1s), ("R3 thinking", r3)):
            el = [r for r in RR if eligible(r, 0.625)]
            T.append(f"   {nm:<16} maj@8 {fmt(sum(r['majority_correct'] for r in RR), len(RR))}  pass@8 "
                     f"{fmt(sum(r['any_sample_correct'] for r in RR), len(RR))}  cov@0.625 {fmt(len(el), len(RR))}  "
                     f"prec@0.625 {fmt(sum(r['majority_correct'] for r in el), len(el))}")
        b = sum(a["majority_correct"] and not c["majority_correct"] for a, c in zip(r3, r1s))
        c_ = sum(c["majority_correct"] and not a["majority_correct"] for a, c in zip(r3, r1s))
        T.append(f"   discordant pairs: thinking right & non-thinking wrong {b}; non-thinking right & thinking wrong {c_}")

    for lang in ["swa", "yor"]:
        k = f"L_copsd_{lang}"
        if k not in R or "R1_copsd_en" not in R:
            continue
        L_, E = R[k], {r["question_id"]: r for r in R["R1_copsd_en"]}
        if any(r["question_id"] not in E for r in L_):
            continue
        Ea = [E[r["question_id"]] for r in L_]
        W = [i for i, r in enumerate(L_) if not r["majority_correct"]]
        T.append(f"\n## Target language {lang} (COPSD translations) vs English R1")
        T.append(f"   L majority accuracy:                      " + fmt(sum(r["majority_correct"] for r in L_), len(L_)))
        T.append(f"   L_wrong:                                  {len(W)}/{len(L_)}")
        T.append(f"   English rescue rate P(EN maj correct | L_wrong): " + fmt(sum(Ea[i]["majority_correct"] for i in W), len(W)))
        T.append(f"   overall P(EN maj correct):                " + fmt(sum(r["majority_correct"] for r in Ea), len(Ea)))
        T.append(f"   P(L pass@8 | L_wrong):                    " + fmt(sum(L_[i]["any_sample_correct"] for i in W), len(W)))
        for t in THRESHOLDS:
            le = [r for r in L_ if eligible(r, t)]
            ew = [i for i in W if eligible(Ea[i], t)]
            T.append(f"   t={t:.3f}: L own consensus cov {fmt(len(le), len(L_))} prec {fmt(sum(r['majority_correct'] for r in le), len(le))} | "
                     f"EN on L_wrong cov {fmt(len(ew), len(W))} prec {fmt(sum(Ea[i]['majority_correct'] for i in ew), len(ew))}")
        st = lambda r: "correct" if r["majority_correct"] else ("tie/no-ans" if r["majority_answer"] is None else "wrong")
        ct = Counter((st(a), st(b)) for a, b in zip(L_, Ea))
        cats = ["correct", "wrong", "tie/no-ans"]
        T.append("   cross-tab rows = L majority, cols = EN majority")
        T.append("   " + "L | EN".rjust(12) + " " + " ".join(f"{c:>11}" for c in cats) + "   total")
        for a in cats:
            T.append("   " + a.rjust(12) + " " + " ".join(f"{ct[(a, b)]:>11}" for b in cats) + f"   {sum(ct[(a, b)] for b in cats):>5}")

    T.append("\n## Pre-registered decision (t = 0.625): USABLE iff Wilson LB of precision >= 0.75 AND coverage >= 0.40")
    T.append("   Each mode that was run is evaluated; a distribution is USABLE if any of its modes passes, and the mode is named.")
    verdict = {}
    for dist, modes in (("COPSD set", ["R1_copsd_en", "R3_copsd_en_thinking"]), ("GSM8K train", ["R2_gsm8k_en"])):
        res = []
        for m in modes:
            if m not in R:
                continue
            ok, k, e, ci, cov, N = decision_row(R[m])
            res.append((m, ok))
            T.append(f"   {dist} / {m}: {'pass' if ok else 'fail'}  precision {fmt(k, e)} (LB {'N/A' if ci is None else f'{ci[0]:.3f}'}), "
                     f"coverage {e}/{N} = {cov:.3f}")
            if R[m][0].get("gold_consistent") is not None:
                sub = [r for r in R[m] if r["gold_consistent"]]
                ok2, k2, e2, ci2, cov2, N2 = decision_row(sub)
                T.append(f"      sensitivity (gold_consistent subset, not used for the verdict): precision {fmt(k2, e2)}, "
                         f"coverage {e2}/{N2} = {cov2:.3f} -> {'pass' if ok2 else 'fail'}")
        good = [m for m, ok in res if ok]
        verdict[dist] = good
        T.append(f"   => {dist}: {'USABLE (relies on ' + ', '.join(good) + ')' if good else 'NOT USABLE'}")

    if cfg and cfg.get("runtime"):
        T.append("\n## Runtime and GPU (observed)")
        for k, v in cfg["runtime"].items():
            T.append(f"  {k}: {v}")
    T.append("\n## Limitations")
    T += ["  - One model, one prompt, one seed; n=8. COPSD set = 500 items, R3 only 100 items (wide intervals).",
          "  - math-verify can miss equivalences (e.g. unusual LaTeX, text answers, intervals vs sets) or accept loose ones; "
          "the gold 'Answer' check above bounds this only for the gold side.",
          "  - Non-thinking answers are read from the last \\boxed{}; thinking answers only from text after </think>.",
          "  - Target-language questions are COPSD's machine translations (not human); English is the original.",
          "  - Precision of the majority answer is measured against gold; it says nothing about the quality of the reasoning trace "
          "a self-distillation teacher would provide."]
    text = "\n".join(T) + "\n"
    (out / "report.txt").write_text(text)
    return text, verdict


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="pilot_a2_out")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["R1_copsd_en", "R2_gsm8k_en", "R3_copsd_en_thinking", "L_copsd_swa", "L_copsd_yor"],
                    help="subset of runs to generate (to continue after an interruption; finished runs' JSONL are kept)")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])
    for nm in ("httpx", "huggingface_hub", "filelock"):
        logging.getLogger(nm).setLevel(logging.WARNING)
    if args.report_only:
        cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else None
        print(build_report(args, cfg)[0])
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    log.info(f"=== pilot_a2 start; argv {sys.argv}; args {vars(args)}")
    runs = load_data(args)
    if args.inspect_only:
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_mathcheck.py"), str(HERE / "test_extract.py")],
                       capture_output=True, text=True, cwd=HERE)
    log.info("unit tests:\n" + r.stdout[-1500:])
    if r.returncode != 0:
        raise SystemExit("unit tests failed")
    wait_for_gpu()
    import torch
    torch.manual_seed(args.seed)
    cfg = {"model": MODEL, "dtype": "bfloat16", "gpu_memory_utilization": 0.9, "prompt_template": PROMPT, "seed": args.seed,
           "n": args.n, "versions": {**versions(), "math_verify": __import__("math_verify").__version__
                                     if hasattr(__import__("math_verify"), "__version__") else "0.9.0 (pip)"},
           "args": vars(args), "argv": sys.argv, "hostname": platform.node(),
           "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"}}
    mon = GpuPeak()
    mon.start()
    start = datetime.now()
    t0 = time.time()
    stats = generate(args, runs)
    mon.stop_ev.set()
    cfg["runs"] = stats
    cfg["runtime"] = {"start": start.isoformat(timespec="seconds"), "end": datetime.now().isoformat(timespec="seconds"),
                      "wall_clock_s (model load + all runs)": round(time.time() - t0, 1),
                      "gpu_generation_s per run": {k: v.get("generation_s") for k, v in stats.items()},
                      "peak_gpu_mem_used_MiB (nvidia-smi poll, incl. vLLM KV reservation)": mon.peak}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    text, verdict = build_report(args, cfg)
    print(text)
    log.info(f"verdict {verdict}")


if __name__ == "__main__":
    main()
