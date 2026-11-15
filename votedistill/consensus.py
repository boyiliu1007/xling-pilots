"""Self-consensus: sample n answers per question, majority-vote them, and score the vote against gold.

This is the label-free teacher signal of the proposal: the English majority answer (and an agreeing trace)
replaces COPSD's gold reference. Gold is used only to measure how reliable that signal is.

  python -m votedistill.consensus generate --out results/consensus_afrimgsm --runs afrimgsm:en afrimgsm:swa
  python -m votedistill.consensus generate --out results/consensus_train --runs copsd:en copsd:swa gsm8k:en
  python -m votedistill.consensus generate ... --thinking --max_tokens 4096     # thinking-mode voting
  python -m votedistill.consensus report --out results/consensus_train

Each run is <dataset>:<lang> and writes <out>/<dataset>_<lang>[_thinking].jsonl, one record per question.
Finished runs are skipped, so an interrupted job can simply be resubmitted. The report covers every run in <out>;
for each target-language run with a matching English run it also measures how often English rescues it.
"""
import argparse
import logging
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from . import data
from .answers_math import cluster_vote, equiv, extract_pred, is_parseable, parse_answer
from .answers_numeric import extract_answer, majority_vote, numeric_equal
from .paths import MODEL, REPO
from .stats import fmt
from .utils import read_json, read_jsonl, setup_logging, versions, write_json, write_jsonl

log = logging.getLogger("votedistill.consensus")

PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
# Answer checking per dataset: 'numeric' = number extraction (GSM8K-style), 'math' = math-verify (COPSD's grader).
ANSWER_MODE = {"afrimgsm": "numeric", "gsm8k": "numeric", "gsm8k3k": "numeric", "copsd": "math"}
GSM8K_N = {"gsm8k": 500, "gsm8k3k": 3000}   # GSM8K train sample size (data.gsm8k_items, same seed)
THRESHOLDS = [0.625, 0.75, 0.875, 1.0]


def sampling(thinking):
    """Qwen3's recommended sampling for each mode."""
    return dict(temperature=0.6, top_p=0.95, top_k=20) if thinking else dict(temperature=0.7, top_p=0.8, top_k=20)


def run_name(dataset, lang, thinking):
    return f"{dataset}_{lang}" + ("_thinking" if thinking else "")


# ----------------------------------------------------------------------------- records

def make_record(run, item, outs, finish, ntoks, mode, thinking, n):
    rec = {"run": run, "question_id": item["qid"], "question": item["question"], "gold_answer": item["gold"],
           "source": item.get("source"), "gold_consistent": item.get("gold_consistent"),
           "raw_outputs": outs, "finish_reasons": finish, "num_generated_tokens": ntoks}
    if mode == "numeric":
        preds = [extract_answer(o) for o in outs]
        maj, counts, share, tie = majority_vote(preds, n)
        corr = [numeric_equal(p, item["gold"]) for p in preds]
        ok = [p is not None for p in preds]
        maj_correct = bool(maj is not None and not tie and numeric_equal(maj, item["gold"]))
    else:
        gp = parse_answer(item["gold"])
        ex = [extract_pred(o, thinking) for o in outs]
        preds, parsed = [s for s, _ in ex], [p for _, p in ex]
        ok = [is_parseable(s, p) for s, p in ex]
        corr = [o_ and equiv(item["gold"], gp, s, p) for o_, (s, p) in zip(ok, ex)]
        rep, counts, share, tie = cluster_vote(preds, parsed, n)
        maj = preds[rep] if rep is not None else None
        maj_correct = bool(rep is not None and equiv(item["gold"], gp, preds[rep], parsed[rep]))
    rec.update({"preds": preds, "sample_correct": corr, "majority_answer": maj, "vote_counts": counts,
                "vote_share": share, "is_tie": tie, "majority_correct": maj_correct,
                "mean_sample_accuracy": sum(corr) / n, "any_sample_correct": any(corr),
                "num_parseable": sum(ok), "num_unparseable": n - sum(ok)})
    return rec


# ----------------------------------------------------------------------------- generation

def load_runs(specs, thinking, seed, lines):
    """specs: ['copsd:en', ...] -> {run_name: (dataset, items)}; checks AfriMGSM row alignment against English."""
    runs, en_cache = {}, {}
    for spec in specs:
        dataset, lang = spec.split(":")
        if dataset not in ANSWER_MODE:
            raise SystemExit(f"unknown dataset {dataset!r}; choose from {sorted(ANSWER_MODE)}")
        if dataset == "afrimgsm":
            items = data.afrimgsm_items(lang, lines)
            if lang != "en":
                if "en" not in en_cache:
                    en_cache["en"] = data.afrimgsm_items("en", [])
                data.check_alignment(en_cache["en"], items, lang, lines)
        elif dataset == "copsd":
            items = data.copsd_items(lang, lines)
        else:
            if lang != "en":
                raise SystemExit("gsm8k is English only")
            items = data.gsm8k_items(lines, n=GSM8K_N[dataset], seed=seed)
        runs[run_name(dataset, lang, thinking)] = (dataset, items)
    return runs


def cmd_generate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"Dataset inspection {datetime.now().isoformat(timespec='seconds')}"]
    runs = load_runs(args.runs, args.thinking, args.seed, lines)
    with open(out / "dataset_inspection.txt", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    si, sk = (map(int, args.shard.split("/")) if args.shard else (0, 1))
    tag = f".shard{si}of{sk}" if args.shard else ""     # shard i of k = items[i::k]; merge with `consensus merge`
    todo = {k: v for k, v in runs.items() if args.overwrite or not (out / f"{k}{tag}.jsonl").exists()}
    for k in runs.keys() - todo.keys():
        log.info(f"[skip] {k}.jsonl exists (use --overwrite to regenerate)")
    if not todo:
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True, text=True, cwd=REPO)
    log.info("unit tests:\n" + r.stdout[-1500:])
    if r.returncode != 0:
        raise SystemExit("unit tests failed; not generating")

    from .gpu import GpuPeak, wait_for_gpu
    wait_for_gpu()
    import torch
    from vllm import LLM, SamplingParams
    torch.manual_seed(args.seed)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed)
    tok = llm.get_tokenizer()
    sc = sampling(args.thinking)
    sp = SamplingParams(n=args.n, max_tokens=args.max_tokens, seed=args.seed, **sc)
    cfg_file = out / f"config{tag}.json"
    cfg = read_json(cfg_file, {})
    cfg.update({"model": MODEL, "prompt_template": PROMPT, "versions": versions()})
    cfg.setdefault("runs", {})
    for name, (dataset, items) in todo.items():
        items = items[: args.limit] if args.limit else items
        items = items[si::sk]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=it["question"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
                   for it in items]
        log.info(f"{name}: {len(items)} items, n={args.n}, {sc}, max_tokens {args.max_tokens}; prompt[0]:\n{prompts[0]}")
        mon = GpuPeak()
        mon.start()
        t0 = time.time()
        res = llm.generate(prompts, sp)
        dt = time.time() - t0
        mon.stop_ev.set()
        recs = [make_record(name, it, [c.text for c in o.outputs], [c.finish_reason for c in o.outputs],
                            [len(c.token_ids) for c in o.outputs], ANSWER_MODE[dataset], args.thinking, args.n)
                for it, o in zip(items, res)]
        write_jsonl(out / f"{name}{tag}.jsonl", recs)
        cfg["runs"][name] = {"items": len(items), "n": args.n, "sampling": sc, "max_tokens": args.max_tokens,
                             "thinking": args.thinking, "answer_mode": ANSWER_MODE[dataset], "seed": args.seed,
                             "generation_s": round(dt, 1), "generated_tokens": sum(sum(r["num_generated_tokens"]) for r in recs),
                             "peak_gpu_mem_used_MiB": mon.peak, "finished": datetime.now().isoformat(timespec="seconds"),
                             "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                             "shard": args.shard}
        write_json(cfg_file, cfg)
        log.info(f"{name}: {dt:.1f}s, maj@{args.n} {sum(r['majority_correct'] for r in recs)}/{len(recs)}")
    if not args.shard:
        print(build_report(out))


def cmd_merge(args):
    """Interleave shard i of k (items[i::k]) back into the original item order; config = union of shard configs."""
    out, k = Path(args.out), args.k
    for name in args.names:
        shards = [read_jsonl(out / f"{name}.shard{i}of{k}.jsonl") for i in range(k)]
        merged = [shards[j % k][j // k] for j in range(sum(map(len, shards)))]
        write_jsonl(out / f"{name}.jsonl", merged)
        cfgs = [read_json(out / f"config.shard{i}of{k}.json", {}) for i in range(k)]
        cfg = read_json(out / "config.json", {})
        cfg.update({x: cfgs[0][x] for x in ("model", "prompt_template", "versions")})
        r = dict(cfgs[0]["runs"][name])
        r.update(items=len(merged), generation_s=sum(c["runs"][name]["generation_s"] for c in cfgs),
                 generated_tokens=sum(c["runs"][name]["generated_tokens"] for c in cfgs),
                 shard=f"merged from {k} shards", host=[c["runs"][name]["host"] for c in cfgs],
                 slurm_job_id=[c["runs"][name]["slurm_job_id"] for c in cfgs])
        cfg.setdefault("runs", {})[name] = r
        write_json(out / "config.json", cfg)
        log.info(f"merged {name}: {len(merged)} items from {k} shards")
    print(build_report(out))


# ----------------------------------------------------------------------------- report

def eligible(r, t):
    """A question's majority is usable as a teacher label at confidence threshold t."""
    return (not r["is_tie"]) and r["majority_answer"] is not None and r["vote_share"] >= t - 1e-9


def n_truncated(r):
    return sum(f == "length" for f in r.get("finish_reasons", []))


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
         f"   truncated (samples):      " + fmt(sum(n_truncated(r) for r in R), S),
         f"   mean vote share:          {np.mean([r['vote_share'] for r in R]):.3f}",
         "   confidence filtering (coverage = eligible / all items; precision = majority correct among eligible):"]
    for t in THRESHOLDS:
        el = [r for r in R if eligible(r, t)]
        T.append(f"     t={t:.3f}: coverage {fmt(len(el), N)}   precision {fmt(sum(r['majority_correct'] for r in el), len(el))}")
    T.append("   vote-share calibration (majority accuracy by number of agreeing samples; ties/no-answer count wrong):")
    for k in range(n + 1):
        sel = [r for r in R if round(r["vote_share"] * n) == k]
        if sel:
            T.append(f"     {k}/{n}: " + fmt(sum(r["majority_correct"] for r in sel), len(sel)))
    if R[0].get("gold_consistent") is not None:
        sub = [r for r in R if r["gold_consistent"]]
        el = [r for r in sub if eligible(r, 0.625)]
        T.append(f"   gold-consistent subset ({len(sub)} items) at t=0.625: coverage {fmt(len(el), len(sub))}  "
                 f"precision {fmt(sum(r['majority_correct'] for r in el), len(el))}")
    return T


def status(r):
    return "correct" if r["majority_correct"] else ("tie/no-ans" if r["majority_answer"] is None else "wrong")


def rescue_section(lname, L, en):
    """How often the English majority is right (and confidently so) on questions the target language gets wrong."""
    E = {r["question_id"]: r for r in en}
    if any(r["question_id"] not in E for r in L):
        return []
    Ea = [E[r["question_id"]] for r in L]
    W = [i for i, r in enumerate(L) if not r["majority_correct"]]
    Lc = [i for i, r in enumerate(L) if r["majority_correct"]]
    T = [f"\n## {lname} vs English",
         f"   L majority accuracy:                              " + fmt(len(Lc), len(L)),
         f"   L_wrong (majority wrong, tie or no answer):       {len(W)}/{len(L)}",
         f"   English rescue rate P(EN maj correct | L_wrong):  " + fmt(sum(Ea[i]["majority_correct"] for i in W), len(W)),
         f"   overall P(EN maj correct):                        " + fmt(sum(r["majority_correct"] for r in Ea), len(Ea)),
         f"   P(EN maj correct | L majority correct):           " + fmt(sum(Ea[i]["majority_correct"] for i in Lc), len(Lc)),
         f"   P(L pass@n | L_wrong):                            " + fmt(sum(L[i]["any_sample_correct"] for i in W), len(W)),
         "   by threshold: EN consensus on L_wrong (coverage of L_wrong, precision) | L's own consensus (coverage, precision)"]
    for t in THRESHOLDS:
        ew = [i for i in W if eligible(Ea[i], t)]
        le = [r for r in L if eligible(r, t)]
        T.append(f"     t={t:.3f}: EN cov {fmt(len(ew), len(W))} prec {fmt(sum(Ea[i]['majority_correct'] for i in ew), len(ew))} | "
                 f"L cov {fmt(len(le), len(L))} prec {fmt(sum(r['majority_correct'] for r in le), len(le))}")
    ct = Counter((status(a), status(b)) for a, b in zip(L, Ea))
    cats = ["correct", "wrong", "tie/no-ans"]
    T.append("   cross-tab: rows = L majority, cols = EN majority")
    T.append("   " + "L | EN".rjust(12) + " " + " ".join(f"{c:>11}" for c in cats) + "   total")
    for a in cats:
        T.append("   " + a.rjust(12) + " " + " ".join(f"{ct[(a, b)]:>11}" for b in cats) + f"   {sum(ct[(a, b)] for b in cats):>5}")
    return T


def build_report(out):
    out = Path(out)
    R = {p.stem: read_jsonl(p) for p in sorted(out.glob("*.jsonl"))}
    cfg = read_json(out / "config.json", {})
    T = [f"Self-consensus report: {out.name}", f"generated {datetime.now().isoformat(timespec='seconds')}",
         "Every rate: k/N = estimate [95% Wilson]. Gold answers are used only to score the vote."]
    if cfg:
        T.append("\n## Config")
        for k in ("model", "prompt_template", "versions"):
            if k in cfg:
                T.append(f"  {k}: {cfg[k]}")
        for name, meta in cfg.get("runs", {}).items():
            T.append(f"  run {name}: {meta}")
    T.append("\n## Per-run metrics")
    for name, recs in R.items():
        T += run_metrics(name, recs, len(recs[0]["raw_outputs"]))
        if recs[0].get("source") is not None and len({r["source"] for r in recs}) > 1:
            T.append("   by source (maj; coverage and precision at t=0.625):")
            for s, c in Counter(r["source"] for r in recs).most_common():
                sel = [r for r in recs if r["source"] == s]
                el = [r for r in sel if eligible(r, 0.625)]
                T.append(f"     {s:>16} n={c:>3}: maj {fmt(sum(r['majority_correct'] for r in sel), len(sel))}  "
                         f"cov {fmt(len(el), len(sel))}  prec {fmt(sum(r['majority_correct'] for r in el), len(el))}")
    for name, recs in R.items():
        dataset, lang, *mode = name.split("_")
        en = "_".join([dataset, "en", *mode])
        if lang != "en" and en in R:
            T += rescue_section(name, recs, R[en])
    T.append("\n## Caveats")
    T += ["  - One model, one prompt, one seed per run; intervals on conditional rates use small subsets.",
          "  - Precision is about the voted final answer only, not the quality of the reasoning trace a teacher would see.",
          "  - Non-thinking answers come from the last \\boxed{}; thinking answers (math mode) only from text after </think>.",
          "  - AfriMGSM English is human-parallel (an upper bound); COPSD target-language questions are machine translations."]
    text = "\n".join(T) + "\n"
    (out / "report.txt").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--runs", nargs="+", required=True, help="dataset:lang, e.g. afrimgsm:swa copsd:en gsm8k:en")
    g.add_argument("--n", type=int, default=8)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--max_tokens", type=int, default=2048)
    g.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--shard", default=None, help="i/k: generate items[i::k] only (merge the k shards with `merge`)")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--names", nargs="+", required=True, help="run names, e.g. copsd_en_thinking")
    m.add_argument("--k", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging(Path(args.out) / "logs" / "consensus.log" if args.cmd == "generate" else None)
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    else:
        print(build_report(args.out))


if __name__ == "__main__":
    main()
