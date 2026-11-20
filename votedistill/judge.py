"""LLM judge (Claude on Amazon Bedrock) for the Swahili eval outputs: a 1-5 score per sample plus an error category.

How the score stays stable from call to call and run to run
  The judge never picks the 1-5 number. It answers four narrow checklist questions about the reasoning, each with
  written definitions (understood_problem, steps, supports_answer, reaches_gold). The final answer comes from the
  rule-based grader that is already in the eval file (math-verify on the \\boxed answer vs gold). A fixed table in
  code (`score_from`) turns those into the score:

    final answer   reasoning (judge checklist)                                       score
    correct        understood yes, steps all_correct, supports_answer yes              5
    correct        understood yes, steps minor_slip, supports_answer yes               4
    correct        steps major_errors, or understood partly/no (answer supported)      3
    correct        steps no_steps, or the work neither supports nor reaches it
                   (supports_answer no and reaches_gold no: right number, not earned)  2
    wrong          steps no_steps                                                      1
    wrong          reaches_gold yes (right value in the work, wrong or missing box)    3
    wrong          understood no                                                       1
    wrong          understood yes, steps minor_slip (one slip from right)              3
    wrong          understood yes, steps all_correct (correct but unfinished)          2
    wrong          understood partly, or steps major_errors                            2
  (rows are checked top to bottom within each half). An unearned right answer (2) deliberately scores below an
  honest near miss (3).

  Every record carries the rubric id (version + hash of the prompt, schema and table); report/compare refuse to mix
  rubric ids, judge models or effort levels. Runs judge the same items and sample indices (fixed seed), so two runs
  compare item by item. `consistency` re-judges samples to measure how often the judge gives the same answers.

Other fields: error_category (main failure, for the qualitative analysis), language_primary and swahili_fluency, and
with --ref (L0-eng English eval file) a comparison with the model's own English solution.

Environment: the default judge (Claude Sonnet 4.6 via its global inference profile) is served only by Bedrock's
bedrock-runtime endpoint and needs anthropic<2, so run this module with the separate env
/data/user_data/boyil2/envs/judge_legacy/bin/python. Models 4.7+ (anthropic.claude-*) use the Messages-API
endpoint and anthropic>=2 (the vllm_pilot env).

Auth: a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK and AWS_REGION, read from the shell or from the git-ignored
.bedrock_env at the repo root (`export AWS_BEARER_TOKEN_BEDROCK='...'` / `export AWS_REGION=us-east-1`).

  PY=/data/user_data/boyil2/envs/judge_legacy/bin/python
  E=results/exp2/decode/<run>/eval/step100_test.jsonl
  $PY -m votedistill.judge run $E --items 20 --samples 1 --dry-run   # plan and first prompt, sends nothing
  $PY -m votedistill.judge run $E --items 20 --samples 1             # pilot: measured cost per judgment
  $PY -m votedistill.judge run $E --budget 20                        # all 200 items x 1 sample, stops at $20
  $PY -m votedistill.judge report results/judge/<run>/step100_test.jsonl
  $PY -m votedistill.judge compare <baseline judged.jsonl> <method judged.jsonl>
  $PY -m votedistill.judge run $E --items 40 --samples 1 --out results/judge/<run>/step100_test.retest.jsonl
  $PY -m votedistill.judge consistency results/judge/<run>/step100_test.jsonl results/judge/<run>/step100_test.retest.jsonl
  $PY -m votedistill.judge handcheck results/judge/<run>/step100_test.jsonl --eval $E
  $PY -m votedistill.judge agreement results/judge/<run>/step100_test.handcheck.tsv
"""
import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .paths import REPO, RESULTS
from .stats import fmt, paired_bootstrap
from .utils import read_jsonl

# Sonnet 4.6 is the only Claude model this AWS account could use (2026-11-20; 4.7+ models were "not available for
# this account"). Its global inference profile is served by bedrock-runtime and needs anthropic<2 (LEGACY_ENV).
DEFAULT_MODEL = "global.anthropic.claude-sonnet-4-6"
DEFAULT_FALLBACK = "none"   # e.g. anthropic.claude-opus-4-8 (Messages-API models only): used if the judge declines
LEGACY_ENV = "/data/user_data/boyil2/envs/judge_legacy"   # uv venv, Python 3.11, "anthropic[bedrock]<2" + datasets
REGION_PREFIXES = {"global", "us", "eu", "jp", "apac", "au"}
# USD per million tokens (input, output), Amazon Bedrock on-demand. Regional (non-global) profiles cost 10% more.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-opus-4-8": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
          "claude-haiku-4-5": (1.0, 5.0)}
CHARS_PER_TOKEN = 3.0   # rough, for the dry-run estimate only (Swahili and garbled text tokenize poorly)
RUBRIC_VERSION = "v3"


class Judgment(BaseModel):
    analysis: str = Field(description="2-5 sentences in English: what the student did, where it first goes wrong.")
    understood_problem: Literal["yes", "partly", "no"]
    steps: Literal["all_correct", "minor_slip", "major_errors", "no_steps"]
    supports_answer: Literal["yes", "no"]
    reaches_gold: Literal["yes", "no"]
    error_category: Literal["none", "misread_question", "wrong_method", "arithmetic", "unfinished",
                            "repetition_loop", "incoherent_text", "answer_mismatch", "other"]
    language_primary: Literal["swahili", "english", "mixed", "other_language", "unintelligible"]
    swahili_fluency: Literal[1, 2, 3, 4, 5]
    reference_alignment: Literal["same_approach", "different_valid_approach", "diverges_with_error",
                                 "no_solution", "no_reference_trace"]


def score_from(j, final_correct):
    """The fixed 1-5 table (module docstring). j: the judge's checklist answers; final_correct: rule grader."""
    und, steps, sup, gold = j["understood_problem"], j["steps"], j["supports_answer"], j["reaches_gold"]
    if final_correct:
        if steps == "no_steps" or (sup == "no" and gold == "no"):   # the work does not produce the right answer
            return 2
        if steps == "major_errors" or und != "yes":
            return 3
        return 4 if steps == "minor_slip" else 5
    if steps == "no_steps":
        return 1
    if gold == "yes":
        return 3
    if und == "no":
        return 1
    if und == "yes" and steps == "minor_slip":
        return 3
    return 2


RUBRIC = """You are grading one output of a small language model that was asked to solve a grade-school math word \
problem posed in Swahili. The model was told to reason and put its final answer in \\boxed{}. Its output may contain \
a reasoning section ending in </think>, followed by a final answer section. Outputs can be long, code-switched, \
repetitive, cut off at the token limit, or partly unintelligible; grade what is actually there.

You are given the problem in English and in Swahili, the gold answer, and sometimes a reference solution in English \
written by the same model when it was asked the English version of the question. The gold answer and reference are \
privileged information for you; the student never saw them.

Grade the whole output: the reasoning section before </think> and the final answer section after it count \
together. A clean final section does not excuse incoherent, unrelated or wrong reasoning before it.

Judge the mathematics separately from the language: a correct solution written in broken Swahili or in English gets \
the same answers to the reasoning questions as one written in fluent Swahili. Language is rated only in \
language_primary and swahili_fluency. Answer each question by its definition below, not by an overall impression.

analysis: 2-5 sentences in English. Say what the student computed and where (if anywhere) it first goes wrong.

understood_problem:
- yes: works with the right quantities and answers the question that was asked.
- partly: misses or misreads one quantity or condition, but works toward the question that was asked.
- no: solves a different problem, or there is no recognisable attempt at this problem.

steps (all the work in the output, the reasoning section and the final answer section together):
- all_correct: every step that is needed is present and correct, and the reasoning is coherent throughout. Awkward \
wording, extra checking, or repetition after the answer do not count against it.
- minor_slip: exactly one small error (one arithmetic slip, one unit or rounding mistake, one dropped term) in an \
otherwise correct and coherent solution.
- major_errors: a wrong operation or setup, more than one error, steps that contradict each other, or a substantial \
part of the reasoning (for example most of the section before </think>) that is incoherent, unrelated to the problem, \
or contradicts the final solution, even when a correct solution appears elsewhere in the output.
- no_steps: no recognisable solution steps anywhere in the output: only an answer, word salad, unrelated text, or a \
loop before any work.

supports_answer:
- yes: the final \\boxed{} answer follows from the student's own written steps.
- no: the boxed answer appears without derivation, contradicts the steps, or there is no boxed answer.

reaches_gold:
- yes: somewhere in the work the student computes the gold value as the answer to the question, even if the boxed \
answer is different or missing.
- no: otherwise.

error_category: the single main reason the output falls short of a correct, fully supported answer.
- none: correct answer, correct steps, nothing wrong.
- misread_question: solves a different problem (wrong quantities, wrong question asked, mistranslated condition).
- wrong_method: understands the problem but sets it up wrongly (wrong operation, missing step, double counting).
- arithmetic: setup is right, a calculation is wrong.
- unfinished: stops or is cut off before an answer, without looping.
- repetition_loop: gets stuck repeating text.
- incoherent_text: mostly word salad or unrelated content, so no real attempt can be assessed.
- answer_mismatch: the work reaches the right value but the \\boxed{} answer is missing or differs from it.
- other: anything else (say what in analysis).
If several apply, pick the earliest failure in the output.

language_primary: the main language of the whole output. swahili / english / mixed (substantial parts in both) / \
other_language / unintelligible (not readable as any language).

swahili_fluency, for the Swahili text in the output, as a reader fluent in Swahili would judge it:
- 5: natural, grammatical Swahili throughout.
- 4: clearly understandable Swahili with minor errors or a few English terms.
- 3: understandable, but with frequent errors, invented words, or heavy code-switching.
- 2: largely broken; only fragments are meaningful Swahili.
- 1: no meaningful Swahili (English only, another language, or word salad).

reference_alignment, compared with the reference solution:
- same_approach: same solution path (same intermediate quantities), whatever the language.
- different_valid_approach: a different but correct path.
- diverges_with_error: starts like a sensible solution, then departs from the reference with an error.
- no_solution: there is no solution attempt to compare.
- no_reference_trace: no reference solution was given (only the gold answer)."""

RUBRIC_ID = RUBRIC_VERSION + "-" + hashlib.sha256(
    (RUBRIC + json.dumps(Judgment.model_json_schema(), sort_keys=True) + inspect.getsource(score_from)).encode()
).hexdigest()[:10]


# ----------------------------------------------------------------------------- problem text and references

def load_questions(bench):
    """{item: (english question, swahili question)} keyed by the eval file's `item` index."""
    from datasets import load_dataset
    if bench == "msvamp":
        en = load_dataset("Mathoctopus/MSVAMP", "en", split="test")["m_query"]
        sw = load_dataset("Mathoctopus/MSVAMP", "sw", split="test")["m_query"]
    else:
        from .data import load_afrimgsm
        en = [r["question"] for r in load_afrimgsm("en")]
        sw = [r["question"] for r in load_afrimgsm("swa")]
    assert len(en) == len(sw), (len(en), len(sw))
    return dict(enumerate(zip(en, sw)))


def final_part(text):
    """The written solution after </think> when there is one (shorter and cleaner than the full trace)."""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def pick_reference(rec):
    """The model's own English solution: the first untruncated sample whose answer equals its English majority vote.
    rec is one record of an English eval file (same format as the Swahili ones). None if nothing agrees."""
    preds = [p for p, f in zip(rec["preds"], rec["formatted"]) if f]
    if not preds:
        return None
    vote = Counter(preds).most_common(1)[0][0]
    for out, p, tr in zip(rec["outputs"], rec["preds"], rec["truncated"]):
        if p == vote and not tr:
            return {"text": final_part(out), "answer": vote, "vote_share": preds.count(vote) / len(rec["preds"])}
    return None


def item_block(item, gold, questions, ref):
    en, sw = questions[item]
    parts = [f"<problem_english>\n{en}\n</problem_english>", f"<problem_swahili>\n{sw}\n</problem_swahili>",
             f"<gold_answer>{gold}</gold_answer>"]
    if ref is None:
        parts.append("<reference_solution>none given</reference_solution>")
    else:
        parts.append(f"<reference_solution answer=\"{ref['answer']}\">\n{ref['text']}\n</reference_solution>")
    return "\n\n".join(parts)


def student_block(text):
    return f"<student_output>\n{text}\n</student_output>\n\nGrade this student output."


# ----------------------------------------------------------------------------- run

def default_out(eval_path):
    p = Path(eval_path).resolve()
    run = p.parent.parent.name if p.parent.name == "eval" else p.parent.name
    return RESULTS / "judge" / run / p.name


def select(recs, n_items, n_samples, seed):
    """Seeded item subset (0 = all) and the first n_samples samples of each: the same for every run of a split."""
    recs = list(recs)
    if n_items and n_items < len(recs):
        recs = sorted(random.Random(seed).sample(recs, n_items), key=lambda r: r["item"])
    return [(r, list(range(min(n_samples or len(r["outputs"]), len(r["outputs"]))))) for r in recs]


def price_of(model):
    return next((v for k, v in PRICES.items() if k in model), None)


def cost(usage, model):
    pr = price_of(model)
    if pr is None:
        return float("nan")
    i, o = pr
    regional = 1.1 if model.split(".", 1)[0] in REGION_PREFIXES - {"global"} else 1.0
    return regional * (usage["input_tokens"] * i + usage["cache_creation_input_tokens"] * i * 1.25
                       + usage["cache_read_input_tokens"] * i * 0.1 + usage["output_tokens"] * o) / 1e6


def load_env_file(path=REPO / ".bedrock_env"):
    """Read `export KEY=value` lines from the git-ignored .bedrock_env at the repo root; the shell's values win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip().removeprefix("export ").strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def uses_runtime(model):
    """Inference-profile ids (global./us./eu./jp./apac./au.) go to Bedrock's bedrock-runtime endpoint (the only one
    serving Claude 4.6 and older); plain anthropic.* ids go to the Messages-API endpoint (bedrock-mantle)."""
    return model.split(".", 1)[0] in REGION_PREFIXES


def make_client(args):
    import anthropic
    load_env_file()
    if "PASTE-YOUR" in os.environ.get("AWS_BEARER_TOKEN_BEDROCK", ""):
        sys.exit("Paste your Bedrock API key into .bedrock_env at the repo root first.")
    if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        sys.exit("Set AWS_REGION in .bedrock_env (e.g. us-east-1) together with AWS_BEARER_TOKEN_BEDROCK.")
    if uses_runtime(args.model):
        # Bedrock accepts the 2026-10-21 API version (anthropic >= 2.0) only for Claude 4.7+; older models need 1.x.
        if int(anthropic.__version__.split(".")[0]) >= 2:
            sys.exit(f"{args.model} needs anthropic<2 on Bedrock; run with {LEGACY_ENV}/bin/python")
        return anthropic.AsyncAnthropicBedrock(max_retries=8, timeout=600.0)
    from anthropic import AsyncAnthropicBedrockMantle, BetaRefusalFallbackMiddleware
    mw = [] if args.fallback_model == "none" else [BetaRefusalFallbackMiddleware([{"model": args.fallback_model}])]
    return AsyncAnthropicBedrockMantle(max_retries=8, timeout=600.0, middleware=mw)


async def judge_one(client, args, block, rec, k):
    import anthropic
    t0 = time.time()
    base = {"item": rec["item"], "sample": k, "gold": rec["gold"], "pred": rec["preds"][k],
            "rule_correct": bool(rec["correct"][k]), "truncated": rec["truncated"][k], "lid": rec["language"][k],
            "judge_model": args.model, "effort": args.effort, "rubric": RUBRIC_ID, "eval_file": str(args.eval)}
    request = dict(
        model=args.model,
        max_tokens=16000,
        system=RUBRIC,
        messages=[{"role": "user", "content": [
            # cached per item: all judged samples of one item share this prefix
            {"type": "text", "text": block, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": student_block(rec["outputs"][k])}]}],
        output_config={"effort": args.effort},
        output_format=Judgment,
    )
    try:
        if uses_runtime(args.model):   # Claude 4.6: adaptive thinking must be requested explicitly
            resp = await client.messages.parse(**request, thinking={"type": "adaptive"})
        else:
            resp = await client.beta.messages.parse(**request)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        raise SystemExit(f"Bedrock rejected the request ({type(e).__name__}): {e.message}\n"
                         "Check the API key, AWS_REGION, and that the model is enabled for your account in that region.")
    except (anthropic.BadRequestError, anthropic.RateLimitError, anthropic.InternalServerError,
            anthropic.APIConnectionError) as e:
        return {**base, "status": "error", "error": f"{type(e).__name__}: {getattr(e, 'message', e)}"}
    u = resp.usage
    usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
             "cache_read_input_tokens": u.cache_read_input_tokens or 0,
             "cache_creation_input_tokens": u.cache_creation_input_tokens or 0}
    out = {**base, "served_by": resp.model, "usage": usage, "cost_usd": cost(usage, args.model if uses_runtime(args.model) else resp.model),
           "seconds": round(time.time() - t0, 1), "stop_reason": resp.stop_reason}
    if resp.stop_reason == "refusal":
        cat = resp.stop_details.category if resp.stop_details else None
        return {**out, "status": "refusal", "error": f"refused ({cat})"}
    if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        return {**out, "status": "error", "error": f"no parsed output (stop_reason {resp.stop_reason})"}
    j = resp.parsed_output.model_dump()
    if "<reference_solution>none given</reference_solution>" in block:
        j["reference_alignment"] = "no_reference_trace"   # nothing to compare with; the judge sometimes guesses anyway
    return {**out, "status": "ok", **j, "score": score_from(j, base["rule_correct"])}


async def run_async(args, plan, blocks, out_path, done):
    client = make_client(args)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    totals = Counter()
    n_todo = sum(1 for r, ks in plan for k in ks if (r["item"], k) not in done)
    t0 = time.time()

    async def one(rec, k):
        async with sem:
            if totals["cost"] >= args.budget:
                totals["skipped (budget)"] += 1
                return
            res = await judge_one(client, args, blocks[rec["item"]], rec, k)
        async with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            totals[res["status"]] += 1
            totals["cost"] += res.get("cost_usd", 0.0) or 0.0
            n = totals["ok"] + totals["error"] + totals["refusal"]
            if n % 10 == 0 or n == n_todo:
                print(f"  {n}/{n_todo} done ({totals['ok']} ok, {totals['error']} errors, {totals['refusal']} refusals),"
                      f" ${totals['cost']:.2f} so far, {time.time() - t0:.0f}s", flush=True)

    async def per_item(rec, ks):
        ks = [k for k in ks if (rec["item"], k) not in done]
        if ks:   # first sample alone writes the item's cache entry; the rest read it
            await one(rec, ks[0])
            await asyncio.gather(*(one(rec, k) for k in ks[1:]))

    await asyncio.gather(*(per_item(r, ks) for r, ks in plan))
    return totals


def cmd_run(args):
    recs = read_jsonl(args.eval)
    plan = select(recs, args.items, args.samples, args.seed)
    questions = load_questions(args.bench)
    num = lambda v: float(str(v).replace(",", ""))
    # guard against a wrong --bench: the eval file's gold must be this dataset's gold for the same item
    from .data import load_afrimgsm
    if args.bench == "afrimgsm":
        golds = {i: r["answer_number"] for i, r in enumerate(load_afrimgsm("swa"))}
    else:
        from datasets import load_dataset
        golds = dict(enumerate(load_dataset("Mathoctopus/MSVAMP", "sw", split="test")["response"]))
    bad = [r["item"] for r in recs if r["item"] not in golds or num(golds[r["item"]]) != num(r["gold"])]
    if bad:
        sys.exit(f"{len(bad)} items' gold answers do not match {args.bench} (e.g. items {bad[:5]}): wrong --bench?")
    refs = {}
    if args.ref:
        refs = {r["item"]: pick_reference(r) for r in read_jsonl(args.ref)}
        missing = [r["item"] for r, _ in plan if r["item"] not in refs]
        if missing:
            sys.exit(f"--ref has no records for items {missing[:10]}... (is it the same split?)")
    blocks = {r["item"]: item_block(r["item"], r["gold"], questions, refs.get(r["item"])) for r, _ in plan}
    out_path = Path(args.out) if args.out else default_out(args.eval)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        prev = [j for j in read_jsonl(out_path) if j["status"] == "ok"]
        other = {(j["rubric"], j["judge_model"], j["effort"]) for j in prev} - {(RUBRIC_ID, args.model, args.effort)}
        if other:
            sys.exit(f"{out_path} has judgments made with {other}; use a different --out for this setting")
        done = {(j["item"], j["sample"]) for j in prev}
    n_req = sum(1 for r, ks in plan for k in ks if (r["item"], k) not in done)
    chars = sum(len(RUBRIC) + len(blocks[r["item"]]) + len(r["outputs"][k]) for r, ks in plan for k in ks
                if (r["item"], k) not in done)
    est_in = chars / CHARS_PER_TOKEN
    pr = price_of(args.model)
    print(f"eval file: {args.eval} ({args.bench}, gold answers checked)\nout: {out_path}\n"
          f"judge: {args.model} (effort {args.effort}), rubric {RUBRIC_ID}, fallback {args.fallback_model}\n"
          f"reference: {args.ref or 'gold answer only'}"
          f"{f' ({sum(v is None for v in refs.values())} items without an agreeing English solution)' if refs else ''}\n"
          f"{len(plan)} items x up to {max(len(ks) for _, ks in plan)} samples; {len(done)} already judged; "
          f"{n_req} requests to send; budget ${args.budget:.2f} (list price)\n"
          f"estimated input ~{est_in / 1e6:.2f}M tokens"
          + (f" (~${est_in * pr[0] / 1e6:.2f} before caching; output and thinking tokens come on top)" if pr else ""))
    if args.dry_run:
        r, ks = plan[0]
        print("\n" + "=" * 80 + "\nSYSTEM:\n" + RUBRIC + "\n\n" + "=" * 80 + "\nUSER (cached part):\n"
              + blocks[r["item"]] + "\n\n" + "=" * 80 + "\nUSER (per sample):\n" + student_block(r["outputs"][ks[0]]))
        return
    if n_req == 0:
        print("nothing to do")
        return
    totals = asyncio.run(run_async(args, plan, blocks, out_path, done))
    print(f"finished: {dict(totals)}")
    n_paid = totals["ok"] + totals["refusal"]
    if n_paid:
        per = totals["cost"] / n_paid
        print(f"mean cost per judgment ${per:.4f} (list price). Projected cost of one run over all {len(recs)} items: "
              + ", ".join(f"{k} sample{'s' if k > 1 else ''}/item ${per * k * len(recs):.0f}" for k in (1, 2, 3, 4)))
    if totals["skipped (budget)"]:
        print(f"stopped at the ${args.budget:.2f} budget; re-run with a higher --budget to continue")
    if totals["error"]:
        print("some requests failed; re-run the same command to retry only those")


# ----------------------------------------------------------------------------- analysis helpers

def load_judged(path):
    """Last successful judgment per (item, sample); refuses files that mix rubric ids, judge models or efforts."""
    J = list({(j["item"], j["sample"]): j for j in read_jsonl(path) if j["status"] == "ok"}.values())
    if not J:
        sys.exit(f"{path}: no successful judgments")
    settings = {(j["rubric"], j["judge_model"], j["effort"]) for j in J}
    if len(settings) > 1:
        sys.exit(f"{path} mixes judge settings {settings}")
    for j in J:   # recompute: the table in score_from is the definition
        j["score"] = score_from(j, j["rule_correct"])
    return J


def setting(J):
    return J[0]["rubric"], J[0]["judge_model"], J[0]["effort"]


def item_means(J):
    by = defaultdict(list)
    for j in J:
        by[j["item"]].append(j["score"])
    return {i: float(np.mean(v)) for i, v in by.items()}


def mean_ci(values):
    m, lo, hi = paired_bootstrap(np.zeros(len(values)), values)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def dist(rows, field, values):
    c = Counter(r[field] for r in rows)
    return "  ".join(f"{v} {fmt(c[v], len(rows))}" for v in values if c[v])


def vals(field):
    return list(Judgment.model_fields[field].annotation.__args__)   # Literal values in schema order


# ----------------------------------------------------------------------------- report / compare

def cmd_report(args):
    J = load_judged(args.judged)
    rub, model, eff = setting(J)
    im = item_means(J)
    L = [f"Judge report: {args.judged}",
         f"{len(J)} judgments on {len(im)} items; judge {model} (effort {eff}), rubric {rub}; served by "
         f"{Counter(j['served_by'] for j in J).most_common()}; cost ${sum(j['cost_usd'] for j in J):.2f} (list price)",
         "",
         f"SCORE (1-5): mean {mean_ci(list(im.values()))} (per-item means, 95% bootstrap CI over items)",
         f"  distribution: {dist(J, 'score', [5, 4, 3, 2, 1])}",
         f"  rule-correct samples: {fmt(sum(j['rule_correct'] for j in J), len(J))}", ""]
    for name, rows in [("rule-correct samples", [j for j in J if j["rule_correct"]]),
                       ("rule-wrong samples", [j for j in J if not j["rule_correct"]])]:
        if rows:
            L.append(f"## {name} (n = {len(rows)})")
            for f in ["score", "understood_problem", "steps", "supports_answer", "reaches_gold"]:
                L.append(f"  {f}: {dist(rows, f, [5, 4, 3, 2, 1] if f == 'score' else vals(f))}")
            L.append("")
    L.append("## error categories (samples with score < 5)")
    low = [j for j in J if j["score"] < 5]
    L.append(f"  {dist(low, 'error_category', vals('error_category'))}" if low else "  none")
    L += ["", "## language",
          f"  language_primary: {dist(J, 'language_primary', vals('language_primary'))}",
          f"  swahili_fluency: mean {np.mean([j['swahili_fluency'] for j in J]):.2f}; "
          f"{dist(J, 'swahili_fluency', [5, 4, 3, 2, 1])}"]
    lid_sw = [j for j in J if j["lid"] == "sw"]
    if lid_sw:
        L.append(f"  LID says Swahili but fluency <= 2: {fmt(sum(j['swahili_fluency'] <= 2 for j in lid_sw), len(lid_sw))}")
    if any(j["reference_alignment"] != "no_reference_trace" for j in J):
        L.append(f"  reference_alignment: {dist(J, 'reference_alignment', vals('reference_alignment'))}")
    text = "\n".join(L)
    print(text)
    Path(args.judged).with_suffix(".report.txt").write_text(text + "\n")


def cmd_compare(args):
    """Paired comparison on the (item, sample) pairs judged in both files: B - A."""
    A, B = load_judged(args.a), load_judged(args.b)
    if setting(A) != setting(B):
        sys.exit(f"different judge settings: {setting(A)} vs {setting(B)}")
    ka, kb = {(j["item"], j["sample"]): j for j in A}, {(j["item"], j["sample"]): j for j in B}
    common = sorted(set(ka) & set(kb))
    if not common:
        sys.exit("no (item, sample) pairs judged in both files")
    ma = item_means([ka[k] for k in common])
    mb = item_means([kb[k] for k in common])
    items = sorted(ma)
    d, lo, hi = paired_bootstrap([ma[i] for i in items], [mb[i] for i in items])
    print(f"A = {args.a}\nB = {args.b}\n{len(common)} shared judgments on {len(items)} items; judge {setting(A)}")
    print(f"  mean score A {np.mean(list(ma.values())):.2f}, B {np.mean(list(mb.values())):.2f}; "
          f"B - A = {d:+.2f} [{lo:+.2f}, {hi:+.2f}] (95% paired bootstrap over items)")
    for name, J, kk in [("A", A, ka), ("B", B, kb)]:
        print(f"  {name} score distribution: {dist([kk[k] for k in common], 'score', [5, 4, 3, 2, 1])}")


def cmd_consistency(args):
    """Same samples judged twice with the same settings: how often does the judge give the same answers?"""
    A, B = load_judged(args.a), load_judged(args.b)
    if setting(A) != setting(B):
        sys.exit(f"different judge settings: {setting(A)} vs {setting(B)}")
    ka, kb = {(j["item"], j["sample"]): j for j in A}, {(j["item"], j["sample"]): j for j in B}
    common = sorted(set(ka) & set(kb))
    if not common:
        sys.exit("no (item, sample) pairs judged in both files")
    sa = np.array([ka[k]["score"] for k in common])
    sb = np.array([kb[k]["score"] for k in common])
    print(f"{len(common)} samples judged twice, judge {setting(A)}")
    print(f"  score identical: {fmt(int((sa == sb).sum()), len(common))}")
    print(f"  score within 1:  {fmt(int((abs(sa - sb) <= 1).sum()), len(common))}")
    print(f"  quadratic-weighted kappa: {qwk(sa, sb):.3f}")
    for f in ["understood_problem", "steps", "supports_answer", "reaches_gold", "error_category", "swahili_fluency"]:
        print(f"  {f} identical: {fmt(sum(ka[k][f] == kb[k][f] for k in common), len(common))}")
    for k in common:
        if ka[k]["score"] != kb[k]["score"]:
            print(f"    item {k[0]} sample {k[1]}: {ka[k]['score']} vs {kb[k]['score']}  "
                  f"(steps {ka[k]['steps']} / {kb[k]['steps']}, understood {ka[k]['understood_problem']} / "
                  f"{kb[k]['understood_problem']})")


def qwk(a, b, levels=(1, 2, 3, 4, 5)):
    """Quadratic-weighted Cohen's kappa between two integer ratings."""
    n = len(levels)
    idx = {v: i for i, v in enumerate(levels)}
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1.0 - (W * O).sum() / den if den else 1.0


# ----------------------------------------------------------------------------- hand check

HAND_FIELDS = ["score", "understood_problem", "steps", "supports_answer", "reaches_gold", "error_category"]


def cmd_handcheck(args):
    """Sample judgments for a manual check, spread over the five score levels."""
    J = load_judged(args.judged)
    eval_recs = {r["item"]: r for r in read_jsonl(args.eval)} if args.eval else None
    rng = random.Random(args.seed)
    by = defaultdict(list)
    for j in J:
        by[j["score"]].append(j)
    pick = []
    while len(pick) < min(args.n, len(J)):   # round-robin over score levels
        for s in [5, 4, 3, 2, 1]:
            if by[s] and len(pick) < args.n:
                pick.append(by[s].pop(rng.randrange(len(by[s]))))
    pick.sort(key=lambda j: (j["item"], j["sample"]))
    base = Path(args.judged).with_suffix("")
    md = [f"# Hand check of {args.judged}", "",
          "Read each output and fill the human_* columns in the .tsv next to this file with the label you would give "
          "(same label sets as the judge; leave blank to agree). Fill the checklist fields, not the score: the score "
          "follows from them by the fixed table in votedistill/judge.py. Then: "
          "python -m votedistill.judge agreement <tsv>", ""]
    for n, j in enumerate(pick, 1):
        md += [f"## {n}. item {j['item']} sample {j['sample']}  (gold {j['gold']}, pred {j['pred']}, "
               f"rule {'correct' if j['rule_correct'] else 'wrong'}, truncated {j['truncated']}, LID {j['lid']})", ""]
        if eval_recs:
            md += ["<details><summary>student output</summary>", "", "~~~~~~",
                   eval_recs[j["item"]]["outputs"][j["sample"]], "~~~~~~", "</details>", ""]
        md += [f"**judge**: score {j['score']}; "
               + ", ".join(f"{f}={j[f]}" for f in Judgment.model_fields if f != "analysis"), "",
               f"> {j['analysis']}", ""]
    Path(f"{base}.handcheck.md").write_text("\n".join(md))
    with open(f"{base}.handcheck.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item", "sample", "rule_correct"] + HAND_FIELDS
                   + [f"human_{x}" for x in HAND_FIELDS if x != "score"] + ["note"])
        for j in pick:
            w.writerow([j["item"], j["sample"], j["rule_correct"]] + [j[x] for x in HAND_FIELDS]
                       + [""] * (len(HAND_FIELDS) - 1) + [""])
    print(f"wrote {base}.handcheck.md and {base}.handcheck.tsv ({len(pick)} judgments)")


def cmd_agreement(args):
    rows = list(csv.DictReader(open(args.tsv), delimiter="\t"))
    print(f"{len(rows)} hand-checked judgments (blank human_* = agrees with the judge)")
    for r in rows:
        human = {f: r[f"human_{f}"].strip() or r[f] for f in HAND_FIELDS if f != "score"}
        r["human_score"] = str(score_from(human, r["rule_correct"] == "True"))
    for f in HAND_FIELDS:
        human = [r[f"human_{f}"].strip() or r[f] for r in rows]
        k = sum(h == r[f] for h, r in zip(human, rows))
        print(f"  {f}: judge agrees with you on {fmt(k, len(rows))}")
        for h, r in zip(human, rows):
            if h != r[f]:
                print(f"      item {r['item']} sample {r['sample']}: judge {r[f]}, you {h}")
    js = np.array([int(r["score"]) for r in rows])
    hs = np.array([int(r["human_score"]) for r in rows])
    print(f"  score within 1: {fmt(int((abs(js - hs) <= 1).sum()), len(rows))}; "
          f"quadratic-weighted kappa {qwk(js, hs):.3f}")


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m votedistill.judge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="judge samples of one eval JSONL")
    r.add_argument("eval", help="eval JSONL (results/.../eval/<name>_<split>.jsonl)")
    r.add_argument("--ref", help="English eval JSONL of the same items (L0-eng); its majority-agreeing solution is "
                                 "the reference. Without it: gold answer only")
    r.add_argument("--bench", choices=["afrimgsm", "msvamp"], default="afrimgsm")
    r.add_argument("--items", type=int, default=0, help="seeded random subset of items (0 = all)")
    r.add_argument("--samples", type=int, default=1, help="first K samples per item (default 1; 0 = all)")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--budget", type=float, default=20.0, help="stop sending once this many USD are spent (list price)")
    r.add_argument("--model", default=DEFAULT_MODEL, help="Bedrock model id, e.g. anthropic.claude-sonnet-5-5")
    r.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--fallback-model", default=DEFAULT_FALLBACK, help="model used if the judge declines; 'none' to disable")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--out", help="output JSONL (default results/judge/<run>/<eval file name>)")
    r.add_argument("--dry-run", action="store_true", help="print the plan, cost estimate and first prompt; send nothing")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("report", help="summarise a judged JSONL")
    s.add_argument("judged")
    s.set_defaults(func=cmd_report)
    c = sub.add_parser("compare", help="paired score comparison of two judged files (B - A)")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)
    k = sub.add_parser("consistency", help="same samples judged twice: how often the judge agrees with itself")
    k.add_argument("a")
    k.add_argument("b")
    k.set_defaults(func=cmd_consistency)
    h = sub.add_parser("handcheck", help="sample judgments for a manual check (.md to read, .tsv to fill)")
    h.add_argument("judged")
    h.add_argument("--eval", help="the eval JSONL that was judged, to include the student outputs")
    h.add_argument("--n", type=int, default=20)
    h.add_argument("--seed", type=int, default=0)
    h.set_defaults(func=cmd_handcheck)
    a = sub.add_parser("agreement", help="judge-vs-you agreement from a filled .handcheck.tsv")
    a.add_argument("tsv")
    a.set_defaults(func=cmd_agreement)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
