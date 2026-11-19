"""Stage 3: assemble the SDRRL fine-tuning set.

For every problem with a voted English reference, four question/answer language combinations (SDRRL 3.2.2):
  en_en   English question  -> English reference
  tgt_en  target question   -> English reference
  en_tgt  English question  -> translated reference     (only if the translation passes the quality filter)
  tgt_tgt target question   -> translated reference     (same)
Questions come from the training set `data` (question_rows): the COPSD set (its English problem and its released
translation) or an Experiment 2 GSM8K set such as gsm8k-mt3k (English GSM8K question and its NLLB translation; only
questions that pass Experiment 2's translation QC). Only the reference is machine-translated here. Questions in all
four subsets are code-switched; answers are not. A small set of natural parallel sentences adds translation and
sentence-completion instructions.

Examples use the COPSD evaluation prompt, so the fine-tuned model is evaluated exactly like the COPSD runs:
  user:      <question>\\n\\n<COPSD instruction in the question's language>
  assistant: <think>\\n<COPSD think prefix in the ANSWER's language> <reasoning>\\n</think>\\n\\n<final answer>
Parallel-text instructions use an empty think block.

Output <out>/sft_data.jsonl ({"id", "subset", "question_id", "user", "assistant"}) and <out>/build_report.txt.
"""
import logging
import random
from collections import Counter
from pathlib import Path

from ..copsd.collator import import_copsd
from ..copsd.evaluate import language_of
from ..data import load_copsd_train
from ..utils import read_jsonl, write_json, write_jsonl
from .codeswitch import code_switch, load_dictionary
from .parallel import instructions, load_pairs

log = logging.getLogger("votedistill.sdrrl.build")
EMPTY_THINK = "<think>\n\n</think>\n\n"


def assistant_turn(think_prefix, reasoning, final):
    return f"<think>\n{think_prefix.strip()} {reasoning.strip()}\n</think>\n\n{final.strip()}"


def question_rows(data="copsd", lang="swa"):
    """{question_id: row with "problem" (English) and "problem_translated"} for the training set `data`.
    copsd: the released COPSD set, ids = row positions. Experiment 2 GSM8K sets (exp2/trainsets.py): ids = GSM8K qids
    (the consensus file's question_id), QC-failed translations already removed."""
    if data == "copsd":
        return dict(enumerate(load_copsd_train(lang)))
    from ..exp2.trainsets import _gsm8k_rows
    if lang != "swa":
        raise SystemExit(f"{data}: only Swahili questions are wired up for SDRRL")
    return {r["qid"]: r for r in _gsm8k_rows(data)}


def translation_verdict(rec, qe, threshold, lang):
    """None if the translated reference is usable, else the reason it is rejected."""
    if rec["reasoning_tgt"] is None:
        return "math_lost"
    if lang == "swa" and language_of(rec["reasoning_tgt"]) != "sw":
        return "off_target_language"
    if qe is not None:
        if qe.get("mean") is None:
            return "nothing_to_score"
        if qe["mean"] < threshold:
            return "low_quality"
    return None


def run(out_dir, lang="swa", threshold=0.8, swap_p=0.15, n_parallel=100, seed=42, no_qe=False, data="copsd"):
    out_dir = Path(out_dir)
    import_copsd()
    from language_config import LANGUAGE_CONFIG
    cfg = {"en": LANGUAGE_CONFIG["EN"], lang: LANGUAGE_CONFIG[lang.upper()]}
    rng = random.Random(seed)
    trans = read_jsonl(out_dir / "translations.jsonl")
    qe_path = out_dir / "qe_scores.jsonl"
    if qe_path.exists():
        qe = {r["question_id"]: r for r in read_jsonl(qe_path)}
        missing = [r["question_id"] for r in trans if r["question_id"] not in qe]
        if missing:
            raise SystemExit(f"{qe_path} lacks {len(missing)} problems (stale file?); rerun the quality stage")
    elif no_qe:
        qe = None
        log.warning("building WITHOUT the translation quality filter (--no_qe)")
    else:
        raise SystemExit(f"{qe_path} not found. Run `python -m votedistill.sdrrl.quality --out {out_dir}` in the comet "
                         "environment, or pass --no_qe to skip the CometKiwi filter (a deviation from SDRRL).")
    rows = question_rows(data, lang)
    dicts = {"en": load_dictionary("en", lang), lang: load_dictionary(lang, "en")}
    questions = {"en": lambda r: r["problem"], lang: lambda r: r["problem_translated"]}

    examples, reject, sw = [], Counter(), Counter()
    no_question = 0
    for rec in trans:
        row = rows.get(rec["question_id"])
        if row is None:            # no usable target-language question (exp2 QC dropped it): skip the problem
            no_question += 1
            continue
        why = translation_verdict(rec, qe and qe[rec["question_id"]], threshold, lang)
        if why:
            reject[why] += 1
        answers = {"en": (rec["reasoning_en"], rec["final_en"])}
        if not why:
            answers[lang] = (rec["reasoning_tgt"], rec["final_tgt"])
        for a_lang, (reasoning, final) in answers.items():
            for q_lang in ("en", lang):
                q, found, swapped, total = code_switch(questions[q_lang](row), dicts[q_lang], swap_p, rng)
                sw.update({f"{q_lang}_found": found, f"{q_lang}_swapped": swapped, f"{q_lang}_words": total})
                sub = f"{'en' if q_lang == 'en' else 'tgt'}_{'en' if a_lang == 'en' else 'tgt'}"
                examples.append({"subset": sub, "question_id": rec["question_id"],
                                 "user": f"{q}\n\n{cfg[q_lang]['student_instruction']}",
                                 "assistant": assistant_turn(cfg[a_lang]["think_prefix"], reasoning, final)})
    if n_parallel:
        for ex in instructions(load_pairs(lang, n_parallel, rng), lang, rng):
            examples.append({"subset": ex["subset"], "question_id": None, "user": ex["user"],
                             "assistant": EMPTY_THINK + ex["answer"]})
    rng.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["id"] = i
    write_jsonl(out_dir / "sft_data.jsonl", examples)

    counts = Counter(e["subset"] for e in examples)
    used = len(trans) - no_question
    stats = {"data": data, "lang": lang, "problems_with_reference": len(trans),
             "problems_without_question": no_question, "problems_used": used, "translations_rejected": dict(reject),
             "translations_accepted": used - sum(reject.values()), "qe_filter": None if qe is None else threshold,
             "examples": len(examples), "by_subset": dict(counts), "swap_p": swap_p, "code_switch": dict(sw),
             "dictionary_size": {k: len(v) for k, v in dicts.items()}, "n_parallel_pairs": n_parallel, "seed": seed}
    write_json(out_dir / "build_stats.json", stats)
    T = [f"SDRRL fine-tuning set: {out_dir.name} (questions: {data})",
         f"  problems with a voted reference: {len(trans)}  (no target-language question, skipped: {no_question})",
         f"  translated references accepted:  {stats['translations_accepted']}  (rejected: {dict(reject) or 'none'})",
         f"  quality filter: " + ("OFF (--no_qe)" if qe is None else f"mean sentence CometKiwi >= {threshold}"),
         f"  examples: {len(examples)}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    for q_lang in ("en", lang):
        w, f, s = sw[f"{q_lang}_words"], sw[f"{q_lang}_found"], sw[f"{q_lang}_swapped"]
        T.append(f"  code-switching, {q_lang} questions: dictionary {len(dicts[q_lang])} words; {f}/{w} = {f / max(w, 1):.1%} of "
                 f"question words are in it; swapped {s} = {s / max(w, 1):.1%} of all words (p = {swap_p} per dictionary word)")
    if qe is not None:
        means = sorted(q["mean"] for q in qe.values() if q.get("mean") is not None)
        T.append(f"  CometKiwi per-problem mean: min {means[0]:.3f}, median {means[len(means) // 2]:.3f}, max {means[-1]:.3f}; "
                 + ", ".join(f">={t}: {sum(m >= t for m in means)}" for t in (0.6, 0.7, 0.8)))
    text = "\n".join(T) + "\n"
    (out_dir / "build_report.txt").write_text(text)
    log.info("\n" + text)
