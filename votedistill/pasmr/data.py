"""PASMR data: the disjoint SFT / RL split of the training questions and the PAM SFT set.

Split: the QC-passing questions of the Experiment 2 dataset (2,805 for gsm8k-mt3k), shuffled with SPLIT_SEED, first
half SFT, second half RL (the paper splits its data evenly between the two stages).
PAM SFT set: for every SFT-half question with an accepted SDRRL translation of the model's own voted English solution
(CometKiwi >= 0.8, math intact, Swahili; votedistill/sdrrl), one example
  user:      <Swahili question>\\n\\n<Swahili COPSD instruction>          (the evaluation prompt)
  assistant: <think>\\n<Swahili think prefix> <translation_start><English question><translation_end>
             <Swahili reasoning>\\n</think>\\n\\n<Swahili final answer>
The English question is the original GSM8K question (in the paper too the pivot target is the source English
question). Label-free: the Swahili solutions are translations of solutions that agree with the English vote; the paper's
"answer correctness" filter is replaced by that vote agreement. Writes split.json, sft_data.jsonl (the format
sdrrl/train.py reads) and build_report.txt into pasmr.data_dir(data).
"""
import json
import logging
import random
from pathlib import Path

from ..utils import read_jsonl, write_json, write_jsonl
from . import PIVOT_CLOSE, PIVOT_OPEN, data_dir

log = logging.getLogger("votedistill.pasmr.data")
SPLIT_SEED = 1234


def split(qids, seed=SPLIT_SEED):
    """Disjoint halves of the question ids: (sft, rl), each sorted."""
    q = sorted(qids)
    random.Random(seed).shuffle(q)
    h = len(q) // 2
    return sorted(q[:h]), sorted(q[h:])


def load_split(data):
    s = json.loads((data_dir(data) / "split.json").read_text())
    return s["sft"], s["rl"]


def pam_target(think_prefix, q_en, reasoning, final):
    return (f"<think>\n{think_prefix.strip()} {PIVOT_OPEN}{q_en.strip()}{PIVOT_CLOSE}\n{reasoning.strip()}\n</think>"
            f"\n\n{final.strip()}")


def build(data, sdrrl_dir, threshold=0.8, lang="swa"):
    from ..copsd.collator import import_copsd
    from ..sdrrl.build import question_rows, translation_verdict
    import_copsd()
    from language_config import LANGUAGE_CONFIG
    cfg = LANGUAGE_CONFIG[lang.upper()]
    out = data_dir(data)
    out.mkdir(parents=True, exist_ok=True)
    rows = question_rows(data, lang)
    sft, rl = split(rows)
    write_json(out / "split.json", {"seed": SPLIT_SEED, "data": data, "n": len(rows), "sft": sft, "rl": rl})
    sdrrl_dir = Path(sdrrl_dir)
    qe = {r["question_id"]: r for r in read_jsonl(sdrrl_dir / "qe_scores.jsonl")}
    sft_set, examples, reject = set(sft), [], {}
    for rec in read_jsonl(sdrrl_dir / "translations.jsonl"):
        q = rec["question_id"]
        if q not in sft_set:
            continue
        why = "no_qe_score" if q not in qe else translation_verdict(rec, qe[q], threshold, lang)
        if why:
            reject[why] = reject.get(why, 0) + 1
            continue
        row = rows[q]
        examples.append({"id": len(examples), "subset": "pam", "question_id": q,
                         "user": f"{row['problem_translated']}\n\n{cfg['student_instruction']}",
                         "assistant": pam_target(cfg["think_prefix"], row["problem"], rec["reasoning_tgt"],
                                                 rec["final_tgt"])})
    random.Random(SPLIT_SEED).shuffle(examples)
    write_jsonl(out / "sft_data.jsonl", examples)
    report = (f"PASMR data: {data}\n  questions {len(rows)}: SFT half {len(sft)}, RL half {len(rl)} (seed {SPLIT_SEED})\n"
              f"  PAM examples (SFT half with an accepted SDRRL translation): {len(examples)}; rejected {reject}\n"
              f"  translations from {sdrrl_dir}\n")
    (out / "build_report.txt").write_text(report)
    log.info("\n" + report)
    return examples
