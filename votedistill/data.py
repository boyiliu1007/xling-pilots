"""Datasets: AfriMGSM (evaluation), the COPSD AfriMGSM training set, and a GSM8K train sample.

Consensus items are dicts {"qid", "question", "gold", ["source", "gold_consistent"]}. Loaders append a
human-readable inspection of what they loaded to `lines` so every run records exactly which data it used.
"""
import json
import random
import re
from collections import Counter

import numpy as np

from .answers_numeric import numeric_equal, parse_number
from .paths import SPLIT_FILE

AFRIMGSM = "masakhane/afrimgsm"
COPSD_TRAIN = "yihongLiu/COPSD-AfriMGSM-TrainDataset"
MAX_ALIGNMENT_MISMATCH = 0.05


# ----------------------------------------------------------------------------- AfriMGSM

def load_afrimgsm(lang, split="test"):
    """Raw AfriMGSM rows. lang: 'en' or an AfriMGSM config such as 'swa', 'yor'."""
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(AFRIMGSM, "eng" if lang == "en" else lang, split=split)]


def _norm_gold(v):
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    return parse_number(str(v))


def afrimgsm_items(lang, lines):
    rows = load_afrimgsm(lang)
    lines.append(f"\n{'=' * 80}\nAfriMGSM {lang}: {AFRIMGSM} / {'eng' if lang == 'en' else lang} / test ({len(rows)} rows)")
    lines.append(f"columns: {list(rows[0])}; question = 'question', gold = 'answer_number'")
    lines.append("example row 0: " + json.dumps(rows[0], ensure_ascii=False))
    return [{"qid": i, "question": r["question"], "gold": _norm_gold(r["answer_number"])} for i, r in enumerate(rows)]


def _question_numbers(q):
    return Counter(n for n in (parse_number(t) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", q or "")) if n is not None)


def _overlap(a, b):
    a, b = _question_numbers(a), _question_numbers(b)
    tot = sum((a | b).values())
    return sum((a & b).values()) / tot if tot else 1.0


def check_alignment(en_items, items, lang, lines):
    """AfriMGSM has no ID column; verify that row i of `lang` translates English row i.
    Checks gold answers item by item and the numbers in the question text, with an off-by-one control."""
    lines.append(f"\n{'=' * 80}\nALIGNMENT {lang} vs en (row order; no ID column)")
    if len(items) != len(en_items):
        raise SystemExit(f"{lang}: {len(items)} rows vs {len(en_items)} English rows")
    mism = [(a["qid"], a["gold"], e["gold"]) for a, e in zip(items, en_items) if not numeric_equal(a["gold"], e["gold"])]
    ov = np.array([_overlap(a["question"], e["question"]) for a, e in zip(items, en_items)])
    shifted = [_overlap(items[i]["question"], en_items[i + 1]["question"]) for i in range(len(items) - 1)]
    rate = len(mism) / len(items)
    lines.append(f"gold mismatches: {len(mism)}/{len(items)} = {rate:.4f}; first 5: {mism[:5]}")
    lines.append(f"question-number overlap (Jaccard) vs English: mean {ov.mean():.3f}, median {np.median(ov):.3f}, "
                 f"items < 0.5: {(ov < 0.5).sum()}; off-by-one control mean {np.mean(shifted):.3f}")
    ok = rate <= MAX_ALIGNMENT_MISMATCH
    lines.append(f"RESULT: {'OK' if ok else 'FAIL'} (max mismatch {MAX_ALIGNMENT_MISMATCH:.0%})")
    if not ok:
        raise SystemExit(f"AfriMGSM {lang} is not row-aligned with English")


# ----------------------------------------------------------------------------- COPSD training set

def load_copsd_train(lang):
    """Raw rows of the released COPSD training set for one target language (500 OpenThoughts problems).
    Fields: problem (English), solution (English reference), Answer (gold), problem_translated, language, source.
    The English fields are identical across the per-language files, so row index is the question ID."""
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(COPSD_TRAIN, data_files=f"data/{lang}-00000-of-00001.parquet", split="train")]


def copsd_items(lang, lines, base_lang="swa"):
    """lang 'en' -> the English problems (read from the base_lang file); otherwise the translated problems.
    gold_consistent marks items whose 'Answer' parses and matches the last \\boxed{} of the reference solution."""
    from .answers_math import equiv, extract_pred, parse_answer
    rows = load_copsd_train(base_lang if lang == "en" else lang)
    lines.append(f"\n{'=' * 80}\nCOPSD train ({COPSD_TRAIN}) {lang}: {len(rows)} rows; "
                 f"sources {dict(Counter(r['source'] for r in rows))}")
    consistent, disagree = set(), []
    for i, r in enumerate(rows):
        gp = parse_answer(r["Answer"])
        s, p = extract_pred(r["solution"])
        if s is not None and equiv(r["Answer"], gp, s, p) and gp:
            consistent.add(i)
        elif s is not None:
            disagree.append((i, r["Answer"], s))
    lines.append(f"gold 'Answer' parseable and equal to the reference solution's last \\boxed{{}}: {len(consistent)}/{len(rows)}; "
                 f"first 5 disagreements: {disagree[:5]}")
    q = "problem" if lang == "en" else "problem_translated"
    return [{"qid": i, "question": r[q], "gold": r["Answer"], "source": r["source"], "gold_consistent": i in consistent}
            for i, r in enumerate(rows)]


# ----------------------------------------------------------------------------- GSM8K

def gsm8k_items(lines, n=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    lines.append(f"\n{'=' * 80}\nGSM8K train: {n} of {len(ds)} rows via random.Random({seed}).sample, sorted; "
                 f"first 10 indices {idx[:10]}; gold = text after '####', commas removed")
    return [{"qid": int(i), "question": ds[i]["question"],
             "gold": float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))} for i in idx]


# ----------------------------------------------------------------------------- AfriMGSM DEV/TEST split

def make_split(n_items=250, n_dev=50, seed=42):
    idx = list(range(n_items))
    random.Random(seed).shuffle(idx)
    return {"seed": seed, "method": f"random.Random({seed}).shuffle(range({n_items})); DEV = first {n_dev}, TEST = rest; sorted",
            "dev": sorted(idx[:n_dev]), "test": sorted(idx[n_dev:])}


def load_split():
    """The fixed AfriMGSM-swa split shared by every COPSD run (DEV for checkpoint selection, TEST for reporting)."""
    if not SPLIT_FILE.exists():
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(make_split(), indent=1))
    return json.loads(SPLIT_FILE.read_text())
