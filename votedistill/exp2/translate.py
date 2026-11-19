"""Machine translation of the GSM8K training sample into Swahili (the gsm8k-mt training set).

Questions: data.gsm8k_items (GSM8K *train*, 500 rows, seed 42), the same questions the English consensus votes on.
Model: NLLB-200 3.3B (eng_Latn -> swh_Latn), beam search, one sentence at a time (NLLB is trained on sentences and
drops content from long multi-sentence inputs). Quality control, no gold labels involved: the multiset of numbers
in the question must be identical before and after translation, the translation must not loop, and its length
must be plausible. Failing questions keep qc_ok = false and are excluded from training.
"""
import logging
import re
from collections import Counter
from datetime import datetime

from ..data import _question_numbers, gsm8k_items
from ..utils import versions, write_json, write_jsonl
from .trainsets import MT_FILE, MT_FILES

log = logging.getLogger("votedistill.exp2.translate")
MT_MODEL = "facebook/nllb-200-3.3B"
SRC, TGT = "eng_Latn", "swh_Latn"
TGTS = {"swa": "swh_Latn", "zul": "zul_Latn", "yor": "yor_Latn", "sna": "sna_Latn", "sot": "sot_Latn"}
# NLLB spells most numbers out in Yoruba (vigesimal words); there the QC only rejects digit-numbers that are not in
# the English question (PREREGISTRATION 2026-11-18 wave 2).
WORDNUM_LANGS = {"yor"}
GEN = dict(num_beams=4, max_new_tokens=256, no_repeat_ngram_size=0)
LEN_RATIO = (0.7, 2.0)   # Swahili / English characters


def split_sentences(text):
    """Sentences, keeping line breaks as hard boundaries. A boundary is [.?!] + whitespace + an uppercase letter,
    digit, quote or '$', so '3.5', 'Mr. Smith' (lowercase follows rarely) and amounts stay intact in practice."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out += [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\"'$(])", line) if s.strip()]
    return out


SW_NUMBER_WORD = re.compile(r"\b(?:wa|vi|ki|m|ma|mi|ji|u|ku|pa|zi|li|i)?"
                            r"(moja|mbili|wili|tatu|nne|tano|sita|saba|nane|tisa|kumi|nusu|robo|thuluthi)\b", re.I)
WORD_MAX = 12   # numbers NLLB may legitimately spell out in Swahili (tatu = 3, wawili = 2, thuluthi mbili = 2/3)


def qc(en, sw, lang="swa"):
    """(ok, reasons). Numbers: every digit-number of the translation must come from the English question; English
    numbers missing from the translation must be small (<= WORD_MAX, e.g. fraction parts) and the translation must
    contain at least as many Swahili number words. Length ratio in LEN_RATIO; no 4-word loop x3."""
    reasons = []
    ne, ns = _question_numbers(en), _question_numbers(sw)
    extra, missing = ns - ne, ne - ns
    if lang in WORDNUM_LANGS:
        missing = Counter()
    words = len(SW_NUMBER_WORD.findall(sw)) if lang == "swa" else 0   # other languages: every number must stay a digit
    if extra or any(abs(float(v)) > WORD_MAX or float(v) != int(float(v)) for v in missing) \
            or words < sum(missing.values()):
        reasons.append(f"numbers {dict(ne)} -> {dict(ns)} (+{words} number words)")
    ratio = len(sw) / max(len(en), 1)
    if not LEN_RATIO[0] <= ratio <= LEN_RATIO[1]:
        reasons.append(f"length ratio {ratio:.2f}")
    w = sw.split()
    grams = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    if grams and grams.most_common(1)[0][1] >= 3:
        reasons.append("repetition")
    return not reasons, reasons


def requalify(lang="swa"):
    """Re-run QC on the stored translations (no new translation)."""
    from ..utils import read_json, read_jsonl
    from .trainsets import l2_file
    f = MT_FILE if lang == "swa" else l2_file(lang)
    recs = read_jsonl(f)
    for r in recs:
        r["qc_ok"], r["qc_fail"] = qc(r["question_en"], r["question_sw" if lang == "swa" else "question_tgt"], lang)
    write_jsonl(f, recs)
    meta = read_json(f.with_suffix(".meta.json"))
    meta.update({"qc": qc.__doc__, "qc_ok": sum(r["qc_ok"] for r in recs),
                 "requalified": datetime.now().isoformat(timespec="seconds")})
    write_json(f.with_suffix(".meta.json"), meta)
    return recs


def translate(batch_size=32, dataset="gsm8k-mt", lang="swa"):
    """dataset gsm8k-mt: 500 questions -> MT_FILE; gsm8k-mt3k: 3000 questions (same seed) -> MT_FILES['gsm8k-mt3k']."""
    from .trainsets import l2_file
    MT_FILE = MT_FILES[dataset] if lang == "swa" else l2_file(lang)
    tgt = TGTS[lang]
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    lines = []
    items = gsm8k_items(lines, n=3000 if dataset == "gsm8k-mt3k" else 500)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang=SRC)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, dtype=torch.bfloat16).cuda().eval()
    bos = tok.convert_tokens_to_ids(tgt)
    sents = [split_sentences(it["question"]) for it in items]
    flat = [s for ss in sents for s in ss]
    log.info(f"{len(items)} questions, {len(flat)} sentences; {MT_MODEL} {GEN}")
    outs = []
    for i in range(0, len(flat), batch_size):
        enc = tok(flat[i:i + batch_size], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, **GEN)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    recs, k = [], 0
    for it, ss in zip(items, sents):
        sw = " ".join(o.strip() for o in outs[k:k + len(ss)])
        k += len(ss)
        ok, why = qc(it["question"], sw, lang)
        recs.append({"qid": it["qid"], "question_en": it["question"], ("question_sw" if lang == "swa" else "question_tgt"): sw,
                     "lang": lang, "gold": it["gold"],
                     "answer_en": ds[it["qid"]]["answer"], "n_sentences": len(ss), "qc_ok": ok, "qc_fail": why})
    MT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(MT_FILE, recs)
    n_ok = sum(r["qc_ok"] for r in recs)
    write_json(MT_FILE.with_suffix(".meta.json"), {
        "model": MT_MODEL, "src": SRC, "tgt": tgt, "generation": GEN, "sentence_split": split_sentences.__doc__,
        "qc": qc.__doc__, "items": len(recs), "qc_ok": n_ok, "selection": lines, "versions": versions(),
        "created": datetime.now().isoformat(timespec="seconds")})
    log.info(f"QC passed {n_ok}/{len(recs)}; failures: {Counter(w.split()[0] for r in recs for w in r['qc_fail'])}")
    return recs


# ----------------------------------------------------------------------------- self-translation (PREREG 2026-11-18 night)
SELF_PROMPT = ("Translate the following math word problem from English into Swahili. Keep every number exactly as "
               "written in digits. Output only the Swahili translation, nothing else.\n\n{question}")


def self_translate():
    """The policy's own base model (MODEL, e.g. Qwen3-4B) translates the gsm8k-mt3k English questions into Swahili:
    non-thinking mode, greedy, whole question at once. Same QC as NLLB (numbers, length ratio, loops) plus the training
    language check (lid_fraction >= LID_MIN), so no external MT system and no gold label is used. Output
    MT_FILES['gsm8k-st3k'] (model-specific), same schema as the NLLB file."""
    from vllm import LLM, SamplingParams
    from ..paths import MODEL
    from ..utils import read_jsonl
    from .grpo import LID_MIN, lid_fraction
    src = list(read_jsonl(MT_FILES["gsm8k-mt3k"]))   # same 3000 questions, qid, gold, answer_en
    out_f = MT_FILES["gsm8k-st3k"]
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, seed=0, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": SELF_PROMPT.format(question=r["question_en"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in src]
    res = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=768))
    recs = []
    for r, o in zip(src, res):
        sw = o.outputs[0].text.strip()
        ok, why = qc(r["question_en"], sw, "swa")
        lid = lid_fraction(sw, "swa")
        if lid < LID_MIN:
            ok, why = False, why + [f"lid {lid:.2f}"]
        recs.append({"qid": r["qid"], "question_en": r["question_en"], "question_sw": sw, "lang": "swa", "gold": r["gold"],
                     "answer_en": r["answer_en"], "lid": lid, "qc_ok": ok, "qc_fail": why,
                     "nllb_qc_ok": r["qc_ok"]})
    out_f.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_f, recs)
    n_ok = sum(r["qc_ok"] for r in recs)
    write_json(out_f.with_suffix(".meta.json"), {
        "model": MODEL, "prompt": SELF_PROMPT, "decoding": "greedy, non-thinking, max_tokens 768", "qc": qc.__doc__,
        "lid_min": LID_MIN, "items": len(recs), "qc_ok": n_ok, "versions": versions(),
        "created": datetime.now().isoformat(timespec="seconds")})
    log.info(f"self-translation QC passed {n_ok}/{len(recs)} (NLLB passed {sum(r['qc_ok'] for r in src)}); "
             f"failures: {Counter(w.split()[0] for r in recs for w in r['qc_fail'])}")
    return recs
