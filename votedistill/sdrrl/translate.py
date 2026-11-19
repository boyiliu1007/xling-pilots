"""Stage 1: translate the voted English references into the target language with NLLB-200.

Output <out>/translations.jsonl, one record per accepted problem:
  question_id, answer, vote_share, reasoning_en, final_en, reasoning_tgt, final_tgt (None if math was lost),
  segments [{"part", "src", "mt"}] (the sentence pairs that quality estimation scores), n_math_lost.
"""
import logging
import time
from pathlib import Path

from ..utils import write_json, write_jsonl
from .mathtext import rebuild, split_document
from .references import load_references

log = logging.getLogger("votedistill.sdrrl.translate")

NLLB_CODE = {"en": "eng_Latn", "swa": "swh_Latn", "yor": "yor_Latn"}
# SDRRL uses NLLB-200-3.3B with beam size 4 and otherwise default generation settings.
DEFAULT_MODEL = "facebook/nllb-200-3.3B"
FINAL_TEMPLATE = {"en": "The final answer is $\\boxed{{{a}}}$.", "swa": "Jibu la mwisho ni $\\boxed{{{a}}}$.",
                  "yor": "Ìdáhùn ìkẹyìn ni $\\boxed{{{a}}}$."}


class Nllb:
    def __init__(self, model=DEFAULT_MODEL, src="en", tgt="swa", beams=4, batch_tokens=6000, max_len=512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, src_lang=NLLB_CODE[src])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=torch.float16).cuda().eval()
        self.bos = self.tok.convert_tokens_to_ids(NLLB_CODE[tgt])
        self.beams, self.batch_tokens, self.max_len = beams, batch_tokens, max_len

    def translate(self, sentences):
        """-> {sentence: translation}; batches of similar length, longest first."""
        uniq = sorted(set(sentences), key=len, reverse=True)
        lens = [len(self.tok(s).input_ids) for s in uniq]
        too_long = sum(n > self.max_len for n in lens)
        if too_long:
            log.warning(f"{too_long} sentences exceed {self.max_len} tokens and are truncated")
        out, i = {}, 0
        while i < len(uniq):
            n = max(1, self.batch_tokens // max(min(lens[i], self.max_len), 1))
            batch = uniq[i:i + n]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.bos, num_beams=self.beams,
                                          max_new_tokens=int(min(lens[i], self.max_len) * 1.5) + 32)
            for s, t in zip(batch, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[s] = t.strip()
            i += n
        return out


def run(votes, out_dir, lang="swa", model=DEFAULT_MODEL, min_share=0.625, pick="shortest", limit=None, keep_qids=None):
    """keep_qids: translate only these problems (e.g. the questions of the training set that have a target-language
    question); None = every accepted reference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs, stats = load_references(votes, min_share, pick)
    log.info(f"references from {votes}: {stats}")
    if keep_qids is not None:
        keep_qids = set(keep_qids)
        stats["dropped_no_question"] = sum(q not in keep_qids for q in refs)
        refs = {q: r for q, r in refs.items() if q in keep_qids}
    if not refs:
        raise SystemExit("no usable references in the vote file")
    qids = sorted(refs)[:limit] if limit else sorted(refs)
    docs = {}
    for q in qids:
        r = refs[q]
        r["final"] = r["final"] or FINAL_TEMPLATE["en"].format(a=r["answer"])
        docs[q] = {"reasoning": split_document(r["reasoning"]), "final": split_document(r["final"])}
    sentences = [p.text for d in docs.values() for part in d.values() for p in part if p.kind == "text"]
    log.info(f"{len(qids)} problems, {len(sentences)} sentences ({len(set(sentences))} unique) -> {model}")
    t0 = time.time()
    mt = Nllb(model, "en", lang).translate(sentences)
    dt = time.time() - t0
    records, lost_docs = [], 0
    for q in qids:
        r, d = refs[q], docs[q]
        reasoning, _, lost_r = rebuild(d["reasoning"], mt)
        final, _, lost_f = rebuild(d["final"], mt)
        # The boxed answer is math, so it survives translation verbatim; a lost final falls back to the template.
        if final is None:
            final = FINAL_TEMPLATE[lang].format(a=r["answer"])
        lost_docs += reasoning is None
        records.append({"question_id": q, "answer": r["answer"], "vote_share": r["vote_share"],
                        "reasoning_en": r["reasoning"], "final_en": r["final"],
                        "reasoning_tgt": reasoning, "final_tgt": final, "n_math_lost": lost_r + lost_f,
                        "segments": [{"part": part, "src": p.text, "mt": mt[p.text]}
                                     for part, ps in d.items() for p in ps if p.kind == "text"]})
    write_jsonl(out_dir / "translations.jsonl", records)
    stats.update({"translated_problems": len(records), "sentences": len(sentences), "unique_sentences": len(set(sentences)),
                  "problems_with_lost_math": lost_docs, "translation_s": round(dt, 1), "model": model, "beams": 4,
                  "lang": lang, "min_share": min_share, "pick": pick, "votes": str(votes)})
    write_json(out_dir / "translate_stats.json", stats)
    log.info(f"wrote {out_dir / 'translations.jsonl'}: {stats}")
