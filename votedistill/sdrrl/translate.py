"""Stage 1: translate the voted English references into the target language with NLLB-200.

Output <out>/translations.jsonl, one record per accepted problem:
  question_id, answer, vote_share, reasoning_en, final_en, reasoning_tgt, final_tgt (None if math was lost),
  segments [{"part", "src", "mt"}] (the sentence pairs that quality estimation scores), n_math_lost.
"""
import json
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


def _new_tokens(src_len):
    return int(src_len * 1.5) + 32


class Nllb:
    """batch_tokens bounds (source + generated tokens) x beams per batch, max_batch the sentences per batch: the
    beam-search state (KV cache, the fp32 logits over NLLB's 256k vocabulary) grows with sequences x beams, and short
    sentences generate many more tokens than they read, so a source-token budget alone ran out of memory."""

    def __init__(self, model=DEFAULT_MODEL, src="en", tgt="swa", beams=4, batch_tokens=24000, max_batch=96, max_len=512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, src_lang=NLLB_CODE[src])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=torch.float16).cuda().eval()
        self.bos = self.tok.convert_tokens_to_ids(NLLB_CODE[tgt])
        self.beams, self.batch_tokens, self.max_batch, self.max_len = beams, batch_tokens, max_batch, max_len

    def _generate(self, batch, src_len):
        """Translations of one batch; on CUDA OOM the batch is split in half and retried."""
        try:
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.bos, num_beams=self.beams,
                                          max_new_tokens=_new_tokens(src_len))
            return [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        except self.torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            enc = gen = None
            self.torch.cuda.empty_cache()
            log.warning(f"OOM on a batch of {len(batch)} (source {src_len} tokens): splitting it")
            h = len(batch) // 2
            return self._generate(batch[:h], src_len) + self._generate(batch[h:], src_len)

    def translate(self, sentences, cache=None):
        """-> {sentence: translation}; batches of similar token length, longest first. cache: a JSONL file of
        {"src", "mt"} pairs, appended after every batch and read back on restart, so an interrupted run resumes."""
        out = {}
        if cache is not None and Path(cache).exists():
            with open(cache) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:      # a line cut off by the interruption
                        continue
                    out[r["src"]] = r["mt"]
        todo = [s for s in set(sentences) if s not in out]
        raw = {s: len(self.tok(s).input_ids) for s in todo}
        lens = {s: min(n, self.max_len) for s, n in raw.items()}
        too_long = sum(n > self.max_len for n in raw.values())
        if too_long:
            log.warning(f"{too_long} sentences exceed {self.max_len} tokens and are truncated")
        todo.sort(key=lambda s: (-lens[s], s))
        log.info(f"{len(out)} sentences cached, {len(todo)} to translate")
        fh = open(cache, "a") if cache is not None else None
        t0, done, i, nb = time.time(), 0, 0, 0
        while i < len(todo):
            src_len = lens[todo[i]]                   # the longest in its batch: todo is sorted by token length
            n = max(1, min(self.max_batch, self.batch_tokens // ((src_len + _new_tokens(src_len)) * self.beams)))
            batch = todo[i:i + n]
            for s, t in zip(batch, self._generate(batch, src_len)):
                out[s] = t
                if fh:
                    fh.write(json.dumps({"src": s, "mt": t}) + "\n")
            if fh:
                fh.flush()
            i += n
            done += len(batch)
            nb += 1
            if nb % 50 == 0 or i >= len(todo):
                el = time.time() - t0
                log.info(f"translated {done}/{len(todo)} ({src_len} source tokens, batch {n}); {el / 60:.1f} min, "
                         f"ETA {el / done * (len(todo) - done) / 60:.0f} min")
        if fh:
            fh.close()
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
    mt = Nllb(model, "en", lang).translate(sentences, cache=out_dir / "mt_cache.jsonl")
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
