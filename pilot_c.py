"""Pilot C: multilingual MCQ option-order bias and cross-lingual headroom (Qwen3-1.7B, Global-MMLU).

  python pilot_c.py --inspect_only                  # Stage 1 (CPU)
  python pilot_c.py --n_items 20 --output_dir pilot_c_smoke
  python pilot_c.py                                 # full run, 500 items
  python pilot_c.py --report_only                   # rebuild report.txt from JSONL
"""
import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from mcq import LETTERS, item_record, map_back, position_of, renorm, shift_order
from stats import fmt, wilson

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-1.7B"
REPOS = ["CohereLabs/Global-MMLU", "CohereForAI/Global-MMLU"]
FIELDS = {"id": "sample_id", "question": "question", "options": ["option_a", "option_b", "option_c", "option_d"],
          "gold": "answer", "subject": "subject", "cs": "cultural_sensitivity_label"}
INSTR = "Answer with only the letter of the correct option (A, B, C, or D)."
TEACHER_LINE = "The following English text is a translation of the same question and options:"
GPU_LIMIT_MIB = 4096
POLL_S = 300
B = 10000
log = logging.getLogger("pilot_c")


# ----------------------------------------------------------------------------- Stage 1

def stage1(args):
    from datasets import get_dataset_config_names, load_dataset
    L = [f"Global-MMLU inspection  {datetime.now().isoformat(timespec='seconds')}"]
    repo, cfgs = None, None
    for r in REPOS:
        try:
            cfgs = get_dataset_config_names(r)
            repo = r
            break
        except Exception as e:
            L.append(f"{r}: unavailable ({type(e).__name__}: {e})")
    if repo is None:
        raise SystemExit("No Global-MMLU repo available")
    L.append(f"repo: {repo}\nconfigs ({len(cfgs)}): {cfgs}")
    langs = ["en"] + args.langs
    data = {}
    for lang in langs:
        if lang not in cfgs:
            raise SystemExit(f"{lang} not in configs")
        ds = load_dataset(repo, lang, split="test")
        cols = ds.column_names
        need = [FIELDS["id"], FIELDS["question"], *FIELDS["options"], FIELDS["gold"], FIELDS["subject"]]
        miss = [c for c in need if c not in cols]
        L.append(f"\n{'=' * 80}\nLANGUAGE {lang}: {repo}/{lang}/test  ({len(ds)} rows)\ncolumns: {cols}\nfeatures: {ds.features}")
        if miss:
            raise SystemExit(f"{lang}: missing columns {miss}")
        L.append(f"fields used: question={FIELDS['question']!r}, options={FIELDS['options']}, gold={FIELDS['gold']!r}, "
                 f"id={FIELDS['id']!r}, subject={FIELDS['subject']!r}, cultural label="
                 f"{FIELDS['cs'] if FIELDS['cs'] in cols else None!r}")
        rows = [dict(r) for r in ds]
        L.append("one full row (row 0):\n" + json.dumps(rows[0], ensure_ascii=False, indent=2))
        L.append(f"gold value counts: {dict(Counter(r[FIELDS['gold']] for r in rows))}")
        if FIELDS["cs"] in cols:
            L.append(f"{FIELDS['cs']} counts: {dict(Counter(r[FIELDS['cs']] for r in rows))}")
        ids = [r[FIELDS["id"]] for r in rows]
        L.append(f"unique sample_ids: {len(set(ids))} / {len(ids)}")
        data[lang] = {r[FIELDS["id"]]: r for r in rows}

    L.append(f"\n{'=' * 80}\nALIGNMENT by sample_id")
    common = set(data["en"])
    for lang in args.langs:
        common &= set(data[lang])
        shared = set(data["en"]) & set(data[lang])
        mism = [i for i in shared if data[lang][i][FIELDS["gold"]].strip() != data["en"][i][FIELDS["gold"]].strip()]
        rate = len(mism) / len(shared)
        L.append(f"{lang}: {len(shared)} ids shared with en (en has {len(data['en'])}, {lang} has {len(data[lang])}); "
                 f"gold mismatches {len(mism)}/{len(shared)} = {rate:.4f}; first 5: {sorted(mism)[:5]}")
        # cultural label consistency across languages
        if FIELDS["cs"] in data["en"][next(iter(shared))]:
            cm = sum(data[lang][i][FIELDS["cs"]] != data["en"][i][FIELDS["cs"]] for i in shared)
            L.append(f"   cultural label differs from en for {cm} ids")
        if rate > 0.05:
            (Path(args.output_dir) / "dataset_inspection.txt").write_text("\n".join(L) + "\n")
            raise SystemExit(f"{lang}: gold mismatch {rate:.2%} > 5%; stopping")
    L.append(f"ids present in all of {langs}: {len(common)}")

    # stratified sample by subject: proportional allocation (largest remainder), fixed seed
    rng = random.Random(args.seed)
    by_subj = defaultdict(list)
    for i in sorted(common):
        by_subj[data["en"][i][FIELDS["subject"]]].append(i)
    tot = len(common)
    quota = {s: args.n_items * len(v) / tot for s, v in by_subj.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: (-(quota[s] - alloc[s]), s))[: args.n_items - sum(alloc.values())]:
        alloc[s] += 1
    ids = []
    for s in sorted(by_subj):
        ids += rng.sample(by_subj[s], min(alloc[s], len(by_subj[s])))
    rng.shuffle(ids)
    L.append(f"sampled {len(ids)} ids, stratified by subject (proportional, largest remainder, seed {args.seed}); "
             f"{sum(1 for s in alloc if alloc[s] > 0)} subjects represented; allocation: {dict(sorted(alloc.items()))}")
    L.append(f"sample gold distribution (en): {dict(Counter(data['en'][i][FIELDS['gold']] for i in ids))}")
    if FIELDS["cs"] in data["en"][ids[0]]:
        L.append(f"sample cultural label distribution (en): {dict(Counter(data['en'][i][FIELDS['cs']] for i in ids))}")
    out = Path(args.output_dir)
    (out / "dataset_inspection.txt").write_text("\n".join(L) + "\n")
    (out / "item_ids.json").write_text(json.dumps(ids, indent=1))
    print("\n".join(L))
    return repo, {lang: [data[lang][i] for i in ids] for lang in langs}


# ----------------------------------------------------------------------------- prompts

def fmt_block(q, opts, s):
    lines = [q.strip(), ""]
    for p, o in enumerate(shift_order(s)):
        lines.append(f"{LETTERS[p]}. {str(opts[o]).strip()}")
    return "\n".join(lines)


def user_text(view, row, en_row, s):
    o = [row[f] for f in FIELDS["options"]]
    if view == "teacher":
        eo = [en_row[f] for f in FIELDS["options"]]
        return (fmt_block(row[FIELDS["question"]], o, s) + f"\n\n{TEACHER_LINE}\n\n"
                + fmt_block(en_row[FIELDS["question"]], eo, s) + f"\n\n{INSTR}")
    return fmt_block(row[FIELDS["question"]], o, s) + f"\n\n{INSTR}"


def views(langs):
    """(view_name, data_lang, kind)."""
    v = [("en", "en", "plain")]
    v += [(l, l, "plain") for l in langs]
    v += [(f"teacher_{l}", l, "teacher") for l in langs]
    return v


# ----------------------------------------------------------------------------- GPU

def other_gpu_mib():
    me = os.getpid()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    procs = [(int(a.split(",")[0]), int(a.split(",")[1])) for a in apps if a.strip()]
    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0])
    others = sum(m for p, m in procs if p != me)
    return others, used, procs


def wait_for_gpu():
    while True:
        others, used, procs = other_gpu_mib()
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        busy = others > GPU_LIMIT_MIB or (not procs and used > GPU_LIMIT_MIB)
        log.info(f"GPU check: other processes {others} MiB, total used {used} MiB, procs {procs} -> "
                 f"{'BUSY, waiting ' + str(POLL_S) + 's' if busy else 'free'}")
        if not busy:
            log.info("nvidia-smi at start:\n" + smi)
            return
        time.sleep(POLL_S)


class GpuPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.ev = 0, threading.Event()

    def run(self):
        while not self.ev.is_set():
            try:
                self.peak = max(self.peak, other_gpu_mib()[1])
            except Exception:
                pass
            self.ev.wait(2)


def letter_token_ids(tok):
    ids = {}
    for c in LETTERS:
        cand = []
        for v in (c, " " + c):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1 and t[0] not in cand:
                cand.append(t[0])
        ids[c] = cand
    return ids


def score(args, items, vlist):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    lid = letter_token_ids(tok)
    log.info("letter token ids: " + json.dumps({c: [(i, tok.decode([i])) for i in v] for c, v in lid.items()}))
    reqs = []  # (view, item_idx, shift, prompt_ids)
    for vname, dl, kind in vlist:
        for k in range(len(items["en"])):
            for s in range(4):
                txt = user_text(kind, items[dl][k], items["en"][k], s)
                p = tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) + args.prefill
                if k == 0 and s == 0:
                    log.info(f"rendered prompt [{vname}] item 0 shift 0:\n{p!r}")
                reqs.append((vname, k, s, tok(p, add_special_tokens=False)["input_ids"]))
    order = sorted(range(len(reqs)), key=lambda i: len(reqs[i][3]))
    res = {}
    t0 = time.time()
    i = 0
    flat = [t for c in LETTERS for t in lid[c]]
    with torch.inference_mode():
        while i < len(order):
            maxlen = len(reqs[order[i]][3])
            j = i
            while j < len(order) and (j - i + 1) * len(reqs[order[j]][3]) <= args.token_budget:
                j += 1
            j = max(j, i + 1)
            batch = [reqs[o] for o in order[i:j]]
            Lm = max(len(b[3]) for b in batch)
            ids = torch.full((len(batch), Lm), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(batch), Lm), dtype=torch.long)
            for r, b in enumerate(batch):  # right padding: positions stay exact
                ids[r, : len(b[3])] = torch.tensor(b[3])
                am[r, : len(b[3])] = 1
            ids, am = ids.cuda(), am.cuda()
            h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
            last = am.sum(1) - 1
            logits = model.lm_head(h[torch.arange(len(batch), device=h.device), last]).float()
            pr = torch.softmax(logits, -1)[:, flat].cpu().numpy()
            for r, b in enumerate(batch):
                raw, c = [], 0
                for ch in LETTERS:
                    raw.append(float(pr[r, c: c + len(lid[ch])].sum()))
                    c += len(lid[ch])
                res[(b[0], b[1], b[2])] = raw
            i = j
    dt = time.time() - t0
    log.info(f"scored {len(reqs)} prompts in {dt:.1f}s; torch peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    return res, lid, dt, len(reqs), torch.cuda.max_memory_allocated() / 2**20


def build_records(items, vlist, res, out):
    recs = {}
    for vname, dl, kind in vlist:
        rows = []
        for k in range(len(items["en"])):
            r = items[dl][k]
            gold = LETTERS.index(r[FIELDS["gold"]].strip())
            mapped, pos, raws, masses = [], [], [], []
            for s in range(4):
                raw = res[(vname, k, s)]
                p, m = renorm(raw)
                mapped.append(map_back(p, s))
                pos.append(int(np.argmax(p)))
                raws.append(raw)
                masses.append(m)
            rec = {"sample_id": r[FIELDS["id"]], "view": vname, "lang": dl, "subject": r[FIELDS["subject"]],
                   "cultural_label": items["en"][k].get(FIELDS["cs"]),
                   "raw_letter_probs_by_position": raws, "letter_mass": masses,
                   "gold_position_by_shift": [position_of(gold, s) for s in range(4)]}
            rec.update(item_record(mapped, pos, gold))
            rows.append(rec)
        with open(out / f"{vname}.jsonl", "w") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        recs[vname] = rows
    return recs


def show_items(items, vlist, recs, n=3):
    L = ["Permutation check: first 3 items, all 4 shifts (EN view and first target view)"]
    for vname in [vlist[0][0], vlist[1][0]]:
        dl = next(d for v, d, _ in vlist if v == vname)
        for k in range(n):
            r = recs[vname][k]
            row = items[dl][k]
            L.append(f"\n[{vname}] item {k} {r['sample_id']} gold original option {LETTERS[r['gold']]}")
            for s in range(4):
                shown = " | ".join(f"{LETTERS[p]}=orig{LETTERS[o]}:{str(row[FIELDS['options'][o]])[:25]!r}"
                                   for p, o in enumerate(shift_order(s)))
                L.append(f"  shift {s}: {shown}")
                L.append(f"     raw letter probs by position {np.round(r['raw_letter_probs_by_position'][s], 3).tolist()} "
                         f"mass {r['letter_mass'][s]:.3f} -> mapped {np.round(r['probs_mapped'][s], 3).tolist()} "
                         f"picked position {LETTERS[r['chosen_position'][s]]} = orig {LETTERS[r['argmax_mapped'][s]]}; "
                         f"gold at position {LETTERS[r['gold_position_by_shift'][s]]}")
            L.append(f"  debiased {LETTERS[r['debiased_answer']]} correct={r['debiased_correct']} "
                     f"order_sensitive={r['order_sensitive']} vote={r['vote_answer']}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- analysis

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def boot(stat, arrays, seed, b=B):
    """Paired bootstrap over items. stat(dict of mean-arrays) -> array of values per resample (NaN allowed)."""
    n = len(next(iter(arrays.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    point = stat({k: np.array([v.mean()]) for k, v in arrays.items()})[0]
    vals = stat({k: v[idx].mean(1) for k, v in arrays.items()})
    ok = vals[~np.isnan(vals)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return point, lo, hi, len(ok)


def bfmt(point, lo, hi, nvalid=None):
    if point is None or np.isnan(point):
        return "N/A"
    s = f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    if nvalid is not None and nvalid < B:
        s += f" ({nvalid}/{B} resamples defined)"
    return s


def arr(recs, key):
    return np.array([float(r[key]) for r in recs])


def build_report(args, cfg=None):
    out = Path(args.output_dir)
    vlist = views(args.langs)
    R = {v: read_jsonl(out / f"{v}.jsonl") for v, _, _ in vlist}
    N = len(R["en"])
    for v in R:
        assert [r["sample_id"] for r in R[v]] == [r["sample_id"] for r in R["en"]]
    T = ["PILOT C — Multilingual MCQ option-order bias and cross-lingual headroom (Qwen3-1.7B, Global-MMLU)",
         f"generated {datetime.now().isoformat(timespec='seconds')};  N = {N} items (same IDs in every view)",
         "Intervals: Wilson 95% for proportions; paired bootstrap (10,000 resamples over items, percentile) for differences."]
    insp = out / "dataset_inspection.txt"
    if insp.exists():
        T.append("\n## Dataset and alignment")
        for l in insp.read_text().splitlines():
            if l.startswith(("repo:", "LANGUAGE", "fields used", "sw:", "yo:", "hi:", "   cultural", "ids present",
                             "sample gold", "sample cultural")) or (l.startswith("sampled") and True):
                T.append("  " + (l if len(l) < 400 else l[:400] + " ..."))
    if cfg:
        T.append("\n## Config")
        for k in ("model", "dtype", "backend", "chat_template", "instruction", "teacher_line", "assistant_prefill", "letter_token_ids",
                  "versions", "args"):
            T.append(f"  {k}: {json.dumps(cfg.get(k), ensure_ascii=False)}")

    T.append("\n## Per-view metrics")
    T.append("  (mean-shift accuracy and picked-letter shares use N*4 decisions; Wilson there treats shifts as independent)")
    for v in R:
        rs = R[v]
        ms = np.array([m for r in rs for m in r["letter_mass"]])
        picks = Counter(p for r in rs for p in r["chosen_position"])
        T.append(f" {v}")
        T.append(f"   letter_mass mean {ms.mean():.3f} (min {ms.min():.3f}, share < 0.5: {(ms < 0.5).mean():.3f})")
        T.append(f"   single-order acc:  " + fmt(int(arr(rs, 'single_order_correct').sum()), N))
        T.append(f"   mean-shift acc:    " + fmt(int(round(arr(rs, 'mean_shift_accuracy').sum() * 4)), 4 * N))
        T.append(f"   debiased acc:      " + fmt(int(arr(rs, 'debiased_correct').sum()), N))
        T.append(f"   vote acc:          " + fmt(int(arr(rs, 'vote_correct').sum()), N)
                 + f"   (vote ties: {sum(r['vote_answer'] is None for r in rs)})")
        T.append(f"   order-sensitive:   " + fmt(int(arr(rs, 'order_sensitive').sum()), N))
        T.append("   picked letter:     " + "  ".join(f"{LETTERS[p]} {fmt(picks[p], 4 * N)}" for p in range(4))
                 + "   (ideal 0.25 each; chance acc 0.25)")

    T.append("\n## H1: order-sensitivity rate, L view minus EN view (paired bootstrap)")
    en_os = arr(R["en"], "order_sensitive")
    for l in args.langs:
        p = boot(lambda d: d["l"] - d["e"], {"l": arr(R[l], "order_sensitive"), "e": en_os}, args.seed)
        T.append(f"  {l}: {arr(R[l], 'order_sensitive').mean():.3f} - {en_os.mean():.3f} = {bfmt(*p)}")

    T.append("\n## H2: accuracy gap decomposition (paired bootstrap)")
    dec = {}
    es, ed = arr(R["en"], "single_order_correct"), arr(R["en"], "debiased_correct")
    for l in args.langs:
        ls, ld = arr(R[l], "single_order_correct"), arr(R[l], "debiased_correct")
        d = {"es": es, "ed": ed, "ls": ls, "ld": ld}
        gs = boot(lambda x: x["es"] - x["ls"], d, args.seed)
        gd = boot(lambda x: x["ed"] - x["ld"], d, args.seed)
        gain = boot(lambda x: x["ld"] - x["ls"], d, args.seed)
        engain = boot(lambda x: x["ed"] - x["es"], d, args.seed)

        def share(x):
            g = x["es"] - x["ls"]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(g > 0, (g - (x["ed"] - x["ld"])) / g, np.nan)
        sh = boot(share, d, args.seed)
        T.append(f"  {l}: gap_single   = {bfmt(*gs)}")
        T.append(f"      gap_debiased = {bfmt(*gd)}")
        T.append(f"      debias_gain_{l} = acc_deb({l}) - acc_single({l}) = {bfmt(*gain)}")
        T.append(f"      (debias_gain_en = {bfmt(*engain)})")
        T.append(f"      share of single-order gap closed by debiasing = "
                 + ("N/A (gap_single <= 0)" if gs[0] <= 0 else bfmt(*sh)))
        dec[l] = {"gain": gain}

    T.append("\n## H3: TEACHER view vs EN and vs L (paired bootstrap)")
    for l in args.langs:
        tv = f"teacher_{l}"
        for key, name in (("order_sensitive", "order-sensitivity"), ("debiased_correct", "debiased acc")):
            t, e, x = arr(R[tv], key), arr(R["en"], key), arr(R[l], key)
            a = boot(lambda d: d["t"] - d["e"], {"t": t, "e": e}, args.seed)
            b = boot(lambda d: d["t"] - d["x"], {"t": t, "x": x}, args.seed)
            T.append(f"  {l} {name}: TEACHER {t.mean():.3f}, EN {e.mean():.3f}, {l} {x.mean():.3f};  "
                     f"TEACHER-EN {bfmt(*a)};  TEACHER-{l} {bfmt(*b)}")

    T.append("\n## Headroom (debiased answers)")
    for l in args.langs:
        tv = f"teacher_{l}"
        lw = [k for k in range(N) if not R[l][k]["debiased_correct"]]
        enr = sum(R["en"][k]["debiased_correct"] for k in lw)
        ter = sum(R[tv][k]["debiased_correct"] for k in lw)
        T.append(f"  {l}: L debiased wrong = {len(lw)}/{N}")
        T.append(f"     P(EN deb correct | L deb wrong):      " + fmt(enr, len(lw)))
        T.append(f"     EN deb correct & L deb wrong / all:   " + fmt(enr, N))
        T.append(f"     P(TEACHER deb correct | L deb wrong): " + fmt(ter, len(lw)))
        T.append(f"     overall EN debiased acc for reference: " + fmt(int(ed.sum()), N))
        dec[l]["head"] = (enr, N)
        dec[l]["accd"] = (int(arr(R[l], "debiased_correct").sum()), N)

    labels = sorted({r["cultural_label"] for r in R["en"] if r["cultural_label"] is not None})
    if labels:
        T.append("\n## Breakdown by cultural_sensitivity_label (debiased acc; headroom = EN correct & L wrong)")
        for lab in labels:
            ks = [k for k in range(N) if R["en"][k]["cultural_label"] == lab]
            line = f"  {lab!r:>5} n={len(ks):>3}: EN {fmt(sum(R['en'][k]['debiased_correct'] for k in ks), len(ks))}"
            for l in args.langs:
                line += (f" | {l} {fmt(sum(R[l][k]['debiased_correct'] for k in ks), len(ks))}"
                         f" headroom {fmt(sum(R['en'][k]['debiased_correct'] and not R[l][k]['debiased_correct'] for k in ks), len(ks))}")
            T.append(line)

    T.append("\n## Subjects by headroom share (subjects with >= 15 items)")
    subj = defaultdict(list)
    for k in range(N):
        subj[R["en"][k]["subject"]].append(k)
    big = {s: ks for s, ks in subj.items() if len(ks) >= 15}
    T.append(f"  {len(big)} subjects qualify: " + ", ".join(f"{s}({len(k)})" for s, k in sorted(big.items())))
    for l in args.langs:
        hs = sorted(((sum(R["en"][k]["debiased_correct"] and not R[l][k]["debiased_correct"] for k in ks), len(ks), s)
                     for s, ks in big.items()), key=lambda x: -x[0] / x[1])
        T.append(f"  {l} top-5:    " + "; ".join(f"{s} {fmt(h, n)}" for h, n, s in hs[:5]))
        T.append(f"  {l} bottom-5: " + "; ".join(f"{s} {fmt(h, n)}" for h, n, s in hs[-5:][::-1]))

    T.append("\n## Pre-registered decision: PASS iff (a) Wilson LB acc_debiased(L) >= 0.30, "
             "(b) share(EN deb correct & L deb wrong) >= 0.10, (c) debias_gain_L > 0 with bootstrap CI excluding 0")
    verdict = {}
    for l in args.langs:
        k, n = dec[l]["accd"]
        lb = wilson(k, n)[0]
        h = dec[l]["head"][0] / dec[l]["head"][1]
        g, glo, ghi, _ = dec[l]["gain"]
        a, b_, c = lb >= 0.30, h >= 0.10, g > 0 and glo > 0
        verdict[l] = a and b_ and c
        T.append(f"  {l}: {'PASS' if verdict[l] else 'FAIL'}  (a) {fmt(k, n)} LB {lb:.3f} -> {a};  "
                 f"(b) {h:.3f} -> {b_};  (c) {bfmt(g, glo, ghi)} -> {c}")

    if cfg and cfg.get("runtime"):
        T.append("\n## Runtime and GPU (observed)")
        for k, v in cfg["runtime"].items():
            T.append(f"  {k}: {v}")
    T.append("\n## Limitations")
    T += ["  - One model, one prompt format, English instruction for all views; next-token letter probabilities only (no reasoning).",
          "  - Only the 4 cyclic shifts, not all 24 permutations; debiasing = averaging over these 4.",
          "  - Global-MMLU sw/yo/hi are translations of MMLU; EN items may be contaminated in pretraining.",
          "  - Headroom counts items where an English-derived label is right; it does not show that distillation would transfer.",
          "  - The TEACHER view contains the English text, so its accuracy is not attainable by an L-only student at test time.",
          f"  - N = {N}; subject and cultural-label breakdowns are small and noisy."]
    text = "\n".join(T) + "\n"
    (out / "report.txt").write_text(text)
    return text, verdict


# ----------------------------------------------------------------------------- main

def versions():
    v = {"python": platform.python_version()}
    for m in ("torch", "transformers", "datasets", "numpy"):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:
            v[m] = f"unavailable ({type(e).__name__})"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["sw", "yo", "hi"])
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="pilot_c_out")
    ap.add_argument("--token_budget", type=int, default=24000, help="max padded tokens per forward batch")
    # Chosen on smoke items by letter_mass only: the model's own greedy prefix is "The correct answer is: **B. ...";
    # "Answer:" / "The correct answer is: " leave most mass on " **" (see run.log of pilot_c_smoke_*).
    ap.add_argument("--prefill", default="The correct answer is: **",
                    help="text appended after the generation prompt (assistant-turn prefix); the next token is scored")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    args.langs = [l for l in args.langs if l != "en"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])
    for name in ("httpx", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.report_only:
        cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else None
        print(build_report(args, cfg)[0])
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    log.info(f"=== pilot_c start; argv {sys.argv}; args {vars(args)}")
    repo, items = stage1(args)
    log.info("Stage 1 passed")
    if args.inspect_only:
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_mcq.py")], capture_output=True, text=True, cwd=HERE)
    log.info("Stage 2 unit tests:\n" + r.stdout[-1500:])
    if r.returncode != 0:
        raise SystemExit("unit tests failed")

    wait_for_gpu()
    vlist = views(args.langs)
    mon = GpuPeak()
    mon.start()
    start = datetime.now()
    t0 = time.time()
    res, lid, dt, nreq, torch_peak = score(args, items, vlist)
    wall = time.time() - t0
    mon.ev.set()
    recs = build_records(items, vlist, res, out)
    log.info(show_items(items, vlist, recs))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = {"model": MODEL, "dtype": "bfloat16", "backend": "HF transformers forward pass (right padding, last real token)",
           "chat_template": "Qwen3 apply_chat_template(enable_thinking=False, add_generation_prompt=True)",
           "instruction": INSTR, "teacher_line": TEACHER_LINE, "dataset": repo, "assistant_prefill": args.prefill,
           "letter_token_ids": {c: [[i, tok.decode([i])] for i in v] for c, v in lid.items()},
           "versions": versions(), "args": vars(args), "argv": sys.argv, "hostname": platform.node(),
           "runtime": {"start": start.isoformat(timespec="seconds"), "end": datetime.now().isoformat(timespec="seconds"),
                       "wall_clock_s (model load + scoring)": round(wall, 1), "scoring_s": round(dt, 1),
                       "prompts_scored": nreq, "torch_peak_allocated_MiB": round(torch_peak),
                       "peak_gpu_mem_used_MiB (nvidia-smi poll, whole GPU)": mon.peak}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    masses = {v: float(np.mean([m for r in rs for m in r["letter_mass"]])) for v, rs in recs.items()}
    log.info(f"mean letter_mass per view: {json.dumps(masses)}")
    if min(masses.values()) < 0.5:
        raise SystemExit(f"letter_mass check FAILED (<0.5): {masses}; debug the prompt before the full run")
    text, verdict = build_report(args, cfg)
    print(text)
    log.info(f"verdict {verdict}")


if __name__ == "__main__":
    main()
