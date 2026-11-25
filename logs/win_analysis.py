"""D vs SDRRL on existing AfriMGSM Swahili TEST outputs (CPU only): token budgets, cost, language proxies, per-question."""
import json, re, numpy as np
from transformers import AutoTokenizer
from votedistill.exp2.grpo import lid_fraction, LID_MIN, _LID
from votedistill.stats import paired_bootstrap
E = "results/exp2_qwen3-4b"
RUNS = {"base": f"{E}/L0/eval/base_test.jsonl", "base+pen": f"{E}/L0-reppen1.1/eval/base_test.jsonl",
        "D": f"{E}/runs/grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s42/eval/step300_test.jsonl",
        "SDRRL": f"{E}/runs/sdrrl__gsm8k-mt3k__exact__s42/eval/step1108_test.jsonl",
        "SDRRL>D": f"{E}/runs/grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300-initsdrrl__s42/eval/step300_test.jsonl"}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
LOOP = re.compile(r"(.{30,}?)\1{3,}", re.S)
from lingua import LanguageDetectorBuilder
det = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().with_preloaded_language_models().build()
def en_share(text):
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$|\\boxed\{[^}]*\}", " ", text, flags=re.S)
    ch = [c.strip() for c in re.split(r"[\n.!?]+", t)]; ch = [c for c in ch if len(re.findall(r"[^\W\d_]{2,}", c)) >= 3]
    if not ch: return 0.0
    n = [len(c) for c in ch]; return sum(k for k, l in zip(n, det.detect_languages_in_parallel_of(ch)) if l is not None and l.name == "ENGLISH") / sum(n)
D = {}
for k, p in RUNS.items():
    R = [json.loads(l) for l in open(p)]
    out = []
    for r in R:
        for o, c, t in zip(r["outputs"], r["correct"], r["truncated"]):
            lid = lid_fraction(o)
            out.append(dict(item=r["item"], ok=bool(c), trunc=bool(t), n=len(tok(o).input_ids), lid=lid, sw=lid >= LID_MIN,
                            en=en_share(o), loop=bool(t and LOOP.search(o[-3000:]))))
    D[k] = out
    print("done", k, flush=True)
items = sorted({x["item"] for x in D["base"]})
def per_item(k, f):
    return [np.mean([f(x) for x in D[k] if x["item"] == i]) for i in items]
print("\n## 1. Swahili accuracy (correct AND >=70% Swahili) when generation is capped at B tokens")
print(f"{'run':9s} " + " ".join(f"{b:>6d}" for b in (512, 1024, 1536, 2048, 3072, 4096)))
for k in RUNS:
    print(f"{k:9s} " + " ".join(f"{100*np.mean([x['ok'] and x['sw'] and x['n'] <= b for x in D[k]]):6.1f}" for b in (512, 1024, 1536, 2048, 3072, 4096)))
for b in (1024, 1536, 2048):
    a = per_item("SDRRL", lambda x: x["ok"] and x["sw"] and x["n"] <= b); d = per_item("D", lambda x: x["ok"] and x["sw"] and x["n"] <= b)
    print(f"  D - SDRRL at {b} tokens: %+.1f [%+.1f, %+.1f]" % tuple(100*v for v in paired_bootstrap(a, d)))
print("\n## 2. Cost")
for k in RUNS:
    n = [x["n"] for x in D[k]]; c = sum(x["ok"] and x["sw"] for x in D[k])
    print(f"{k:9s} mean tokens/answer {np.mean(n):6.0f}  median {np.median(n):5.0f}  tokens per correct Swahili answer {sum(n)/max(c,1):7.0f}")
print("\n## 3. Language proxies (all outputs)")
for k in RUNS:
    X = D[k]
    print(f"{k:9s} mean Swahili share {np.mean([x['lid'] for x in X]):.2f} | >=90% Swahili {100*np.mean([x['lid']>=0.9 for x in X]):5.1f}% | "
          f"mean English share {np.mean([x['en'] for x in X]):.3f} | any English >=10% {100*np.mean([x['en']>=0.1 for x in X]):5.1f}% | "
          f"looping {100*np.mean([x['loop'] for x in X]):4.1f}% | truncated {100*np.mean([x['trunc'] for x in X]):4.1f}%")
print("\n## 4. Per question (Swahili accuracy), by difficulty for the untrained model (base Swahili any-language accuracy)")
base = dict(zip(items, per_item("base", lambda x: x["ok"])))
bins = [("base 0%", lambda v: v == 0), ("base (0,25%]", lambda v: 0 < v <= 0.25), ("base >25%", lambda v: v > 0.25)]
for name, f in bins:
    sel = [i for i in items if f(base[i])]
    row = " ".join(f"{k} {100*np.mean([x['ok'] and x['sw'] for x in D[k] if x['item'] in sel]):5.1f}" for k in ("D", "SDRRL", "SDRRL>D"))
    print(f"  {name:13s} n={len(sel):3d}: {row}")
d = per_item("D", lambda x: x["ok"] and x["sw"]); s = per_item("SDRRL", lambda x: x["ok"] and x["sw"])
print(f"  questions where D > SDRRL: {sum(a > b for a, b in zip(d, s))}, SDRRL > D: {sum(b > a for a, b in zip(d, s))}, tie: {sum(a == b for a, b in zip(d, s))}")
