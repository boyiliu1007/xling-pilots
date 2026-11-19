"""Stage 2: reference-free translation quality (CometKiwi) for every translated sentence.

Runs in its own environment, because `unbabel-comet` pins dependencies that conflict with vLLM. This file imports
only the standard library and `comet`, so it works there without the rest of the package's requirements:

  uv venv /path/to/envs/comet --python 3.11 && uv pip install -p /path/to/envs/comet unbabel-comet
  /path/to/envs/comet/bin/python -m votedistill.sdrrl.quality --out results/sdrrl

The default model, Unbabel/wmt22-cometkiwi-da, is gated: accept its license on Hugging Face and log in
(`hf auth login`) first. CometKiwi reads source and translation jointly with a 512-token limit, so long solutions
are scored sentence by sentence; a problem's score is the mean over its sentences.

Output <out>/qe_scores.jsonl: question_id, mean, min, scores (aligned with `segments` in translations.jsonl).
"""
import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"


def score_pairs(pairs, model_name=DEFAULT_MODEL, batch_size=32, gpus=1):
    """pairs: [(src, mt)] -> {(src, mt): score}."""
    from comet import download_model, load_from_checkpoint
    try:
        ckpt = download_model(model_name)
    except Exception as e:
        raise SystemExit(f"could not download {model_name} ({type(e).__name__}: {e}).\n"
                         "COMET reports every download failure this way. Usual causes: the model is gated (accept the "
                         "license on its Hugging Face page) or anonymous requests are rate-limited on a shared IP.\n"
                         "Both are fixed by logging in: `hf auth login` (or set HF_TOKEN).")
    model = load_from_checkpoint(ckpt)
    uniq = sorted(set(pairs))
    out = model.predict([{"src": s, "mt": t} for s, t in uniq], batch_size=batch_size, gpus=gpus)
    return dict(zip(uniq, (float(x) for x in out.scores)))


def run(out_dir, model_name=DEFAULT_MODEL, batch_size=32, gpus=1):
    out_dir = Path(out_dir)
    recs = [json.loads(l) for l in open(out_dir / "translations.jsonl")]
    scores = score_pairs([(s["src"], s["mt"]) for r in recs for s in r["segments"]], model_name, batch_size, gpus)
    with open(out_dir / "qe_scores.jsonl", "w") as fh:
        for r in recs:
            sc = [scores[(s["src"], s["mt"])] for s in r["segments"]]
            fh.write(json.dumps({"question_id": r["question_id"], "model": model_name,
                                 "mean": sum(sc) / len(sc) if sc else None, "min": min(sc) if sc else None,
                                 "scores": sc}) + "\n")
    means = sorted(sum(v) / len(v) for v in ([scores[(s["src"], s["mt"])] for s in r["segments"]] for r in recs) if v)
    q = lambda p: means[min(len(means) - 1, int(p * len(means)))]
    print(f"{len(recs)} problems, {len(scores)} sentence pairs scored with {model_name}")
    print(f"per-problem mean score: min {means[0]:.3f}  p25 {q(.25):.3f}  median {q(.5):.3f}  p75 {q(.75):.3f}  max {means[-1]:.3f}")
    for t in (0.6, 0.7, 0.75, 0.8, 0.85):
        print(f"  mean >= {t}: {sum(m >= t for m in means)}/{len(means)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory holding translations.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args()
    run(a.out, a.model, a.batch_size, a.gpus)
