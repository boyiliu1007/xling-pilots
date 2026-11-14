"""AfriMGSM evaluation of the base model and LoRA checkpoints with COPSD's own prompt, extraction and grader.

Outputs <run_dir>/eval/<name>_<split>.jsonl (one record per item, all n samples). The sampling settings are pinned
in <run_dir>/eval_config.json on first use; later calls with different settings are refused, so every file in a
run directory is comparable.
"""
import logging
import re
import time
from collections import Counter
from pathlib import Path

from ..data import load_afrimgsm, load_split
from ..gpu import wait_for_gpu
from ..paths import MODEL
from ..utils import read_json, read_jsonl, write_json, write_jsonl
from .collator import import_copsd_eval
from .config import SEED

log = logging.getLogger("votedistill.copsd.eval")

SW_WORDS = set("""na ya wa kwa ni la za katika kwamba hii hiyo hizo huo kila kama au lakini sasa hivyo pia ili kwenye yake
wake yao jumla baada kisha je hapa hivi ambayo ambao ambaye kuwa nini zaidi moja mbili tatu nne tano hatua kupata jibu
tunahitaji tuna basi kutoka hadi kati mara ndani hapo kwanza pili tena yote wote kiasi idadi bei siku saa kila""".split())
EN_WORDS = set("""the and of to is we so that in for this it with be are then total each let first find need what how
there which from by on at as if has have was will can but not or our step answer number per now since thus therefore
calculate given""".split())


def language_of(text):
    """'sw' / 'en' / 'other' by function-word counts on the generated text with LaTeX/math removed (Swahili only)."""
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$|\\boxed\{[^}]*\}", " ", text, flags=re.S)
    w = re.findall(r"[a-zA-Z']+", t.lower())
    sw = sum(x in SW_WORDS for x in w)
    en = sum(x in EN_WORDS for x in w)
    if sw == 0 and en == 0:
        return "other"
    return "sw" if sw > en else "en"


def pin_eval_config(run_dir, cfg, ckpt_dir):
    p = Path(run_dir) / "eval_config.json"
    old = read_json(p)
    if old is not None and old["sampling"] != cfg:
        raise SystemExit(f"{p} pins {old['sampling']}, but this call uses {cfg}; use a new --run_dir")
    write_json(p, {"sampling": cfg, "protocol": "COPSD african_langs_eval/evaluate_math.py: build_qwen3_prompt, "
                   "extract_boxed_answer, grade_answer (math-verify); maj = Counter vote over formatted answers",
                   "checkpoint_dir": str(ckpt_dir) if ckpt_dir else None})


def evaluate(run_dir, names, split, cfg, ckpt_dir=None, lang="swa"):
    """names: 'base' and/or 'step<N>' (LoRA adapter at <ckpt_dir>/step-<N>/adapter). Existing outputs are skipped."""
    run_dir = Path(run_dir)
    pin_eval_config(run_dir, cfg, ckpt_dir)
    todo = [n for n in names if not (run_dir / "eval" / f"{n}_{split}.jsonl").exists()]
    for n in set(names) - set(todo):
        log.info(f"[skip] {n}_{split}.jsonl exists")
    if not todo:
        return
    if any(n != "base" for n in todo) and ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required to evaluate checkpoints")
    EM = import_copsd_eval()
    from .styles import apply_style
    apply_style(cfg.get("prompt_style", "copsd"))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    wait_for_gpu()
    rows = load_afrimgsm(lang)
    ids = load_split()[split]
    tok = AutoTokenizer.from_pretrained(MODEL)
    max_len = cfg["max_new_tokens"] + 2048
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_cpu_loras=2, max_model_len=max_len, seed=SEED)
    prompts = [EM.build_qwen3_prompt(tok, rows[i]["question"], lang.upper(), enable_thinking=cfg["enable_thinking"]) for i in ids]
    log.info(f"eval {cfg}, max_model_len {max_len}; prompt example:\n{prompts[0]}")
    (run_dir / "eval").mkdir(exist_ok=True)
    for li, name in enumerate(todo):
        lora = None if name == "base" else LoRARequest(name, li + 1, str(Path(ckpt_dir) / f"step-{int(name[4:])}" / "adapter"))
        sp = SamplingParams(n=cfg["val_n"], temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                            max_tokens=cfg["max_new_tokens"], seed=SEED,
                            repetition_penalty=cfg.get("repetition_penalty", 1.0))
        t0 = time.time()
        res = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        dt = time.time() - t0
        recs = []
        for i, o in zip(ids, res):
            gt = EM.get_afrimgsm_ground_truth(rows[i])
            texts = [c.text for c in o.outputs]
            preds = [EM.extract_boxed_answer(t) for t in texts]
            corr = [EM.grade_answer(p, gt) for p in preds]
            fmt_ = [p is not None for p in preds]
            fp = [p for p, f in zip(preds, fmt_) if f]
            maj = EM.grade_answer(Counter(fp).most_common(1)[0][0], gt) if fp else False   # COPSD majority vote
            recs.append({"model": name, "split": split, "item": i, "gold": gt, "outputs": texts, "preds": preds,
                         "correct": corr, "formatted": fmt_, "truncated": [c.finish_reason == "length" for c in o.outputs],
                         "language": [language_of(t) for t in texts],
                         "answer_language": [language_of(t.split("</think>", 1)[1]) if "</think>" in t else None
                                             for t in texts],
                         "pass_at_n": any(corr), "maj_correct": maj,
                         "mean_acc": sum(corr) / len(corr)})
        write_jsonl(run_dir / "eval" / f"{name}_{split}.jsonl", recs)
        log.info(f"EVAL {name} {split}: {summarize(recs)} ({dt:.0f}s)")


def summarize(recs):
    """pass@n = any sample correct; maj@n = majority answer correct; mean_acc = average per-sample accuracy."""
    n, S = len(recs), sum(len(r["correct"]) for r in recs)
    k = len(recs[0]["correct"])
    return {"n": n, f"pass@{k}": sum(r["pass_at_n"] for r in recs) / n, f"maj@{k}": sum(r["maj_correct"] for r in recs) / n,
            "mean_acc": sum(r["mean_acc"] for r in recs) / n,
            "format_rate": sum(sum(r["formatted"]) for r in recs) / S,
            "trunc_rate": sum(sum(r["truncated"]) for r in recs) / S,
            "sw_rate": sum(sum(l == "sw" for l in r["language"]) for r in recs) / S}


def evaluated(run_dir, split):
    """{name: records} for every model evaluated on `split` in run_dir, base first, then by step."""
    files = sorted(Path(run_dir, "eval").glob(f"*_{split}.jsonl"))
    names = sorted((p.name[: -len(f"_{split}.jsonl")] for p in files), key=lambda n: -1 if n == "base" else int(n[4:]))
    return {n: read_jsonl(Path(run_dir, "eval", f"{n}_{split}.jsonl")) for n in names}


def select_checkpoint(run_dir, metric="pass_at_n", names=None):
    """Checkpoint with the highest DEV score (ties -> earlier step), among `names` or all evaluated checkpoints."""
    dev = {n: R for n, R in evaluated(run_dir, "dev").items() if n != "base"}
    if names:
        if any(n not in dev for n in names):
            return None
        dev = {n: dev[n] for n in names}
    if not dev:
        return None
    score = {n: sum(r[metric] for r in R) / len(R) for n, R in dev.items()}
    return sorted(dev, key=lambda n: (-score[n], int(n[4:])))[0]
