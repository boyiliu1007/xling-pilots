"""Vote-filtered self-SFT: a deliberate, label-free alternative to the clipped COPSD stage 1 (arm D).

Data (generated once, shared by all seeds): for every vote-accepted training question of the dataset (trainsets.py,
L3-vote rows), the base model samples GEN_N answers to the *student's* Swahili prompt with the COPSD training sampling
(T 1.1, top-p 0.95, top-k 20, 2048 tokens). An answer is kept when it finished, has a final answer after </think>,
and that answer is equivalent to the English majority vote (no gold). At most PER_Q answers per question are kept.

Training: the COPSD student LoRA, optimizer, LR schedule (100-step horizon), gradient clip, batch (32 = 4 x 8) and
token-normalised loss of the COPSD trainer, with the distillation loss replaced by next-token NLL on the kept answers.
SFT_STEPS optimizer steps; checkpoints every 10 in the trainer's format, so a stage-2 run can continue from step 30
exactly like from a clipped COPSD stage 1 (runs.INITS["s1sft30"]).
"""
import fcntl
import logging
import random
import time
from datetime import datetime

import numpy as np

from ..answers_math import equiv, extract_pred, parse_answer
from ..copsd.collator import make_collator
from ..copsd.config import OFFICIAL, SAVE_EVERY
from ..paths import MODEL
from ..utils import read_jsonl, versions, write_json, write_jsonl
from .levels import EXP2_RESULTS
from .screen import _prompt_ids
from .trainsets import training_rows

log = logging.getLogger("votedistill.exp2.selfsft")
GEN_N = 8
PER_Q = 2
GEN_SEED = 42
SFT_DIR = EXP2_RESULTS / "selfsft"


def gen_file(data):
    return SFT_DIR / f"gen_{data}.jsonl"


def generate(data):
    """Sample, filter by agreement with the English vote, write gen_<data>.jsonl (all samples, with a `keep` flag)."""
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SFT_DIR / f".lock_{data}", "w") as lk:      # several seeds may start at once: generate only once
        fcntl.flock(lk, fcntl.LOCK_EX)
        if gen_file(data).exists():
            return read_jsonl(gen_file(data))
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        rows, desc = training_rows(data, "L3-vote")
        tok = AutoTokenizer.from_pretrained(MODEL)
        col = make_collator(tok, OFFICIAL["max_length"], False, False)
        sp_ids = [_prompt_ids(col, r, "student") for r in rows]
        llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=8192, seed=GEN_SEED)
        sps = [SamplingParams(n=GEN_N, temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"],
                              top_k=OFFICIAL["top_k"], min_p=0.0, max_tokens=OFFICIAL["max_completion_length"],
                              seed=GEN_SEED * 1_000_003 + i) for i in range(len(rows))]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in sp_ids], sps, use_tqdm=False)
        recs = []
        for r, p, o in zip(rows, sp_ids, outs):
            m = r["vote_answer"]
            kept = 0
            for j, c in enumerate(o.outputs):
                s, pp = extract_pred(c.text, thinking=True)
                ok = c.finish_reason == "stop" and s is not None and equiv(m, parse_answer(m), s, pp)
                keep = ok and kept < PER_Q
                kept += keep
                recs.append({"dataset": r["dataset"], "qid": r["qid"], "sample": j, "student_prompt_ids": p,
                             "completion_ids": list(c.token_ids), "text": c.text, "finish_reason": c.finish_reason,
                             "pred": s, "vote_answer": m, "agrees": bool(ok), "keep": bool(keep)})
        write_jsonl(gen_file(data), recs)
        n_keep = sum(r["keep"] for r in recs)
        write_json(gen_file(data).with_suffix(".meta.json"), {
            "doc": __doc__, "rows": desc, "gen_n": GEN_N, "per_question_cap": PER_Q, "seed": GEN_SEED,
            "samples": len(recs), "finished": sum(r["finish_reason"] == "stop" for r in recs),
            "agree_with_vote": sum(r["agrees"] for r in recs), "kept": n_keep,
            "questions_with_kept": len({r["qid"] for r in recs if r["keep"]}), "generation_s": round(time.time() - t0),
            "versions": versions(), "created": datetime.now().isoformat(timespec="seconds")})
        log.info(f"self-SFT data: {len(recs)} samples, kept {n_keep} "
                 f"({len({r['qid'] for r in recs if r['keep']})} questions) in {time.time() - t0:.0f}s")
        import gc
        import torch
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return recs


def train_sft(name, data, layout, seed):
    import shutil
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
    from ..copsd.config import GA, MICRO_BS
    from ..copsd.train import LOSS_CSV_HEADER, _load_rng, _save_rng, latest_ckpt
    from .runs import SFT_STEPS, paths
    if layout != "exact":
        raise SystemExit("self-SFT is defined for the exact layout only")
    rd, ck = paths(name)
    rd.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    recs = [r for r in generate(data) if r["keep"]]
    if len(recs) < MICRO_BS * GA:
        raise SystemExit(f"only {len(recs)} vote-consistent answers: too few for self-SFT")
    start = latest_ckpt(ck) or 0
    if start >= SFT_STEPS:
        log.info("self-SFT already complete")
        return
    write_json(rd / "config.json", {"model": MODEL, "method": "vote-filtered self-SFT (selfsft.py)", "doc": __doc__,
                                    "examples": len(recs), "questions": len({r["qid"] for r in recs}),
                                    "data": str(gen_file(data)), "steps": SFT_STEPS, "seed": seed, "layout": layout,
                                    "official_optimizer": {k: OFFICIAL[k] for k in ("learning_rate", "max_grad_norm",
                                                           "lora_r", "lora_alpha", "max_steps")},
                                    "batch": f"{MICRO_BS} x {GA}", "versions": versions()})
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                                             lora_dropout=OFFICIAL["lora_dropout_configured"],
                                             target_modules=OFFICIAL["lora_target_modules"]))
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OFFICIAL["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, 0, OFFICIAL["max_steps"])
    loss_csv = rd / "train_loss.csv"
    if start:
        d = ck / f"step-{start}"
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        _load_rng(st["rng"])
        lines = loss_csv.read_text().splitlines()
        loss_csv.write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start]) + "\n")
    else:
        loss_csv.write_text(LOSS_CSV_HEADER)
    # example order: per-epoch permutation (seeded), 32 per optimizer step
    order, ep = [], 0
    while len(order) < SFT_STEPS * MICRO_BS * GA:
        g = torch.Generator().manual_seed(seed + ep)
        order += torch.randperm(len(recs), generator=g).tolist()
        ep += 1
    for step in range(start + 1, SFT_STEPS + 1):
        t0 = time.time()
        idx = order[(step - 1) * MICRO_BS * GA: step * MICRO_BS * GA]
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for m0 in range(0, len(idx), MICRO_BS):
            mb = [recs[i] for i in idx[m0:m0 + MICRO_BS]]
            n_tok = sum(len(r["completion_ids"]) for r in mb)
            for r in mb:
                p, c = r["student_prompt_ids"], r["completion_ids"]
                ids = torch.tensor([p + c], device="cuda")
                lab = torch.tensor([[-100] * len(p) + c], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids).logits[:, :-1].float()
                nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                                        ignore_index=-100, reduction="sum")
                ((nll / n_tok) / GA).backward()
                losses.append(float(nll) / n_tok)
                del logits, nll
        gn = float(torch.nn.utils.clip_grad_norm_(params, OFFICIAL["max_grad_norm"]))
        opt.step()
        sch.step()
        dt = time.time() - t0
        mlen = np.mean([len(recs[i]["completion_ids"]) for i in idx])
        with open(loss_csv, "a") as fh:
            fh.write(f"{step},{sum(losses) / (len(idx) / MICRO_BS):.6f},{gn:.6f},{sch.get_last_lr()[0]:.4e},{len(idx)},"
                     f"{mlen:.1f},0.0000,1.0000,0.0,{dt:.1f},{dt:.1f},{torch.cuda.max_memory_allocated() / 2**20:.0f},0\n")
        log.info(f"self-SFT step {step}: grad_norm {gn:.4f} lr {sch.get_last_lr()[0]:.3e} {dt:.1f}s")
        if step % SAVE_EVERY == 0:
            d = ck / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "rng": _save_rng(), "step": step},
                       d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
