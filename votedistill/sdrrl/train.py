"""Stage 4: supervised fine-tuning on the SDRRL set (cross-entropy on the assistant turn).

Same base model and LoRA configuration as the COPSD runs, and checkpoints in the same layout
(<ckpt_dir>/step-<N>/adapter), so `python -m votedistill.copsd eval` evaluates them with the identical protocol.
Resumable from the last complete checkpoint: the data order depends only on the seed and the epoch.
"""
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

from ..copsd.config import OFFICIAL
from ..paths import MODEL
from ..utils import read_jsonl, versions, write_json

log = logging.getLogger("votedistill.sdrrl.train")
IM_END = "<|im_end|>\n"


def encode(tok, ex, max_length):
    """-> (input_ids, labels) with the user turn masked out of the loss, or None if the example is too long."""
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["user"]}], tokenize=False,
                                     add_generation_prompt=False) + "<|im_start|>assistant\n"
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(ex["assistant"] + IM_END, add_special_tokens=False).input_ids
    if len(p) + len(a) > max_length:
        return None                        # truncating would cut off the final answer
    return p + a, [-100] * len(p) + a


def schedule(n, epochs, batch, seed):
    """Batches of example indices for every optimizer step: a fresh permutation per epoch."""
    steps = []
    for e in range(epochs):
        idx = list(range(n))
        random.Random(seed + e).shuffle(idx)
        steps += [idx[i:i + batch] for i in range(0, n, batch)]
    return steps


def latest_ckpt(ckpt_dir):
    done = sorted(int(p.name.split("-")[1]) for p in Path(ckpt_dir).glob("step-*") if (p / "COMPLETE").exists())
    return done[-1] if done else None


KEEP_EVERY = 100   # adapters kept besides the latest checkpoint: every KEEP_EVERY steps and each epoch boundary


def prune_checkpoints(ckpt_dir, step, per_epoch):
    """After step's checkpoint is complete: drop the optimizer state of every older checkpoint (only the latest is
    needed to resume) and older checkpoints that are neither a multiple of KEEP_EVERY nor an epoch boundary
    (a Qwen3-4B adapter is ~0.5 GB, its state ~1 GB; keeping all 111 of a 4-epoch run took 164 GB)."""
    for d in Path(ckpt_dir).glob("step-*"):
        s = int(d.name.split("-")[1])
        if s >= step:
            continue
        (d / "state.pt").unlink(missing_ok=True)
        if s % KEEP_EVERY and s % per_epoch:
            shutil.rmtree(d)


def train(run_dir, ckpt_dir, epochs=4, lr=5e-5, batch=32, warmup_ratio=0.1, max_length=4096, save_every=10,
          seed=42, max_steps=None, data_dir=None, run_name=None):
    """data_dir: folder holding sft_data.jsonl (default run_dir). Experiment 2 runs share one data folder across seeds
    (exp2/runs.py, context "sdrrl"); the run folder gets config.json and sft_loss.csv. config.json records
    sft.final_step (the last checkpoint written), which Experiment 2's evaluation reads."""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    run_dir, ckpt_dir = Path(run_dir), Path(ckpt_dir)
    data_dir = Path(data_dir) if data_dir else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    data, dropped = [], 0
    for ex in read_jsonl(data_dir / "sft_data.jsonl"):
        enc = encode(tok, ex, max_length)
        if enc is None:
            dropped += 1
        else:
            data.append(enc)
    steps = schedule(len(data), epochs, batch, seed)
    total = len(steps)
    stop_at = min(total, max_steps) if max_steps else total
    start = latest_ckpt(ckpt_dir) or 0
    log.info(f"{len(data)} examples ({dropped} dropped: longer than {max_length} tokens), {epochs} epochs, "
             f"batch {batch} -> {total} steps; start at {start}, stop at {stop_at}")
    if start >= stop_at:
        log.info("training already complete")
        return
    write_json(run_dir / "config.json", {
        "method": "SDRRL (label-free adaptation): SFT with cross-entropy on the assistant turn", "model": MODEL,
        "run": run_name, "data_dir": str(data_dir),
        "sft": {"epochs": epochs, "learning_rate": lr, "batch": batch, "warmup_ratio": warmup_ratio,
                "lr_scheduler": "linear with warmup", "max_length": max_length, "save_every": save_every, "seed": seed,
                "total_steps": total, "final_step": stop_at, "examples": len(data), "dropped_too_long": dropped,
                "optimizer": "AdamW, no weight decay",
                "max_grad_norm": 1.0, "precision": "bf16 autocast, fp32 LoRA weights"},
        "lora": {"r": OFFICIAL["lora_r"], "alpha": OFFICIAL["lora_alpha"], "dropout": 0.0,
                 "target_modules": OFFICIAL["lora_target_modules"]},
        "checkpoint_dir": str(ckpt_dir), "versions": versions()})

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                                             lora_dropout=0.0, target_modules=OFFICIAL["lora_target_modules"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    loss_csv = run_dir / "sft_loss.csv"
    if start:
        d = ckpt_dir / f"step-{start}"
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        if loss_csv.exists():              # drop rows written after this checkpoint by an interrupted run
            lines = loss_csv.read_text().splitlines()
            loss_csv.write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start]) + "\n")
        log.info(f"RESUMED from {d}")
    if not loss_csv.exists():
        loss_csv.write_text("step,epoch,loss,grad_norm,lr,n_examples,n_target_tokens,step_s,peak_alloc_mib\n")

    model.train()
    for step in range(start + 1, stop_at + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        idx = steps[step - 1]
        n_tok = sum(sum(t != -100 for t in data[i][1]) for i in idx)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in idx:                      # one example at a time: token-level mean over the batch, low peak memory
            ids = torch.tensor([data[i][0]]).cuda()
            lab = torch.tensor([data[i][1]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids).logits
            ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), lab[0, 1:], ignore_index=-100, reduction="sum")
            (ce / n_tok).backward()
            loss_sum += float(ce.detach())
            del logits, ce
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sch.step()
        dt = time.time() - t0
        epoch = (step - 1) * batch // len(data)
        with open(loss_csv, "a") as fh:
            fh.write(f"{step},{epoch},{loss_sum / n_tok:.6f},{gn:.6f},{sch.get_last_lr()[0]:.4e},{len(idx)},{n_tok},"
                     f"{dt:.1f},{torch.cuda.max_memory_allocated() / 2**20:.0f}\n")
        log.info(f"step {step}/{total} epoch {epoch}: loss {loss_sum / n_tok:.4f} grad_norm {gn:.3f} "
                 f"lr {sch.get_last_lr()[0]:.2e} tokens {n_tok} {dt:.1f}s")
        per_epoch = -(-len(data) // batch)
        if step % save_every == 0 or step == stop_at or step % per_epoch == 0:
            d = ckpt_dir / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "step": step}, d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
            log.info(f"checkpoint saved: {d}")
            prune_checkpoints(ckpt_dir, step, per_epoch)
