"""Single-GPU COPSD training loop (resumable from the last complete checkpoint).

Each optimizer step: the student (base + LoRA) samples completions for 32 target-language prompts with vLLM;
student and teacher (same weights, LoRA disabled, privileged prompt) score those completions; the student is
updated toward the teacher's token distributions with the COPSD loss.
"""
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..data import load_copsd_train
from ..gpu import gpu_used_mib, wait_for_gpu
from ..paths import MODEL
from ..utils import versions, write_json
from .collator import make_collator, prepare_rows
from .config import GA, MICRO_BS, OFFICIAL, SAVE_EVERY, SEED, deviations
from .loss import generalized_jsd_loss

log = logging.getLogger("votedistill.copsd.train")
LOSS_CSV_HEADER = ("step,loss,grad_norm,lr,n_prompts,mean_completion_tokens,frac_truncated,frac_boxed,"
                   "gen_s,train_s,step_s,peak_alloc_mib,gpu_used_mib\n")


def data_schedule(n_rows, steps, seed=SEED):
    """HF Trainer-like order: per-epoch random permutation, micro-batches of 4, an optimizer step every GA micro-batches
    or at the end of an epoch (500/4 = 125 micro-batches -> 15 steps of 32 prompts + 1 step of 20 per epoch)."""
    import torch
    sched, epoch = [], 0
    while len(sched) < steps:
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(n_rows, generator=g).tolist()
        mbs = [perm[i:i + MICRO_BS] for i in range(0, n_rows, MICRO_BS)]
        for i in range(0, len(mbs), GA):
            sched.append({"epoch": epoch, "micro_batches": mbs[i:i + GA]})
        epoch += 1
    return sched[:steps]


def _save_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def _load_rng(s):
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def latest_ckpt(ckpt_dir):
    steps = sorted(int(p.name.split("-")[1]) for p in Path(ckpt_dir).glob("step-*") if (p / "COMPLETE").exists())
    return steps[-1] if steps else None


def prune_old_states(ckpt_dir, step):
    """Only the latest optimizer/RNG state is needed to resume; adapters of every step are kept (disk: a Qwen3-4B
    state.pt is ~1 GB per checkpoint). Called after step's COMPLETE marker is written."""
    for p in Path(ckpt_dir).glob("step-*/state.pt"):
        if int(p.parent.name.split("-")[1]) < step:
            p.unlink()


LAYOUTS = {
    "official": "teacher/student prompt right-padded to the longest prompt of its micro-batch, completion appended after "
                "the pads, pad-only attention mask, no position ids (as multilingual_opsd_trainer.py): the completion's "
                "positions are shifted by the number of pads",
    "exact": "pads removed before the completion is appended (each sequence is scored one at a time, so no padding is "
             "needed): the model sees exactly the prompt + completion it would see unbatched",
}


def unpad(ids, pad):
    """Strip right padding; refuses prompts with a pad token anywhere else."""
    keep = ids[ids != pad]
    if not (ids[:len(keep)] == keep).all():
        raise ValueError("pad token inside a prompt (not only right padding)")
    return keep


def write_train_config(run_dir, ckpt_dir, lang, max_steps, include_problem_en, include_reference_solution_en,
                       seed=SEED, layout="official", data_desc=None, reference_desc=None,
                       clip=OFFICIAL["jsd_token_clip"], objective="copsd", teacher_temp=None, rollout_rep_pen=1.0):
    write_json(run_dir / "config.json", {
        "model": MODEL, "language": lang, "official": OFFICIAL,
        "ours": {"gpus": 1, "micro_batch": MICRO_BS, "grad_accum": GA, "effective_batch": MICRO_BS * GA,
                 "train_steps": max_steps, "lr_schedule_horizon": OFFICIAL["max_steps"], "save_every": SAVE_EVERY,
                 "attn": "sdpa", "checkpoint_dir": str(ckpt_dir), "seed": seed,
                 "layout": {"name": layout, "description": LAYOUTS[layout]},
                 "loss_token_clip": clip, "objective": objective, "teacher_temperature": teacher_temp,
                 "rollout_repetition_penalty": rollout_rep_pen},
        "data": data_desc or "COPSD released training set (500 rows)",
        "teacher_context": {"include_problem_en": include_problem_en, "reference_solution": (
            (reference_desc or "gold COPSD `solution`") if include_reference_solution_en else "none")},
        "versions": versions(), "copsd_commit": "4f4410730e9b", "deviations": deviations(max_steps)})


def train(run_dir, ckpt_dir, lang="swa", max_steps=OFFICIAL["max_steps"], stop_after=0, no_save=False,
          include_problem_en=True, include_reference_solution_en=True, seed=SEED, layout="official", rows=None,
          data_desc=None, reference_desc=None, clip=OFFICIAL["jsd_token_clip"], objective="copsd", teacher_temp=None,
          rollout_rep_pen=1.0):
    """rollout_rep_pen: vLLM repetition penalty for the student's rollouts (1.0 = official; >1 = penalty-rollout
    variant: the student still samples its own rollouts, but with a decoding-time guard against loops).
    objective: "copsd" (distillation from the teacher prompt; teacher_temp, if set, replaces the loss temperature
    for the teacher's logits only, e.g. a hotter copy of the base model = an entropy-raising target), or
    "unlikelihood" (no teacher: -log(1 - p_student(y)) at positions whose token continues a verbatim 8-token repeat
    of the student's own rollout; Welleck et al. 2019). clip: the per-(token, vocab) loss clamp (official 0.05; None = no clip). rows: prepared training rows (copsd.collator.prepare_rows format, `solution` = the teacher's reference);
    default the released COPSD set. layout: see LAYOUTS. seed: data order, LoRA init, generation and RNG state."""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    run_dir, ckpt_dir = Path(run_dir), Path(ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0 if no_save else (latest_ckpt(ckpt_dir) or 0)
    if start_step >= max_steps:
        log.info(f"training already complete (step {start_step})")
        return
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout}; known: {list(LAYOUTS)}")
    if not no_save:
        write_train_config(run_dir, ckpt_dir, lang, max_steps, include_problem_en, include_reference_solution_en,
                           seed, layout, data_desc, reference_desc, clip, objective, teacher_temp, rollout_rep_pen)
    wait_for_gpu()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collator = make_collator(tok, OFFICIAL["max_length"], include_problem_en, include_reference_solution_en)
    if rows is None:
        rows = prepare_rows(load_copsd_train(lang), lang)
    sched = data_schedule(len(rows), OFFICIAL["max_steps"], seed)

    # vLLM first (it sizes itself from free memory); 0.4 of the GPU as in the official colocate setting
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=OFFICIAL["vllm_gpu_memory_utilization"],
              enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, max_model_len=8192,
              max_num_seqs=MICRO_BS * GA, seed=seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                                             lora_dropout=OFFICIAL["lora_dropout_configured"],
                                             target_modules=OFFICIAL["lora_target_modules"]))
    for m in model.modules():                         # disable_dropout=True (TRL GOLDConfig default)
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OFFICIAL["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, 0, OFFICIAL["max_steps"])
    loss_csv = run_dir / ("timing_steps.csv" if no_save else "train_loss.csv")
    if no_save and loss_csv.exists():
        loss_csv.unlink()
    if start_step:
        d = ckpt_dir / f"step-{start_step}"
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        _load_rng(st["rng"])
        if loss_csv.exists():   # drop loss rows written after this checkpoint by an interrupted run
            lines = loss_csv.read_text().splitlines()
            loss_csv.write_text("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start_step]) + "\n")
        log.info(f"RESUMED from {d}: step {start_step}, lr {sch.get_last_lr()[0]:.3e}")
    else:
        log.info(f"fresh start; trainable params {sum(p.numel() for p in params):,}; dtype {params[0].dtype}")
    if not loss_csv.exists():
        loss_csv.write_text(LOSS_CSV_HEADER)

    sync_dir = Path(f"/tmp/votedistill_vllm_adapter_{os.environ.get('SLURM_JOB_ID', os.getpid())}")   # node-local

    def sync_adapter(step):
        if sync_dir.exists():
            shutil.rmtree(sync_dir)
        model.save_pretrained(str(sync_dir))
        return LoRARequest(f"student_step{step}", step + 1, str(sync_dir))

    pad, temp = tok.pad_token_id, OFFICIAL["temperature"]
    stop_at = min(max_steps, start_step + stop_after) if stop_after else max_steps
    for step in range(start_step + 1, stop_at + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        lora = sync_adapter(step - 1) if step > 1 else None   # step 1: LoRA B=0 -> base model == initial student
        batches = [collator([rows[i] for i in mb]) for mb in sched[step - 1]["micro_batches"]]
        prompts = [t.replace(tok.pad_token, "") for b in batches
                   for t in tok.batch_decode(b["student_prompts"], skip_special_tokens=False)]
        sps = [SamplingParams(n=1, temperature=temp, top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"], min_p=0.0,
                              repetition_penalty=rollout_rep_pen, presence_penalty=0.0, max_tokens=OFFICIAL["max_completion_length"],
                              seed=seed * 1_000_003 + step * 1000 + j) for j in range(len(prompts))]
        outs = llm.generate(prompts, sps, lora_request=lora, use_tqdm=False)
        comps = [list(o.outputs[0].token_ids) for o in outs]
        texts = [o.outputs[0].text for o in outs]
        t_gen = time.time() - t0

        model.train()
        opt.zero_grad(set_to_none=True)
        mb_losses, k = [], 0
        for b in batches:
            n_mb = len(b["student_prompts"])
            mc = comps[k:k + n_mb]
            k += n_mb
            # official: labels = generated ids with the per-example prompt and all pad tokens masked
            n_tokens = sum(sum(1 for t in c if t != pad) for c in mc)
            mb_loss = 0.0
            for i in range(n_mb):   # per-sample backward == official micro-batch loss (sum over tokens / micro-batch token count)
                c = torch.tensor(mc[i], dtype=torch.long)
                sp_i, tp_i = b["student_prompts"][i], b["teacher_prompts"][i]
                if layout == "exact":
                    sp_i, tp_i = unpad(sp_i, pad), unpad(tp_i, pad)
                s_ids = torch.cat([sp_i, c]).unsqueeze(0).cuda()
                t_ids = torch.cat([tp_i, c]).unsqueeze(0).cuda()
                lab = c.clone()
                lab[lab == pad] = -100
                lab = lab.unsqueeze(0).cuda()
                L = len(mc[i])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s_logits = model(input_ids=s_ids, attention_mask=(s_ids != pad).long(), logits_to_keep=L + 1).logits[:, :-1, :]
                    if objective == "unlikelihood":
                        from ..exp2.mechanism import loop_flags
                        lp = torch.log_softmax(s_logits.float() / temp, -1)
                        py = lp.gather(-1, lab.clamp(min=0).unsqueeze(-1))[..., 0].exp()
                        m = torch.tensor(loop_flags(mc[i]), device=py.device).unsqueeze(0) & (lab != -100)
                        per = -(torch.log1p(-py.clamp(max=1 - 1e-6)) * m).sum()
                        t_logits = lp
                    else:
                        with torch.no_grad(), model.disable_adapter():
                            t_logits = model(input_ids=t_ids, attention_mask=(t_ids != pad).long(), logits_to_keep=L + 1).logits[:, :-1, :]
                        if teacher_temp:   # the loss divides both by temp: teacher effectively at teacher_temp
                            t_logits = t_logits.float() * (temp / teacher_temp)
                        per = generalized_jsd_loss(s_logits, t_logits, labels=lab, beta=OFFICIAL["beta"], temperature=temp,
                                                   reduction="sum", token_clip=clip)
                loss_i = per / max(n_tokens, 1)
                (loss_i / len(batches)).backward()   # HF Trainer divides each micro-batch loss by the GA steps in this step
                mb_loss += float(loss_i.detach())
                del s_logits, t_logits, per, loss_i
            mb_losses.append(mb_loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, OFFICIAL["max_grad_norm"]))
        opt.step()
        sch.step()
        t_all = time.time() - t0
        loss = float(np.mean(mb_losses))
        trunc = np.mean([o.outputs[0].finish_reason == "length" for o in outs])
        boxed = np.mean(["\\boxed" in t for t in texts])
        mlen = np.mean([len(c) for c in comps])
        pk = torch.cuda.max_memory_allocated() / 2**20
        used = gpu_used_mib()
        with open(loss_csv, "a") as fh:
            fh.write(f"{step},{loss:.6f},{gn:.6f},{sch.get_last_lr()[0]:.4e},{len(prompts)},{mlen:.1f},{trunc:.4f},"
                     f"{boxed:.4f},{t_gen:.1f},{t_all - t_gen:.1f},{t_all:.1f},{pk:.0f},{used}\n")
        log.info(f"step {step}: loss {loss:.5f} grad_norm {gn:.4f} lr {sch.get_last_lr()[0]:.3e} prompts {len(prompts)} "
                 f"mean_len {mlen:.0f} trunc {trunc:.2f} boxed {boxed:.2f} gen {t_gen:.1f}s train {t_all - t_gen:.1f}s "
                 f"total {t_all:.1f}s torch_peak {pk:.0f}MiB gpu_used {used}MiB")
        if step % SAVE_EVERY == 0 and not no_save:
            d = ckpt_dir / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "rng": _save_rng(), "step": step},
                       d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
            prune_old_states(ckpt_dir, step)
            log.info(f"checkpoint saved: {d}")
    log.info(f"train loop exit at step {stop_at}")
