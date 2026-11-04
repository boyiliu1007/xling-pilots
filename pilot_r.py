"""Pilot R: gold COPSD reproduction on AfriMGSM Swahili with Qwen3-1.7B on one A6000 (resumable).

Reuses the official COPSD code where possible:
  - prompts / labels / think prefix: COPSD language_config.LANGUAGE_CONFIG + MultilingualSelfDistillationDataCollator
  - loss: OPSDTrainer.generalized_jsd_loss (copied verbatim below)
  - eval: african_langs_eval/evaluate_math.py (LANGUAGE_PROMPTS, extract_boxed_answer, grade_answer)

Subcommands (all idempotent; see run_pilot_r.sbatch):
  split                      make DEV/TEST split of AfriMGSM swa test
  train  [--stop_after N]    train / resume until --max_steps (default 60)
  eval   --names ... --split dev|test
  report
"""
import argparse
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

COPSD = Path("/data/user_data/boyil2/ext/COPSD")
sys.path.insert(0, str(COPSD))
sys.path.insert(0, str(COPSD / "african_langs_eval"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HOME", "/data/user_data/boyil2/hf_cache")

MODEL = "Qwen/Qwen3-1.7B"
LANG = "SWA"
OUT = HERE / "pilot_r_out"
CKPT = Path("/data/user_data/boyil2/pilot_r_ckpt")          # checkpoints (large) live on the data disk
SYNC_DIR = Path(f"/tmp/pilot_r_vllm_adapter_{os.environ.get('SLURM_JOB_ID', 'local')}")  # node-local, per-step vLLM sync

# ---- official COPSD Qwen3-1.7B hyperparameters (african_langs_scripts/run_all_opsd_1.7b_train.sh) ----
OFFICIAL = dict(learning_rate=5e-6, max_grad_norm=0.1, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                num_processes=4, effective_batch_size=32, max_steps=100, save_steps=5, max_completion_length=2048,
                max_length=20000, vllm_gpu_memory_utilization=0.4, lora_r=64, lora_alpha=128,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                temperature=1.1, top_p=0.95, top_k=20, beta=0, lmbda=1, fixed_teacher=True,
                student_enable_thinking=True, jsd_token_clip=0.05, reason_first=False,
                include_problem_en=True, include_reference_solution_en=True,
                # TRL 0.26 GOLDConfig / SFTConfig / TrainingArguments defaults not overridden by the script:
                disable_dropout=True, lora_dropout_configured=0.05, bf16_mixed=True, gradient_checkpointing=True,
                vllm_sync_frequency=1, lr_scheduler="linear (TrainingArguments default), no warmup",
                weight_decay=0.0, optimizer="AdamW (betas 0.9/0.999, eps 1e-8)", seed=42,
                eval=dict(val_n=12, temperature=1.0, top_p=0.95, top_k=-1, max_new_tokens=1024, enable_thinking=True))
MICRO_BS = 4
GA = 8                     # 1 GPU x 4 x 8 = 32 = official 4 GPUs x 4 x 2
TRAIN_STEPS = 60           # min(official 100, 60)
SAVE_EVERY = 10
log = logging.getLogger("pilot_r")

# ---- variants: "r" = Pilot R (thinking, official eval); "r2" = Pilot R2 (non-thinking everywhere) ----
VARIANT = "r"
EVAL_R2 = dict(val_n=12, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=2048, enable_thinking=False)


def set_variant(v):
    global VARIANT, OUT, CKPT, SYNC_DIR
    VARIANT = v
    if v == "r2":
        OUT = HERE / "pilot_r2_out"
        CKPT = Path("/data/user_data/boyil2/pilot_r2_ckpt")
        SYNC_DIR = Path(f"/tmp/pilot_r2_vllm_adapter_{os.environ.get('SLURM_JOB_ID', 'local')}")
    if v == "r3":
        OUT = HERE / "pilot_r3_out"          # evaluation only; reads Pilot R's checkpoints (CKPT unchanged)


def r2_instruction():
    # COPSD's own Swahili text asking to think step by step *in Swahili*: language_config.py line 681 (SWA
    # teacher_final_instruction). At import, language_config.py line 884 overwrites it with the plain student instruction
    # for all African languages, so it is quoted literally here.
    return "Tafadhali fikiri hatua kwa hatua kwa Kiswahili, na uweke jibu lako la mwisho ndani ya \\boxed{}."


def make_collator(tok):
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    if VARIANT == "r":
        return MultilingualSelfDistillationDataCollator(tok, max_length=OFFICIAL["max_length"], reason_first=False,
                                                        student_enable_thinking=True, include_problem_en=True,
                                                        include_reference_solution_en=True)
    import multilingual_data_collator as MDC
    MDC.LANGUAGE_CONFIG[LANG]["student_instruction"] = r2_instruction()   # Swahili-reasoning instruction in the user prompt

    class R2Collator(MultilingualSelfDistillationDataCollator):
        # teacher gets the same non-thinking assistant prefix as the student (no <think> + think prefix)
        def _append_qwen3_think_prefix(self, base_prompt, lang_code):
            return self._append_qwen3_assistant_prefix(base_prompt)
    return R2Collator(tok, max_length=OFFICIAL["max_length"], reason_first=False, student_enable_thinking=False,
                      include_problem_en=True, include_reference_solution_en=True)


def eval_prompt(tok, question):
    import evaluate_math as EM
    if VARIANT in ("r", "r3"):
        return EM.build_qwen3_prompt(tok, question, LANG, enable_thinking=True)
    return tok.apply_chat_template([{"role": "user", "content": f"{question}\n\n{r2_instruction()}"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def eval_cfg():
    if VARIANT == "r3":   # COPSD paper main setting: thinking, n=12, T=1.0, top_p 0.95, 4096 new tokens
        return {**OFFICIAL["eval"], "max_new_tokens": 4096}
    return OFFICIAL["eval"] if VARIANT == "r" else EVAL_R2


def setup_logging():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])
    for nm in ("httpx", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(nm).setLevel(logging.WARNING)


def wait_gpu():
    from pilot_c import wait_for_gpu
    wait_for_gpu()


def peak_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.split()[0]
        return int(o)
    except Exception:
        return -1


# ============================================================================ data

def load_train_rows():
    from datasets import load_dataset
    ds = load_dataset("yihongLiu/COPSD-AfriMGSM-TrainDataset", data_files="data/swa-00000-of-00001.parquet", split="train")
    rows = []
    for r in ds:
        r = dict(r)
        # Released data stores the translation as `problem_translated`; the official train script expects
        # `problem_<lang>` (README: "adapt the script to the training set structure we released").
        r["problem_swa"] = r["problem_translated"]
        r["target_lang"] = LANG                      # add_target_language()
        r["problem_en"] = r["problem"]
        rows.append(r)
    return rows


def load_afrimgsm():
    from datasets import load_dataset
    ds = load_dataset("masakhane/afrimgsm", "swa", split="test")
    return [dict(r) for r in ds]


def cmd_split(args):
    p = OUT / "split.json"
    if p.exists():
        log.info(f"split exists: {p}")
        return
    rows = load_afrimgsm()
    idx = list(range(len(rows)))
    random.Random(42).shuffle(idx)
    split = {"seed": 42, "method": "random.Random(42).shuffle(range(250)); DEV = first 50, TEST = rest; sorted",
             "dev": sorted(idx[:50]), "test": sorted(idx[50:])}
    p.write_text(json.dumps(split, indent=1))
    log.info(f"split saved: DEV {len(split['dev'])}, TEST {len(split['test'])}")


# ============================================================================ loss (verbatim from COPSD OPSDTrainer)
# Copied verbatim from OPSDTrainer.generalized_jsd_loss in https://github.com/cisnlp/COPSD (commit 4f4410730e9b),
# which extends TRL's GKDTrainer.generalized_jsd_loss (https://github.com/huggingface/trl, Apache-2.0) with top_k and
# token_clip. The COPSD repository does not declare a license; all credit for this function belongs to its authors.

def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean",
                         logits_are_probs=False, top_k=None, token_clip=None):
    import torch
    import torch.nn.functional as F
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]
    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    return jsd


# ============================================================================ training

def data_schedule(n_rows, steps, seed=42):
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


def save_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def load_rng(s):
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def latest_ckpt():
    if not CKPT.exists():
        return None
    steps = sorted(int(p.name.split("-")[1]) for p in CKPT.glob("step-*") if (p / "COMPLETE").exists())
    return steps[-1] if steps else None


def cmd_train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    start_step = latest_ckpt() or 0
    if start_step >= args.max_steps:
        log.info(f"training already complete (step {start_step})")
        return
    wait_gpu()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collator = make_collator(tok)
    rows = load_train_rows()
    sched = data_schedule(len(rows), OFFICIAL["max_steps"])

    # vLLM first (it sizes itself from free memory); 0.4 of the GPU as in the official colocate setting
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=OFFICIAL["vllm_gpu_memory_utilization"],
              enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, max_model_len=8192,
              max_num_seqs=MICRO_BS * GA, seed=42)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    lcfg = LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
                      lora_dropout=OFFICIAL["lora_dropout_configured"], target_modules=OFFICIAL["lora_target_modules"])
    model = get_peft_model(model, lcfg)
    for m in model.modules():                         # disable_dropout=True (TRL GOLDConfig default)
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OFFICIAL["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, 0, OFFICIAL["max_steps"])
    loss_csv = OUT / ("timing_steps.csv" if args.no_save else "train_loss.csv")
    if args.no_save and loss_csv.exists():
        loss_csv.unlink()
    if start_step and args.no_save:
        raise SystemExit("--no_save timing run must start from step 0 (checkpoints exist)")
    if start_step:
        d = CKPT / f"step-{start_step}"
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        st = torch.load(d / "state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        load_rng(st["rng"])
        # drop loss rows written after this checkpoint (from an interrupted run)
        if loss_csv.exists():
            lines = loss_csv.read_text().splitlines()
            keep = [lines[0]] + [l for l in lines[1:] if int(l.split(",")[0]) <= start_step]
            loss_csv.write_text("\n".join(keep) + "\n")
        log.info(f"RESUMED from {d}: step {start_step}, lr {sch.get_last_lr()[0]:.3e}")
    else:
        log.info(f"fresh start; trainable params {sum(p.numel() for p in params):,}; dtype {params[0].dtype}")
    if not loss_csv.exists():
        loss_csv.write_text("step,loss,grad_norm,lr,n_prompts,mean_completion_tokens,frac_truncated,frac_boxed,"
                            "gen_s,train_s,step_s,peak_alloc_mib,gpu_used_mib\n")

    def sync_adapter(step):
        if SYNC_DIR.exists():
            shutil.rmtree(SYNC_DIR)
        model.save_pretrained(str(SYNC_DIR))
        return LoRARequest(f"student_step{step}", step + 1, str(SYNC_DIR))

    pad = tok.pad_token_id
    temp = OFFICIAL["temperature"]
    stop_at = min(args.max_steps, start_step + args.stop_after) if args.stop_after else args.max_steps
    for step in range(start_step + 1, stop_at + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        plan = sched[step - 1]
        lora = sync_adapter(step - 1) if step > 1 else None   # step 1: LoRA B=0 -> base model == initial student
        batches = [collator([rows[i] for i in mb]) for mb in plan["micro_batches"]]
        prompts = []
        for b in batches:
            txt = tok.batch_decode(b["student_prompts"], skip_special_tokens=False)
            prompts += [t.replace(tok.pad_token, "") for t in txt]
        sps = [SamplingParams(n=1, temperature=temp, top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"], min_p=0.0,
                              repetition_penalty=1.0, presence_penalty=0.0, max_tokens=OFFICIAL["max_completion_length"],
                              seed=42 * 1_000_003 + step * 1000 + j) for j in range(len(prompts))]
        outs = llm.generate(prompts, sps, lora_request=lora, use_tqdm=False)
        comps = [list(o.outputs[0].token_ids) for o in outs]
        texts = [o.outputs[0].text for o in outs]
        t_gen = time.time() - t0

        model.train()
        opt.zero_grad(set_to_none=True)
        mb_losses, k = [], 0
        for b in batches:
            P_s, P_t = b["student_prompt_length"], b["teacher_prompt_length"]
            n_mb = len(b["student_prompts"])
            mc = comps[k:k + n_mb]
            k += n_mb
            # official: labels = generated ids with the per-example prompt and all pad tokens masked
            n_tokens = sum(sum(1 for t in c if t != pad) for c in mc)
            mb_loss = 0.0
            for i in range(n_mb):   # per-sample backward == official micro-batch loss (sum over tokens / micro-batch token count)
                c = torch.tensor(mc[i], dtype=torch.long)
                s_ids = torch.cat([b["student_prompts"][i], c]).unsqueeze(0).cuda()
                t_ids = torch.cat([b["teacher_prompts"][i], c]).unsqueeze(0).cuda()
                s_mask = (s_ids != pad).long()
                t_mask = (t_ids != pad).long()
                lab = c.clone()
                lab[lab == pad] = -100
                lab = lab.unsqueeze(0).cuda()
                L = len(mc[i])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s_logits = model(input_ids=s_ids, attention_mask=s_mask, logits_to_keep=L + 1).logits[:, :-1, :]
                    with torch.no_grad(), model.disable_adapter():
                        t_logits = model(input_ids=t_ids, attention_mask=t_mask, logits_to_keep=L + 1).logits[:, :-1, :]
                    per = generalized_jsd_loss(s_logits, t_logits, labels=lab, beta=OFFICIAL["beta"], temperature=temp,
                                               reduction="sum", token_clip=OFFICIAL["jsd_token_clip"])
                loss_i = per / max(n_tokens, 1)
                (loss_i / len(batches)).backward()   # HF Trainer divides each micro-batch loss by GA steps in this step
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
        used = peak_mib()
        with open(loss_csv, "a") as fh:
            fh.write(f"{step},{loss:.6f},{gn:.6f},{sch.get_last_lr()[0]:.4e},{len(prompts)},{mlen:.1f},{trunc:.4f},"
                     f"{boxed:.4f},{t_gen:.1f},{t_all - t_gen:.1f},{t_all:.1f},{pk:.0f},{used}\n")
        log.info(f"step {step}: loss {loss:.5f} grad_norm {gn:.4f} lr {sch.get_last_lr()[0]:.3e} prompts {len(prompts)} "
                 f"mean_len {mlen:.0f} trunc {trunc:.2f} boxed {boxed:.2f} gen {t_gen:.1f}s train {t_all - t_gen:.1f}s "
                 f"total {t_all:.1f}s torch_peak {pk:.0f}MiB gpu_used {used}MiB")
        if step % SAVE_EVERY == 0 and not args.no_save:
            d = CKPT / f"step-{step}"
            if d.exists():
                shutil.rmtree(d)
            (d / "adapter").mkdir(parents=True)
            model.save_pretrained(str(d / "adapter"))
            torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "rng": save_rng(), "step": step},
                       d / "state.pt")
            (d / "COMPLETE").write_text(datetime.now().isoformat())
            log.info(f"checkpoint saved: {d}")
    log.info(f"train loop exit at step {stop_at}")


# ============================================================================ evaluation

SW_WORDS = set("""na ya wa kwa ni la za katika kwamba hii hiyo hizo huo kila kama au lakini sasa hivyo pia ili kwenye yake
wake yao jumla baada kisha je hapa hivi ambayo ambao ambaye kuwa nini zaidi moja mbili tatu nne tano hatua kupata jibu
tunahitaji tuna basi kutoka hadi kati mara ndani hapo kwanza pili tena yote wote kiasi idadi bei siku saa kila""".split())
EN_WORDS = set("""the and of to is we so that in for this it with be are then total each let first find need what how
there which from by on at as if has have was will can but not or our step answer number per now since thus therefore
calculate given""".split())


def language_of(text):
    """'sw' / 'en' / 'other' by function-word counts on the generated text with LaTeX/math removed."""
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$|\\boxed\{[^}]*\}", " ", text, flags=re.S)
    w = re.findall(r"[a-zA-Z']+", t.lower())
    sw = sum(x in SW_WORDS for x in w)
    en = sum(x in EN_WORDS for x in w)
    if sw == 0 and en == 0:
        return "other"
    return "sw" if sw > en else "en"


def cmd_eval(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import evaluate_math as EM
    split = json.loads((OUT / "split.json").read_text())
    rows = load_afrimgsm()
    E = eval_cfg()
    todo = []
    for name in args.names:
        out = OUT / "eval" / f"{name}_{args.split}.jsonl"
        if out.exists():
            log.info(f"[skip] {out} exists")
            continue
        todo.append(name)
    if not todo:
        return
    wait_gpu()
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_cpu_loras=2, max_model_len=max(4096, E["max_new_tokens"] + 2048), seed=42)
    log.info(f"eval config: {E}; max_model_len {max(4096, E['max_new_tokens'] + 2048)}")
    ids = split[args.split]
    prompts = [eval_prompt(tok, rows[i]["question"]) for i in ids]
    log.info(f"eval prompt example:\n{prompts[0]}")
    for li, name in enumerate(todo):
        lora = None
        if name != "base":
            step = int(name.split("step")[1])
            lora = LoRARequest(name, li + 1, str(CKPT / f"step-{step}" / "adapter"))
        sp = SamplingParams(n=E["val_n"], temperature=E["temperature"], top_p=E["top_p"], top_k=E["top_k"],
                            max_tokens=E["max_new_tokens"], seed=42)
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
            recs.append({"model": name, "split": args.split, "item": i, "gold": gt, "outputs": texts, "preds": preds,
                         "correct": corr, "formatted": fmt_, "truncated": [c.finish_reason == "length" for c in o.outputs],
                         "language": [language_of(t) for t in texts], "pass_at_n": any(corr), "maj_correct": maj,
                         "mean_acc": sum(corr) / len(corr)})
        (OUT / "eval").mkdir(exist_ok=True)
        with open(OUT / "eval" / f"{name}_{args.split}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(recs)
        log.info(f"EVAL {name} {args.split}: {json.dumps(s)} ({dt:.0f}s)")


def summarize(recs):
    n = len(recs)
    S = sum(len(r["correct"]) for r in recs)
    return {"n": n, "pass@12": sum(r["pass_at_n"] for r in recs) / n, "maj@12": sum(r["maj_correct"] for r in recs) / n,
            "mean_acc": sum(r["mean_acc"] for r in recs) / n,
            "format_rate": sum(sum(r["formatted"]) for r in recs) / S,
            "trunc_rate": sum(sum(r["truncated"]) for r in recs) / S,
            "sw_rate": sum(sum(l == "sw" for l in r["language"]) for r in recs) / S}


# ============================================================================ report

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def cmd_report(args):
    from stats import fmt, wilson
    T = [{"r": "PILOT R — gold COPSD reproduction on AfriMGSM Swahili (Qwen3-1.7B, 1x A6000)",
          "r2": "PILOT R2 — gold COPSD on AfriMGSM Swahili, NON-THINKING mode (Qwen3-1.7B, 1x A6000)",
          "r3": "PILOT R3 — Pilot R checkpoints re-evaluated with COPSD's paper protocol (thinking, 4096 new tokens)"}[VARIANT],
         f"generated {datetime.now().isoformat(timespec='seconds')}"]
    cfg = json.loads((OUT / "config.json").read_text()) if (OUT / "config.json").exists() else {}
    T.append("\n## Setup and deviations")
    for d in cfg.get("deviations", []):
        T.append(f"  - {d}")
    split = json.loads((OUT / "split.json").read_text())
    T.append(f"  split: {split['method']}  (DEV {len(split['dev'])}, TEST {len(split['test'])})")
    if VARIANT in ("r", "r3"):
        T.append(f"  eval protocol: COPSD evaluate_math.py for SWA: thinking prompt + think prefix, n=12, T=1.0, top_p 0.95, "
                 f"top_k -1, max_new_tokens {eval_cfg()['max_new_tokens']}; extract_boxed_answer + grade_answer (math-verify); "
                 f"maj@12 = COPSD Counter vote")
        if VARIANT == "r3":
            T.append(f"  checkpoints: Pilot R LoRA adapters from {CKPT} (no training in R3); same DEV/TEST split as Pilot R")
    else:
        T.append(f"  eval protocol (R2): Qwen3 chat template enable_thinking=False (empty think block), user prompt = question + "
                 f"COPSD Swahili instruction '{r2_instruction()}'; n=12, T=0.7, top_p 0.8, top_k 20, max_new_tokens 2048; "
                 f"COPSD extract_boxed_answer + grade_answer (math-verify); maj@12 = COPSD Counter vote")
        lc_ = OUT / "loss_check.json"
        if lc_.exists():
            T.append("\n## Loss-sign check (before training)")
            for k, v in json.loads(lc_.read_text()).items():
                T.append(f"  {k}: {json.dumps(v)}")
    T.append("  language heuristic: generated text with LaTeX/$..$/boxed removed; count Swahili vs English function words "
             "from fixed lists; 'sw' if Swahili count > English count (ties/none -> not sw)")

    lc = OUT / "train_loss.csv"
    if lc.exists():
        L = [l.split(",") for l in lc.read_text().splitlines()]
        h, rows_ = L[0], L[1:]
        col = {k: i for i, k in enumerate(h)}
        st = [float(r[col["step_s"]]) for r in rows_]
        T.append("\n## Training")
        T.append(f"  steps logged: {len(rows_)}; seconds/step mean {np.mean(st):.1f} (min {min(st):.1f}, max {max(st):.1f}); "
                 f"total step time {sum(st) / 3600:.2f} h; peak torch alloc {max(float(r[col['peak_alloc_mib']]) for r in rows_):.0f} MiB; "
                 f"peak GPU used (incl. vLLM 0.4 share) {max(int(r[col['gpu_used_mib']]) for r in rows_)} MiB")
        T.append("  step  loss      grad_norm  mean_len  trunc  boxed")
        for r in rows_:
            if int(r[0]) % 5 == 0 or int(r[0]) <= 3:
                T.append(f"  {r[0]:>4}  {float(r[1]):.5f}  {float(r[2]):.4f}    {float(r[col['mean_completion_tokens']]):>6.0f}   "
                         f"{float(r[col['frac_truncated']]):.2f}   {float(r[col['frac_boxed']]):.2f}")

    ev = OUT / "eval"
    def get(name, sp):
        p = ev / f"{name}_{sp}.jsonl"
        return read_jsonl(p) if p.exists() else None

    names = ["base"] + [f"step{s}" for s in range(SAVE_EVERY, TRAIN_STEPS + 1, SAVE_EVERY)]
    T.append("\n## DEV curve (50 items; used only for checkpoint selection)")
    T.append("  model    pass@12        maj@12   mean_acc  format  trunc  sw-lang")
    dev = {}
    for nm in names:
        R = get(nm, "dev")
        if R is None:
            T.append(f"  {nm:<8} (not evaluated)")
            continue
        s = summarize(R)
        dev[nm] = s
        T.append(f"  {nm:<8} {fmt(sum(r['pass_at_n'] for r in R), len(R)):<26} {s['maj@12']:.3f}   {s['mean_acc']:.3f}    "
                 f"{s['format_rate']:.3f}   {s['trunc_rate']:.3f}  {s['sw_rate']:.3f}")
    ck = [n for n in dev if n != "base"]
    sel = None
    if ck:
        sel = sorted(ck, key=lambda n: (-dev[n]["pass@12"], int(n[4:])))[0]
        T.append(f"  SELECTED (highest DEV pass@12, ties -> earlier step): {sel}")

    T.append("\n## TEST (200 items): base vs selected checkpoint")
    B = get("base", "test")
    Ssel = get(sel, "test") if sel else None
    verdict = None
    if B and Ssel:
        assert [r["item"] for r in B] == [r["item"] for r in Ssel]
        for nm, R in (("base", B), (sel, Ssel)):
            s = summarize(R)
            T.append(f"  {nm:<8} pass@12 {fmt(sum(r['pass_at_n'] for r in R), len(R))}  maj@12 {fmt(sum(r['maj_correct'] for r in R), len(R))}  "
                     f"mean_acc {s['mean_acc']:.3f}  format {s['format_rate']:.3f}  trunc {s['trunc_rate']:.3f}  sw-lang {s['sw_rate']:.3f}")
        a = np.array([[r["pass_at_n"] for r in B], [r["pass_at_n"] for r in Ssel]], dtype=float)
        m = np.array([[r["mean_acc"] for r in B], [r["mean_acc"] for r in Ssel]], dtype=float)
        f_ = np.array([[np.mean(r["formatted"]) for r in B], [np.mean(r["formatted"]) for r in Ssel]], dtype=float)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, a.shape[1], size=(10000, a.shape[1]))
        def bs(x):
            d = x[1] - x[0]
            v = d[idx].mean(1)
            return d.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)
        dp, dm, df = bs(a), bs(m), bs(f_)
        T.append(f"  Δ pass@12   = {dp[0] * 100:+.2f} pts  [{dp[1] * 100:+.2f}, {dp[2] * 100:+.2f}]  (paired bootstrap, 10,000 resamples)")
        T.append(f"  Δ mean_acc  = {dm[0] * 100:+.2f} pts  [{dm[1] * 100:+.2f}, {dm[2] * 100:+.2f}]")
        T.append(f"  Δ format    = {df[0] * 100:+.2f} pts  [{df[1] * 100:+.2f}, {df[2] * 100:+.2f}]   (next to Δ accuracy: do gains track format?)")
        disc_up = int(((a[1] == 1) & (a[0] == 0)).sum())
        disc_dn = int(((a[1] == 0) & (a[0] == 1)).sum())
        T.append(f"  pass@12 discordant items: gained {disc_up}, lost {disc_dn}")
        # accuracy among formatted samples, to separate format from correctness
        facc = {}
        for nm, R in (("base", B), (sel, Ssel)):
            fc = sum(c for r in R for c, f in zip(r["correct"], r["formatted"]) if f)
            fn = sum(sum(r["formatted"]) for r in R)
            facc[nm] = fc / fn if fn else float("nan")
            T.append(f"  {nm:<8} accuracy among formatted samples: {fmt(fc, fn)}")
        dfa = facc[sel] - facc["base"]
        T.append(f"  Δ accuracy among formatted samples = {dfa * 100:+.2f} pts")
        if VARIANT == "r":
            verdict = "GO" if (dp[0] >= 0.03 and dp[1] > 0) else "NO-GO"
        else:
            verdict = "GO" if (dp[0] >= 0.03 and dp[1] > 0 and dfa >= -0.03) else "NO-GO"
            T.append(f"  rule parts: Δpass@12 >= 3 pts: {dp[0] >= 0.03}; CI excludes 0: {dp[1] > 0}; "
                     f"formatted-accuracy drop <= 3 pts: {dfa >= -0.03}")
    else:
        T.append("  (not available yet)")

    T.append("\n## ORACLE (best checkpoint on TEST; NOT used for the decision)")
    best = None
    for nm in names[1:]:
        R = get(nm, "test")
        if R is not None:
            s = summarize(R)
            T.append(f"  {nm:<8} TEST pass@12 {s['pass@12']:.3f}  mean_acc {s['mean_acc']:.3f}  format {s['format_rate']:.3f}")
            if best is None or s["pass@12"] > best[1]:
                best = (nm, s["pass@12"])
    if best:
        T.append(f"  ORACLE = {best[0]} (TEST pass@12 {best[1]:.3f})")

    if VARIANT == "r3" and B:
        T.append("\n## Reference only: COPSD paper base (Qwen3-1.7B, Swahili, pass@12, 4096 tokens) = 14.8")
        T.append(f"  our base TEST pass@12 = {100 * summarize(B)['pass@12']:.1f} on a 200-item subset of the 250-item AfriMGSM "
                 f"swa test (different items, sampling seed and software; not a like-for-like comparison)")
        Bd = get("base", "dev")
        if Bd:
            allp = (sum(r["pass_at_n"] for r in B) + sum(r["pass_at_n"] for r in Bd)) / (len(B) + len(Bd))
            T.append(f"  our base pass@12 on all 250 items (DEV+TEST pooled) = {100 * allp:.1f}")
    if VARIANT == "r":
        T.append("\n## Pre-registered decision: GO iff TEST Δpass@12 >= +3 pts AND its 95% bootstrap CI excludes 0")
    else:
        T.append(f"\n## Pre-registered decision ({VARIANT.upper()}): GO iff TEST Δpass@12 >= +3 pts AND its 95% bootstrap CI excludes 0 "
                 "AND accuracy among formatted samples does not drop by more than 3 pts")
    T.append(f"  => {verdict or 'PENDING'}")
    text = "\n".join(T) + "\n"
    (OUT / "report.txt").write_text(text)
    print(text)


def cmd_losscheck(args):
    """Per-token loss on one training micro-batch, with and without the official 0.05 per-(token,vocab) clip.
    Unclipped per-token value = sum_v p_t (log p_t - log p_s) = KL(teacher||student) >= 0; negative => bug."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    wait_gpu()
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    collator = make_collator(tok)
    rows = load_train_rows()
    plan = data_schedule(len(rows), OFFICIAL["max_steps"])[0]["micro_batches"][: args.n_microbatches]
    batches = [collator([rows[i] for i in mb]) for mb in plan]
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=8192, seed=42)
    lora = LoRARequest("check", 1, args.adapter) if args.adapter else None
    prompts = [t.replace(tok.pad_token, "") for b in batches
               for t in tok.batch_decode(b["student_prompts"], skip_special_tokens=False)]
    sps = [SamplingParams(n=1, temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"],
                          max_tokens=OFFICIAL["max_completion_length"], seed=42 * 1_000_003 + j) for j in range(len(prompts))]
    comps = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sps, lora_request=lora, use_tqdm=False)]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=128,
                                                  target_modules=OFFICIAL["lora_target_modules"]))
    pad, temp = tok.pad_token_id, OFFICIAL["temperature"]
    unc, clp, k = [], [], 0
    with torch.no_grad():
        for b in batches:
            for i in range(len(b["student_prompts"])):
                c = torch.tensor(comps[k], dtype=torch.long)
                k += 1
                s_ids = torch.cat([b["student_prompts"][i], c]).unsqueeze(0).cuda()
                t_ids = torch.cat([b["teacher_prompts"][i], c]).unsqueeze(0).cuda()
                lab = c.clone()
                lab[lab == pad] = -100
                lab = lab.unsqueeze(0).cuda()
                L = len(c)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = model(input_ids=s_ids, attention_mask=(s_ids != pad).long(), logits_to_keep=L + 1).logits[:, :-1]
                    with model.disable_adapter():
                        t = model(input_ids=t_ids, attention_mask=(t_ids != pad).long(), logits_to_keep=L + 1).logits[:, :-1]
                    u = generalized_jsd_loss(s, t, labels=lab, beta=0, temperature=temp, reduction="none", token_clip=None)
                    cl = generalized_jsd_loss(s, t, labels=lab, beta=0, temperature=temp, reduction="none",
                                              token_clip=OFFICIAL["jsd_token_clip"])
                unc.append(u.float().sum(-1).cpu())
                clp.append(cl.float().sum(-1).cpu())
                del s, t, u, cl
    U, C = torch.cat(unc), torch.cat(clp)
    res = {"setting": args.label, "variant": VARIANT, "adapter": args.adapter or "none (LoRA B=0 -> student = base)",
           "n_samples": k, "n_tokens": int(U.numel()),
           "unclipped_per_token": {"min": float(U.min()), "mean": float(U.mean()), "max": float(U.max()),
                                   "frac_negative(<-1e-4)": float((U < -1e-4).float().mean())},
           "clipped_per_token": {"min": float(C.min()), "mean": float(C.mean()), "max": float(C.max()),
                                 "frac_negative(<-1e-4)": float((C < -1e-4).float().mean())},
           "loss_unclipped (official batchmean)": float(U.mean()), "loss_clipped (official batchmean)": float(C.mean())}
    p = HERE / "pilot_r2_out" / "loss_check.json"
    p.parent.mkdir(exist_ok=True)
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres[args.label] = res
    p.write_text(json.dumps(allres, indent=2))
    log.info(f"LOSSCHECK {json.dumps(res)}")
    if float(U.min()) < -1e-4:
        log.error("UNCLIPPED KL IS NEGATIVE -> BUG; stopping")
        raise SystemExit(2)


def cmd_selected(args):
    """Print the DEV-selected checkpoint name (highest DEV pass@12, ties -> earlier step) if all DEV evals exist."""
    names = [f"step{s}" for s in range(SAVE_EVERY, TRAIN_STEPS + 1, SAVE_EVERY)]
    if args.names != ["base"]:          # restrict selection to the checkpoints given (e.g. R3 fallback 20/40/60)
        names = args.names
    dev = {}
    for nm in names:
        p = OUT / "eval" / f"{nm}_dev.jsonl"
        if not p.exists():
            print("NONE")
            return
        dev[nm] = summarize(read_jsonl(p))["pass@12"]
    print(sorted(names, key=lambda n: (-dev[n], int(n[4:])))[0])


def write_config():
    import importlib.metadata as md
    vers = {"python": platform.python_version()}
    for m in ("torch", "vllm", "transformers", "peft", "datasets", "math-verify"):
        try:
            vers[m] = md.version(m)
        except Exception:
            vers[m] = "?"
    cfg = {"model": MODEL, "language": LANG, "official": OFFICIAL, "ours": {
        "gpus": 1, "micro_batch": MICRO_BS, "grad_accum": GA, "effective_batch": MICRO_BS * GA, "train_steps": TRAIN_STEPS,
        "lr_schedule_horizon": OFFICIAL["max_steps"], "save_every": SAVE_EVERY, "vllm_gpu_memory_utilization": 0.4,
        "attn": "sdpa", "checkpoint_dir": str(CKPT)},
        "versions": vers, "copsd_commit": "4f4410730e9b", "deviations": [
        "1 GPU instead of 4: micro-batch 4 x grad-accum 8 = 32 prompts/step (official 4 GPUs x 4 x 2 = 32). Same per-micro-batch loss normalisation.",
        "Backward is done one sample at a time inside each micro-batch, with the micro-batch padding layout and the micro-batch token count as denominator: identical loss/gradient to the official micro-batch of 4, lower peak memory.",
        "Completions are not right-padded to 2048 before the forward pass (official pads every completion to max_completion_length); pads are masked and causal, so the loss is unchanged.",
        "Steps: 60 = min(official 100, 60). LR schedule kept on the official 100-step linear horizon, so steps 1-60 see the same LR as the official run.",
        "Checkpoints every 10 steps (official save_steps 5); DEV evaluation at steps 10..60 (official evaluates 5..50).",
        "Own implementation of the OPSD training loop (official: TRL 0.26 GOLD/SFTTrainer + accelerate + DeepSpeed + vLLM 0.11 colocate). "
        "Collator, prompts, think prefix and generalized_jsd_loss are imported/copied from the official repo.",
        "Generation: all 32 prompts of a step in one vLLM call (official: one call per micro-batch, same weights since no optimizer step in between).",
        "Student weights reach vLLM as a LoRA adapter (vLLM LoRA, saved after every optimizer step) instead of merge_adapter + load_weights; numerically equivalent up to bf16 rounding.",
        "Data order: per-epoch torch.randperm(seed 42+epoch), micro-batches of 4, optimizer step every 8 micro-batches or at epoch end (HF-like); not the identical sample order of the official run.",
        "Attention: PyTorch SDPA (official: flash_attention_2). Same right-padded prompt layout with masked pads.",
        "Software: torch 2.14 / transformers 5.20 / vLLM 0.33 / peft 0.21 (official pins torch 2.8 / transformers 4.57.1 / vLLM 0.11 / peft 0.17.1).",
        "Training data field mapping: problem_translated -> problem_swa (released HF data vs official script's expected column).",
        "Divergence: official flags give beta=0 => F.kl_div(student_logp, teacher_logp) = KL(teacher || student), i.e. FORWARD KL over the full vocabulary "
        "(temperature 1.1 applied to both logits, per-(token,vocab) clip at 0.05). The task description said reverse KL; we follow the code.",
        "Evaluation sampling uses vLLM seed 42 (official: unseeded) and max_model_len 4096 (official 40960; generations are capped at 1024 new tokens anyway).",
    ]}
    cfg["variant"] = VARIANT
    if VARIANT == "r3":
        cfg["r3"] = {"eval": eval_cfg(), "max_model_len": eval_cfg()["max_new_tokens"] + 2048,
                     "checkpoints": str(CKPT), "training": "none (Pilot R adapters reused)"}
        cfg["deviations"] += ["R3: evaluation only. Same Pilot R checkpoints, split and eval code; max_new_tokens 4096 "
                              "(COPSD paper main setting), vLLM max_model_len 6144.",
                              "R3 budget fallback (task rule): base DEV took 845 s of generation (+~3 min load) at 4096 tokens, "
                              "so the full plan (7 DEV + 2 TEST evals, ~3.8 h; ~6.5 h with ORACLE TEST evals of every checkpoint) "
                              "exceeds 2 h of GPU. DEV is evaluated for base and steps 20, 40, 60 only; selection is among "
                              "steps 20/40/60; TEST is run for base and the selected checkpoint only, so the ORACLE row covers "
                              "only checkpoints evaluated on TEST (not a full oracle)."]
    if VARIANT == "r2":
        cfg["r2"] = {"eval": EVAL_R2, "train_rollouts": "enable_thinking=False; sampling unchanged (T 1.1, top_p 0.95, top_k 20, max 2048)",
                     "student_user_instruction": r2_instruction(),
                     "assistant_prefix": "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n (official collator non-thinking branch)"}
        cfg["deviations"] += [
            "R2: thinking OFF for training rollouts (official collator student_enable_thinking=False branch: empty think block) and evaluation.",
            "R2: the Swahili-reasoning request moves from the think prefix into the user prompt, using COPSD's own Swahili "
            "'teacher_final_instruction' text (\"...fikiri hatua kwa hatua kwa Kiswahili...\") for student training prompts and evaluation.",
            "R2: teacher also gets the non-thinking assistant prefix (official collator would give the teacher '<think>' + think prefix "
            "even for a non-thinking student); chosen so the teacher scores the completion in the mode it was generated in.",
            "R2: evaluation uses the Qwen3 chat template with enable_thinking=False (empty think block), n=12, T=0.7, top_p 0.8, top_k 20, "
            "max_new_tokens 2048 (COPSD's own non-thinking eval prompt ends at '<|im_start|>assistant\\n' without the empty think block).",
            "R2: same split, data, seed, LoRA/optimizer settings and 60 steps as Pilot R; only base and the DEV-selected checkpoint are run on TEST.",
        ]
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "train", "eval", "report", "config", "selected", "losscheck"])
    ap.add_argument("--variant", choices=["r", "r2", "r3"], default="r")
    ap.add_argument("--adapter", default=None, help="losscheck: LoRA adapter dir for the student")
    ap.add_argument("--label", default="check", help="losscheck: key in loss_check.json")
    ap.add_argument("--n_microbatches", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--stop_after", type=int, default=0, help="run at most this many steps in this invocation")
    ap.add_argument("--no_save", action="store_true", help="timing run: do not write checkpoints")
    ap.add_argument("--names", nargs="+", default=["base"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    set_variant(args.variant)
    if args.cmd == "selected":
        return cmd_selected(args)
    setup_logging()
    log.info(f"=== pilot_{args.variant} {args.cmd} argv {sys.argv} host {platform.node()} job {os.environ.get('SLURM_JOB_ID')}")
    {"split": cmd_split, "train": cmd_train, "eval": cmd_eval, "report": cmd_report, "config": lambda a: write_config(),
     "losscheck": cmd_losscheck}[args.cmd](args)


if __name__ == "__main__":
    main()
