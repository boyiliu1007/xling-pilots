"""COPSD hyperparameters and how this single-GPU reimplementation deviates from the official run."""

# Official COPSD Qwen3-1.7B settings (african_langs_scripts/run_all_opsd_1.7b_train.sh, commit 4f4410730e9b).
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
                weight_decay=0.0, optimizer="AdamW (betas 0.9/0.999, eps 1e-8)", seed=42)

# COPSD paper evaluation protocol (main setting). The official script's own default is 1024 new tokens, which
# truncates most thinking-mode answers; 4096 reproduces the paper's Qwen3-1.7B Swahili base (14.8 pass@12).
EVAL = dict(val_n=12, temperature=1.0, top_p=0.95, top_k=-1, max_new_tokens=4096, enable_thinking=True)

MICRO_BS = 4      # 1 GPU x 4 x 8 = 32 prompts per step = official 4 GPUs x 4 x 2
GA = 8
SAVE_EVERY = 10
SEED = 42


def deviations(max_steps):
    return [
        "1 GPU instead of 4: micro-batch 4 x grad-accum 8 = 32 prompts/step (official 4 GPUs x 4 x 2 = 32). Same per-micro-batch loss normalisation.",
        "Backward is done one sample at a time inside each micro-batch, with the micro-batch token count as denominator: identical loss/gradient to the official micro-batch of 4, lower peak memory.",
        "Completions are not right-padded to 2048 before the forward pass (official pads every completion to max_completion_length); pads are masked and causal, so the loss is unchanged.",
        f"Steps: {max_steps}. The LR schedule keeps the official {OFFICIAL['max_steps']}-step linear horizon, so each step sees the official LR.",
        f"Checkpoints every {SAVE_EVERY} steps (official save_steps {OFFICIAL['save_steps']}).",
        "Own implementation of the OPSD training loop (official: TRL 0.26 GOLD/SFTTrainer + accelerate + DeepSpeed + vLLM 0.11 colocate). "
        "Collator, prompts, think prefix and generalized_jsd_loss are imported/copied from the official repo.",
        "Generation: all 32 prompts of a step in one vLLM call (official: one call per micro-batch, same weights since no optimizer step in between).",
        "Student weights reach vLLM as a LoRA adapter saved after every optimizer step, instead of merge_adapter + load_weights; equivalent up to bf16 rounding.",
        "Data order: per-epoch torch.randperm(seed 42+epoch), micro-batches of 4, optimizer step every 8 micro-batches or at epoch end (HF-like); not the identical sample order of the official run.",
        "Attention: PyTorch SDPA (official: flash_attention_2). Same right-padded prompt layout with masked pads.",
        "Software: torch 2.14 / transformers 5.20 / vLLM 0.33 / peft 0.21 (official pins torch 2.8 / transformers 4.57.1 / vLLM 0.11 / peft 0.17.1).",
        "Training data field mapping: problem_translated -> problem_<lang> (released HF data vs the official script's expected column).",
        "Loss: official flags give beta=0 => F.kl_div(student_logp, teacher_logp) = KL(teacher || student), i.e. FORWARD KL over the full vocabulary "
        "(temperature 1.1 on both logits, per-(token, vocab) clip at 0.05). The clip makes the reported loss go negative; the unclipped KL is never negative.",
        "Evaluation sampling uses vLLM seed 42 (official: unseeded) and max_model_len = max_new_tokens + 2048 (official 40960).",
    ]
