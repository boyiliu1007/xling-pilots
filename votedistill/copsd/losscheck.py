"""Sanity check of the loss sign on a few training micro-batches.

Unclipped per-token value = sum_v p_t (log p_t - log p_s) = KL(teacher || student) >= 0, so a negative value is a bug.
The official per-(token, vocab) clip at 0.05 can make the clipped value negative; that is expected.
"""
import json
import logging
from pathlib import Path

from ..data import load_copsd_train
from ..gpu import wait_for_gpu
from ..paths import MODEL
from .collator import make_collator, prepare_rows
from .config import OFFICIAL, SEED
from .loss import generalized_jsd_loss
from .train import data_schedule

log = logging.getLogger("votedistill.copsd.losscheck")


def loss_check(run_dir, label, adapter=None, n_microbatches=2, lang="swa"):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    wait_for_gpu()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    collator = make_collator(tok, OFFICIAL["max_length"])
    rows = prepare_rows(load_copsd_train(lang), lang)
    plan = data_schedule(len(rows), OFFICIAL["max_steps"])[0]["micro_batches"][:n_microbatches]
    batches = [collator([rows[i] for i in mb]) for mb in plan]
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=8192, seed=SEED)
    lora = LoRARequest("check", 1, adapter) if adapter else None
    prompts = [t.replace(tok.pad_token, "") for b in batches
               for t in tok.batch_decode(b["student_prompts"], skip_special_tokens=False)]
    sps = [SamplingParams(n=1, temperature=OFFICIAL["temperature"], top_p=OFFICIAL["top_p"], top_k=OFFICIAL["top_k"],
                          max_tokens=OFFICIAL["max_completion_length"], seed=SEED * 1_000_003 + j) for j in range(len(prompts))]
    comps = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sps, lora_request=lora, use_tqdm=False)]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=OFFICIAL["lora_r"], lora_alpha=OFFICIAL["lora_alpha"],
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
    res = {"adapter": adapter or "none (LoRA B=0 -> student = base)", "n_samples": k, "n_tokens": int(U.numel()),
           "unclipped_per_token": {"min": float(U.min()), "mean": float(U.mean()), "max": float(U.max()),
                                   "frac_negative(<-1e-4)": float((U < -1e-4).float().mean())},
           "clipped_per_token": {"min": float(C.min()), "mean": float(C.mean()), "max": float(C.max()),
                                 "frac_negative(<-1e-4)": float((C < -1e-4).float().mean())},
           "loss_unclipped (official batchmean)": float(U.mean()), "loss_clipped (official batchmean)": float(C.mean())}
    p = Path(run_dir) / "loss_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres[label] = res
    p.write_text(json.dumps(allres, indent=2))
    log.info(f"LOSSCHECK {json.dumps(res)}")
    if float(U.min()) < -1e-4:
        raise SystemExit("UNCLIPPED KL IS NEGATIVE -> bug in the loss or the prompt/label alignment")
