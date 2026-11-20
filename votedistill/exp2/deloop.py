"""De-looped teacher for COPSD (option 3; PREREGISTRATION 2026-11-16 entry).

A penalty-decoded self-SFT checkpoint (Jeff's method, rerun here in a separate results tree, VOTEDISTILL_RESULTS) has
removed most of the base model's loops from its weights. Its LoRA is merged into the base weights and saved as a new
model directory; running COPSD with BASE_MODEL=<that directory> makes both the student's starting point and the teacher
(the same weights, LoRA disabled) the de-looped model, so the teacher's raw distribution needs no penalty.
The English votes are the base model's (copied into the new model's results tree); the de-looped model is used only
as teacher / student, not to re-vote.
"""
import shutil
from pathlib import Path


def merge(adapter_dir, out_dir, base_model="Qwen/Qwen3-1.7B"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(out_dir)
    if (out / "MERGED").exists():
        return out
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out))
    (out / "MERGED").write_text(f"base {base_model}\nadapter {adapter_dir}\n")
    return out


def setup_tree(model_dir, results_root):
    """Copy the base model's English votes into results/exp2_<model tag>/consensus (what training_rows reads)."""
    src = Path(results_root) / "exp2" / "consensus"
    dst = Path(results_root) / f"exp2_{Path(model_dir).name.lower()}" / "consensus"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("copsd_en_thinking.jsonl", "gsm8k_en_thinking.jsonl"):
        if not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    return dst
