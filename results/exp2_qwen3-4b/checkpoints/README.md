# Exp2 checkpoints (LoRA adapters)

Final (step 300) adapters of the main method **D** (`votelidref`), seeds 43 and 44.
Base model: `Qwen/Qwen3-4B` (thinking mode). LoRA r64 / alpha 128. GRPO, lr 5e-6, 16 questions x 8 rollouts per step,
300 steps, trained on gsm8k-mt3k (GSM8K machine-translated to Swahili). Reward: matches the model's own English
majority vote + Swahili language check + number overlap with its own English solution (no gold labels).
AfriMGSM-swa test Swahili accuracy: s43 26.2, s44 25.0.

The `.safetensors` files are stored with Git LFS: run `git lfs install` then `git lfs pull` after cloning.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "results/exp2_qwen3-4b/checkpoints/grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300__s43/step-300/adapter")
```
