"""Machine-specific locations. Override with environment variables instead of editing code."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SPLIT_FILE = RESULTS / "splits" / "afrimgsm_swa_dev50_test200.json"

# Checkout of https://github.com/cisnlp/COPSD (commit 4f4410730e9b): data collator, language config, eval helpers.
COPSD_DIR = Path(os.environ.get("COPSD_DIR", "/data/user_data/boyil2/ext/COPSD"))
# LoRA checkpoints are large and live outside the repo.
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", "/data/user_data/boyil2"))

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")

os.environ.setdefault("HF_HOME", "/data/user_data/boyil2/hf_cache")
# FlashInfer's sampler JIT-compiles on first use and needs ninja/nvcc; use vLLM's PyTorch top-k/top-p sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
