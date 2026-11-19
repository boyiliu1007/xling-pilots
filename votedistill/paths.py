"""Machine-specific locations. Defaults live under /data/user_data/$USER; override with environment variables."""
import getpass
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SPLIT_FILE = RESULTS / "splits" / "afrimgsm_swa_dev50_test200.json"

USER_DATA = Path(os.environ.get("USER_DATA", f"/data/user_data/{getpass.getuser()}"))
# Checkout of https://github.com/cisnlp/COPSD (commit 4f4410730e9b): data collator, language config, eval helpers.
COPSD_DIR = Path(os.environ.get("COPSD_DIR", USER_DATA / "ext" / "COPSD"))
COPSD_COMMIT = "4f4410730e9b"
# LoRA checkpoints are large and live outside the repo.
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", USER_DATA / "votedistill_ckpt"))
# Downloaded resources that are not Hugging Face models or datasets (SDRRL: FreeDict dictionaries, MAFAND parallel text).
CACHE = Path(os.environ.get("VOTEDISTILL_CACHE", USER_DATA / "votedistill_cache"))

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")

os.environ.setdefault("HF_HOME", str(USER_DATA / "huggingface-cache"))
# FlashInfer's sampler JIT-compiles on first use and needs ninja/nvcc; use vLLM's PyTorch top-k/top-p sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
