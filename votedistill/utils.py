import importlib.metadata as md
import json
import logging
import platform
import sys
from pathlib import Path


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def write_jsonl(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def versions(packages=("torch", "vllm", "transformers", "datasets", "peft", "math-verify")):
    v = {"python": platform.python_version()}
    for p in packages:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = "not installed"
    return v


def setup_logging(log_file=None):
    """Log to stdout, and also to log_file if given (append)."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    for name in ("httpx", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
