"""PASMR length check (before the SRL run): finish rates and lengths of Swahili rollouts and of English pivot answers."""
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from votedistill.copsd.collator import make_collator
from votedistill.copsd.config import OFFICIAL
from votedistill.exp2.screen import _prompt_ids
from votedistill.pasmr.srl import rl_rows, en_prompt_ids, extract_pivot
M = "Qwen/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(M)
col = make_collator(tok, OFFICIAL["max_length"], False, False)
rows = rl_rows("gsm8k-mt3k")[:64]
llm = LLM(model=M, dtype="bfloat16", gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64, max_model_len=8704, seed=1)
import sys
pam = LoRARequest("pam", 1, sys.argv[1])


def report(tag, o):
    L = np.array([len(x.outputs[0].token_ids) for x in o])
    fin = np.array([x.outputs[0].finish_reason == "stop" for x in o])
    print(f"RES {tag}: finished(8192) {fin.mean():.0%} | median {np.median(L):.0f} | p90 {np.percentile(L, 90):.0f} | "
          f"within 2048 {(L <= 2048).mean():.0%}, within 4096 {(L <= 4096).mean():.0%}", flush=True)


sw = llm.generate([{"prompt_token_ids": _prompt_ids(col, r, "student")} for r in rows],
                  SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8192, seed=1), lora_request=pam, use_tqdm=False)
report("SW PAM T1.0", sw)
print(f"RES SW pivot found {np.mean([extract_pivot(x.outputs[0].text) is not None for x in sw]):.0%}", flush=True)
ens = [en_prompt_ids(tok, r["problem"]) for r in rows]
for tag, lora, T in [("EN PAM T1.0", pam, 1.0), ("EN base T1.0", None, 1.0), ("EN base T0.6", None, 0.6)]:
    o = llm.generate([{"prompt_token_ids": p} for p in ens],
                     SamplingParams(temperature=T, top_p=1.0 if T == 1.0 else 0.95, max_tokens=8192, seed=1),
                     lora_request=lora, use_tqdm=False)
    report(tag, o)
