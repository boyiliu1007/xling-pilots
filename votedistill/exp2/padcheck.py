"""How much does COPSD's training layout change the teacher's distributions?

copsd/train.py (like the official trainer) right-pads each teacher prompt to the longest prompt of its micro-batch and
appends the completion after the pads. Pads are masked, but no position ids are passed, so the completion's positions
are shifted by the number of pads. This compares, on the screen's rollouts, the teacher's next-token distribution
under that layout with the exact unpadded one: per completion token KL(exact || padded) and |delta log p(y_t)|.
Micro-batches follow the training schedule's first steps (4 prompts each), so the pad counts are realistic.
"""
import logging

import numpy as np

from ..data import load_copsd_train
from ..paths import MODEL
from ..utils import read_jsonl, write_json
from ..copsd.collator import make_collator, prepare_rows
from ..copsd.config import OFFICIAL
from .levels import teacher_rows
from .screen import screen_dir

log = logging.getLogger("votedistill.exp2.padcheck")


def padcheck(name, context="L3-gold", n_microbatches=25, lang="swa"):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sd = screen_dir(name)
    recs = {}
    for r in read_jsonl(sd / "rollouts.jsonl"):
        recs.setdefault(r["problem"], r)                       # first rollout of each problem
    probs = sorted(recs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = prepare_rows(load_copsd_train(lang), lang)
    trows, flags = teacher_rows(rows, context)
    col = make_collator(tok, OFFICIAL["max_length"], **flags)
    pad = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    kls, dlp, npads = [], [], []
    for b in range(n_microbatches):
        mb = probs[4 * b: 4 * b + 4]
        if len(mb) < 4:
            break
        batch = col([trows[i] for i in mb])
        for j, i in enumerate(mb):
            comp = [t for t in recs[i]["completion_ids"]]
            if not comp:
                continue
            padded = batch["teacher_prompts"][j].tolist()
            exact = [t for t in padded if t != pad]
            npads.append(len(padded) - len(exact))
            out = []
            for prompt, use_mask in ((exact, False), (padded, True)):
                ids = torch.tensor([prompt + comp], device="cuda")
                kw = {"attention_mask": (ids != pad).long()} if use_mask else {}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(model(input_ids=ids, logits_to_keep=len(comp) + 1, **kw).logits[0, :-1].float())
            le, lp = F.log_softmax(out[0], -1), F.log_softmax(out[1], -1)
            kls.append((le.exp() * (le - lp)).sum(-1).cpu().numpy())
            y = torch.tensor(comp, device="cuda")[:, None]
            dlp.append((le.gather(-1, y) - lp.gather(-1, y))[:, 0].abs().cpu().numpy())
            del out, le, lp
    K, Dl, P = np.concatenate(kls), np.concatenate(dlp), np.array(npads)
    nz = P > 0
    per_seq_kl = np.array([k.mean() for k in kls])
    res = {"context": context, "sequences": int(len(P)), "tokens": int(len(K)),
           "pads_per_teacher_prompt": {"mean": float(P.mean()), "max": int(P.max()), "share_with_pads": float(nz.mean())},
           "KL(exact||padded) per token": {"mean": float(K.mean()), "p99": float(np.percentile(K, 99)), "max": float(K.max())},
           "|delta log p(y)| per token": {"mean": float(Dl.mean()), "p99": float(np.percentile(Dl, 99))},
           "mean KL per sequence, with pads vs without": [float(per_seq_kl[nz].mean()) if nz.any() else None,
                                                          float(per_seq_kl[~nz].mean()) if (~nz).any() else None],
           "note": "sequences without pads measure bf16/kernel noise only (same input, mask given vs not)"}
    write_json(sd / f"padcheck_{context}.json", res)
    return res
