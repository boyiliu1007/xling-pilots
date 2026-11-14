"""Why does clipped distillation teach the student to stop, and why does faithful (unclipped) distillation make it
worse? Token-level diagnostic on the base model's own training-prompt rollouts (screen copsd_k2).

Per completion position t (sampled token y), with student s = base model on the Swahili prompt and teacher c = base
model with a teacher context, both at the loss temperature T = 1.1:
  loop      the 8-token window ending at t already occurred earlier in the rollout (y continues a verbatim repeat)
  push(v)   change the loss asks for in the student's logit of token v (= minus the gradient wrt z_v, up to 1/T):
            full:    p_c(v) - p_s(v)
            clipped: p_c(v) * keep(v) - m * p_s(v),  keep(v) = [p_c(v) log(p_c(v)/p_s(v)) <= clip],  m = sum p_c * keep
  evaluated for v = y (continuing the text, i.e. the loop when loop = 1), v = </think> and v = <|im_end|>.
Positive push = raise that token. Reported as means by position type (loop / non-loop, thinking / after </think>).
Also, for trained student checkpoints (student prompt + LoRA), the realised change log p_ckpt(v) - log p_base(v) on the
same text, for the same tokens: what the clipped and unclipped runs actually learned.
"""
import logging
import time

import numpy as np

from ..copsd.collator import make_collator, prepare_rows
from ..copsd.config import OFFICIAL
from ..data import load_copsd_train
from ..paths import MODEL
from ..utils import read_json, read_jsonl, write_json
from .levels import teacher_rows
from .screen import _completion_logits, _prompt_ids, screen_dir, select_problems

log = logging.getLogger("votedistill.exp2.mechanism")
CONTEXTS = ("L1", "L3-gold")
CKPTS = {"clip": ("L3-gold__copsd__exact__s42", 100), "noclip": ("L3-gold__copsd__exact__noclip__s42", 100),
         "clip_L1": ("L1__copsd__exact__s42", 100), "noclip_L1": ("L1__copsd__exact__noclip__s42", 100)}
LOOP_N = 8
CLIP = 0.05


def loop_flags(ids, n=LOOP_N):
    seen, out = set(), np.zeros(len(ids), bool)
    for t in range(n - 1, len(ids)):
        k = tuple(ids[t - n + 1:t + 1])
        out[t] = k in seen
        seen.add(k)
    return out


def mechanism(name="copsd_k2", samples=(0,), contexts=CONTEXTS, ckpts=CKPTS, limit=None):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .runs import paths
    sd = screen_dir(name)
    cfg = read_json(sd / "config.json")
    recs = [r for r in read_jsonl(sd / "rollouts.jsonl") if r["completion_ids"] and r["sample"] in samples][:limit]
    tok = AutoTokenizer.from_pretrained(MODEL)
    END_THINK, IM_END = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    rows = prepare_rows(load_copsd_train(cfg["lang"]), cfg["lang"])
    idx = select_problems(len(rows), cfg["n_problems"])
    tprompts = {}
    for c in contexts:
        trows, flags = teacher_rows(rows, c)
        col = make_collator(tok, OFFICIAL["max_length"], **flags)
        tprompts[c] = {i: _prompt_ids(col, trows[i], "teacher") for i in idx}
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    model = base
    adapters = []
    for k, (run, step) in ckpts.items():
        a = paths(run)[1] / f"step-{step}" / "adapter"
        if not a.exists():
            log.warning(f"skip checkpoint {k}: {a} missing")
            continue
        model = PeftModel.from_pretrained(model, str(a), adapter_name=k) if model is base else model
        if k not in model.peft_config:
            model.load_adapter(str(a), adapter_name=k)
        adapters.append(k)
    model.eval()
    T = OFFICIAL["temperature"]
    out = {k: [] for k in ("rollout", "loop", "in_think", "rel_pos", "finished", "ps_y", "ent_s")}
    for c in contexts:
        for v in ("y", "end", "im"):
            out[f"{c}|p_t_{v}"] = []
            for kind in ("full", "clip"):
                out[f"{c}|{kind}|{v}"] = []
        out[f"{c}|frac_kept"] = []
        out[f"{c}|ent_t"] = []
    for k in adapters:
        for v in ("y", "end", "im", "esc", "ent"):
            out[f"ck|{k}|{v}"] = []
    t0 = time.time()

    def logits(prompt, comp, adapter=None):
        if adapter is None:
            if model is base:
                return _completion_logits(model, prompt, comp)
            with model.disable_adapter():
                return _completion_logits(model, prompt, comp)
        model.set_adapter(adapter)
        return _completion_logits(model, prompt, comp)

    with torch.no_grad():
        for n, r in enumerate(recs):
            comp = r["completion_ids"]
            L = len(comp)
            y = torch.tensor(comp, device="cuda")[:, None]
            et = comp.index(END_THINK) if END_THINK in comp else L
            out["rollout"].append(np.full(L, n, np.int32))
            out["loop"].append(loop_flags(comp))
            out["in_think"].append(np.arange(L) < et)
            out["rel_pos"].append((np.arange(L) / L).astype(np.float32))
            out["finished"].append(np.full(L, r["finish_reason"] == "stop"))
            s_raw = logits(r["student_prompt_ids"], comp)
            ls = F.log_softmax(s_raw.float() / T, -1)
            ps = ls.exp()
            ls1 = F.log_softmax(s_raw.float(), -1)
            out["ps_y"].append(ls1.gather(-1, y)[:, 0].exp().cpu().numpy())
            out["ent_s"].append((-(ls1.exp() * ls1).sum(-1)).cpu().numpy())
            ids = {"y": y, "end": torch.full_like(y, END_THINK), "im": torch.full_like(y, IM_END)}
            for c in contexts:
                t_raw = logits(tprompts[c][r["problem"]], comp).float()
                lt1 = F.log_softmax(t_raw, -1)
                out[f"{c}|ent_t"].append((-(lt1.exp() * lt1).sum(-1)).cpu().numpy())
                lt = F.log_softmax(t_raw / T, -1)
                pt = lt.exp()
                del t_raw, lt1
                keep = (pt * (lt - ls) <= CLIP).float()
                m = (pt * keep).sum(-1, keepdim=True)
                full = pt - ps
                clip = pt * keep - m * ps
                out[f"{c}|frac_kept"].append(m[:, 0].cpu().numpy())
                for v, ix in ids.items():
                    out[f"{c}|p_t_{v}"].append(pt.gather(-1, ix)[:, 0].cpu().numpy())
                    out[f"{c}|full|{v}"].append(full.gather(-1, ix)[:, 0].cpu().numpy())
                    out[f"{c}|clip|{v}"].append(clip.gather(-1, ix)[:, 0].cpu().numpy())
                del lt, pt, keep, full, clip
            for k in adapters:
                lk = F.log_softmax(logits(r["student_prompt_ids"], comp, k).float(), -1)
                for v, ix in ids.items():
                    out[f"ck|{k}|{v}"].append((lk.gather(-1, ix) - ls1.gather(-1, ix))[:, 0].cpu().numpy())
                # loop escape: change in log(1 - p(y)); entropy change (T = 1)
                pk_y, pb_y = lk.gather(-1, y)[:, 0].exp(), ls1.gather(-1, y)[:, 0].exp()
                out[f"ck|{k}|esc"].append((torch.log1p(-pk_y.clamp(max=1 - 1e-7)) - torch.log1p(-pb_y.clamp(max=1 - 1e-7))).cpu().numpy())
                out[f"ck|{k}|ent"].append(((-(lk.exp() * lk).sum(-1)) - (-(ls1.exp() * ls1).sum(-1))).cpu().numpy())
                del lk
            del s_raw, ls, ps, ls1
            if (n + 1) % 50 == 0:
                log.info(f"{n + 1}/{len(recs)} rollouts ({time.time() - t0:.0f}s)")
    arr = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(sd / "mechanism.npz", **arr)
    write_json(sd / "mechanism_config.json", {"contexts": list(contexts), "checkpoints": {k: ckpts[k] for k in adapters},
                                              "rollouts": len(recs), "samples": list(samples), "clip": CLIP,
                                              "temperature": T, "loop_n": LOOP_N, "doc": __doc__})
    return mechanism_report(name)


def mechanism_report(name="copsd_k2"):
    sd = screen_dir(name)
    z = dict(np.load(sd / "mechanism.npz"))
    cfg = read_json(sd / "mechanism_config.json")
    loop, think, fin = z["loop"], z["in_think"], z["finished"]
    groups = {"loop (verbatim repeat), thinking": loop & think, "non-loop, thinking": ~loop & think,
              "after </think>": ~think}
    T = [f"Mechanism diagnostic: {name}, {cfg['rollouts']} rollouts, {len(loop)} positions; loss T={cfg['temperature']}, "
         f"clip {cfg['clip']}. push = change the loss asks for in the student's logit (positive = raise).", "",
         f"  share of positions: loop {loop.mean():.3f}; thinking {think.mean():.3f}; student p(y) median "
         f"{np.median(z['ps_y']):.3f} (loop {np.median(z['ps_y'][loop]):.3f}, non-loop {np.median(z['ps_y'][~loop]):.3f}); "
         f"student entropy mean loop {z['ent_s'][loop].mean():.3f}, non-loop {z['ent_s'][~loop].mean():.3f}", ""]
    for c in cfg["contexts"]:
        T.append(f"## teacher {c}")
        T.append(f"  {'positions':<36} {'n':>8} {'p_t(y)':>7} {'p_s(y)':>7} | {'push y full':>11} {'clip':>9} | "
                 f"{'p_t(</think>)':>13} {'push </think> full':>18} {'clip':>9} | {'push <im_end> full':>18} {'clip':>9} | {'m':>5}")
        for g, mk in groups.items():
            if not mk.any():
                continue
            f = lambda k: z[f"{c}|{k}"][mk].mean()
            T.append(f"  {g:<36} {mk.sum():8d} {f('p_t_y'):7.3f} {z['ps_y'][mk].mean():7.3f} | {f('full|y'):+11.5f} "
                     f"{f('clip|y'):+9.5f} | {f('p_t_end'):13.5f} {f('full|end'):+18.6f} {f('clip|end'):+9.6f} | "
                     f"{f('full|im'):+18.6f} {f('clip|im'):+9.6f} | {f('frac_kept'):5.3f}")
        # sum of pushes (total direction over all positions of a type): sign tells what training accumulates
        T.append("  entropy (T=1): " + "; ".join(f"{g.split(',')[0]}: student {z['ent_s'][mk].mean():.3f} teacher "
                                                  f"{z[f'{c}|ent_t'][mk].mean():.3f}" for g, mk in groups.items() if mk.any()))
        T.append(f"  totals over positions (sum of pushes): " + "; ".join(
            f"{g.split(',')[0]}: y {z[f'{c}|full|y'][mk].sum():+.1f}/{z[f'{c}|clip|y'][mk].sum():+.1f}, "
            f"</think> {z[f'{c}|full|end'][mk].sum():+.2f}/{z[f'{c}|clip|end'][mk].sum():+.2f}"
            for g, mk in groups.items() if mk.any()) + "   (full/clip)")
        T.append("")
    ck = sorted({k.split("|")[1] for k in z if k.startswith("ck|")})
    if ck:
        T.append("## what trained checkpoints learned: mean change in log p vs the base student on the same text")
        T.append(f"  {'checkpoint':<14} " + " ".join(f"{g[:22] + ' y':>26} {'</think>':>9} {'<im_end>':>9}" for g in groups))
        for k in ck:
            T.append(f"  {k:<14} " + " ".join(
                f"{z[f'ck|{k}|y'][mk].mean():+26.3f} {z[f'ck|{k}|end'][mk].mean():+9.3f} {z[f'ck|{k}|im'][mk].mean():+9.3f}"
                for g, mk in groups.items()))
        T.append("")
        T.append("  confidence: Δ entropy (nats) and loop escape Δlog(1 - p(y)) (loop positions; > 0 = more likely to leave the loop)")
        for k in ck:
            T.append(f"  {k:<14} " + "  ".join(f"{g[:22]}: Δent {z[f'ck|{k}|ent'][mk].mean():+.3f}" for g, mk in groups.items())
                     + f"   loop escape {z[f'ck|{k}|esc'][loop & think].mean():+.3f} (median {np.median(z[f'ck|{k}|esc'][loop & think]):+.3f})")
        T.append("")
        T.append("  by relative position (deciles), thinking positions only: mean Δlog p(</think>)")
        dec = np.minimum((z["rel_pos"] * 10).astype(int), 9)
        for k in ck:
            T.append(f"  {k:<14} " + " ".join(f"{z[f'ck|{k}|end'][(dec == d) & think].mean():+6.2f}" for d in range(10)))
    text = "\n".join(T) + "\n"
    (sd / "mechanism.txt").write_text(text)
    return text
