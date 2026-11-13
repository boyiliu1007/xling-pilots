"""Where and how COPSD's per-(token, vocab) loss clip binds: a follow-up to the D screen (screen.py).

COPSD's loss (copsd/loss.py, official flags) is forward KL(teacher || student) at T = 1.1, summed over the vocabulary
after clamping every summand from above:
    term_v = p_t(v) [log p_t(v) - log p_s(v)],   loss_t = sum_v min(term_v, c),   c = 0.05
A clamped summand is constant, so its gradient is zero. With S_c = {v : term_v > c} (the clamped entries), the
gradient with respect to the student's temperature-scaled logits z at one position is
    full (no clip):  g   = p_s - p_t
    clipped:         g_c = -p_t * 1[v not in S_c] + m_c * p_s,   m_c = sum_{v not in S_c} p_t(v)
(up to the common factor 1/T). This module re-scores the stored screen rollouts (base model = the step-0 student)
under each teacher context and reports, per context and clip value c:
  any_clipped      share of positions with at least one clamped entry (the exact quantity; the screen report's
                   "tokens with fkl > clip" was a per-position proxy)
  n_clipped        mean number of clamped vocabulary entries per position
  top1_clipped     share of positions whose teacher top-1 token is clamped
  sampled_clipped  share of positions whose sampled (student) token is clamped
  loss_kept        sum of the clipped loss / sum of the unclipped KL
  grad_norm_ratio  sqrt(sum |g_c|^2 / sum |g|^2)                     (pooled over positions)
  grad_cos         sum <g_c, g> / sqrt(sum |g_c|^2 sum |g|^2)         (pooled)
  top1_pull        mean of -g[a] (a = teacher top-1; > 0 means "raise a"), full vs clipped
  top1_reversed    share of positions where the full gradient raises a but the clipped gradient lowers it
and, for pairs of contexts (A, B), how much of the gradient *difference* between them survives the clip:
  pair_diff_kept   sqrt(sum |g_c(A) - g_c(B)|^2 / sum |g(A) - g(B)|^2)
also by relative position in the rollout (deciles). Gradients are with respect to logits, not LoRA parameters: they
describe what each position asks the student's next-token distribution to do.

Outputs results/exp2/screen/<name>/clipstats.npz (per-position arrays) and clipstats.txt.
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

log = logging.getLogger("votedistill.exp2.clipstats")
CLIPS = (0.05, 0.2, 0.5)
CONTEXTS = ("L1", "L2-gold", "L3-gold", "L3-shuffled-gold")
PAIRS = (("L3-gold", "L3-shuffled-gold"), ("L3-gold", "L1"), ("L2-gold", "L1"), ("L3-gold", "L2-gold"))
CHUNK = 256
FIELDS = ("any", "n", "top1", "samp", "loss_c", "dot", "gc2", "pull_c")


def _cf(c):
    return f"{c:g}"


def clipstats(name="copsd_k2", contexts=CONTEXTS, clips=CLIPS, samples=None, limit=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sd = screen_dir(name)
    cfg = read_json(sd / "config.json")
    recs = [r for r in read_jsonl(sd / "rollouts.jsonl") if r["completion_ids"] and (samples is None or r["sample"] in samples)][:limit]
    T = OFFICIAL["temperature"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = prepare_rows(load_copsd_train(cfg["lang"]), cfg["lang"])
    idx = select_problems(len(rows), cfg["n_problems"])
    tprompts = {}
    for c in contexts:
        trows, flags = teacher_rows(rows, c)
        col = make_collator(tok, OFFICIAL["max_length"], **flags)
        tprompts[c] = {i: _prompt_ids(col, trows[i], "teacher") for i in idx}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    pairs = [p for p in PAIRS if p[0] in contexts and p[1] in contexts]

    out = {"rollout": [], "rel_pos": [], "finished": []}
    for c in contexts:
        out[f"{c}|fkl"], out[f"{c}|g2"], out[f"{c}|pull"] = [], [], []
        for cv in clips:
            for f in FIELDS:
                out[f"{c}|{_cf(cv)}|{f}"] = []
    for a, b in pairs:
        out[f"{a}~{b}|full"] = []
        for cv in clips:
            out[f"{a}~{b}|{_cf(cv)}"] = []

    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(recs):
            comp = r["completion_ids"]
            L = len(comp)
            s_log = _completion_logits(model, r["student_prompt_ids"], comp)
            t_logs = {c: _completion_logits(model, tprompts[c][r["problem"]], comp) for c in contexts}
            y = torch.tensor(comp, device="cuda")
            out["rollout"].append(np.full(L, n, np.int32))
            out["rel_pos"].append((np.arange(L) / L).astype(np.float32))
            out["finished"].append(np.full(L, r["finish_reason"] == "stop"))
            for s0 in range(0, L, CHUNK):
                ls = F.log_softmax(s_log[s0:s0 + CHUNK].float() / T, -1)
                ps = ls.exp()
                yy = y[s0:s0 + CHUNK, None]
                g_full, g_clip = {}, {}
                for c in contexts:
                    lt = F.log_softmax(t_logs[c][s0:s0 + CHUNK].float() / T, -1)
                    pt = lt.exp()
                    term = pt * (lt - ls)
                    g = ps - pt
                    a1 = pt.argmax(-1, keepdim=True)
                    out[f"{c}|fkl"].append(term.sum(-1).cpu().numpy())
                    out[f"{c}|g2"].append((g * g).sum(-1).cpu().numpy())
                    out[f"{c}|pull"].append((-g.gather(-1, a1))[:, 0].cpu().numpy())
                    g_full[c] = g
                    for cv in clips:
                        over = term > cv
                        keep = (~over).float()
                        m = (pt * keep).sum(-1, keepdim=True)
                        gc = -pt * keep + m * ps
                        vals = {"any": over.any(-1), "n": over.sum(-1), "top1": over.gather(-1, a1)[:, 0],
                                "samp": over.gather(-1, yy)[:, 0], "loss_c": term.clamp(max=cv).sum(-1),
                                "dot": (gc * g).sum(-1), "gc2": (gc * gc).sum(-1), "pull_c": -gc.gather(-1, a1)[:, 0]}
                        for f in FIELDS:
                            out[f"{c}|{_cf(cv)}|{f}"].append(vals[f].float().cpu().numpy())
                        g_clip[(c, cv)] = gc
                    del lt, pt, term
                for a, b in pairs:
                    d = g_full[a] - g_full[b]
                    out[f"{a}~{b}|full"].append((d * d).sum(-1).cpu().numpy())
                    for cv in clips:
                        d = g_clip[(a, cv)] - g_clip[(b, cv)]
                        out[f"{a}~{b}|{_cf(cv)}"].append((d * d).sum(-1).cpu().numpy())
                del g_full, g_clip, ls, ps
            del s_log, t_logs
            if (n + 1) % 50 == 0:
                log.info(f"{n + 1}/{len(recs)} rollouts ({time.time() - t0:.0f}s)")
    arr = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(sd / "clipstats.npz", **arr)
    write_json(sd / "clipstats_config.json", {"contexts": list(contexts), "clips": list(clips), "pairs": pairs,
                                              "rollouts": len(recs), "samples": samples, "temperature": T,
                                              "doc": __doc__})
    log.info(f"done in {time.time() - t0:.0f}s")
    return clipstats_report(name)


def clipstats_report(name="copsd_k2"):
    sd = screen_dir(name)
    z = dict(np.load(sd / "clipstats.npz"))
    cfg = read_json(sd / "clipstats_config.json")
    ctxs, clips, pairs = cfg["contexts"], cfg["clips"], [tuple(p) for p in cfg["pairs"]]
    dec = np.minimum((z["rel_pos"] * 10).astype(int), 9)
    fin = z["finished"].astype(bool)
    T = [f"Clip diagnostics: {name}, {cfg['rollouts']} rollouts, {len(dec)} positions (base model = step-0 student), "
         f"KL at T={cfg['temperature']}", "Definitions: votedistill/exp2/clipstats.py docstring.", ""]

    def block(mask, title):
        if not mask.any():
            return
        T.append(f"## {title} ({mask.sum()} positions)")
        T.append(f"  {'context':<18} {'clip':>5} {'any_clip':>9} {'n_clip':>7} {'top1_clip':>10} {'samp_clip':>10} "
                 f"{'loss_kept':>10} {'|g_c|/|g|':>10} {'cos':>6} {'pull full':>10} {'pull clip':>10} {'top1_rev':>9}")
        for c in ctxs:
            g2, fkl, pull = z[f"{c}|g2"][mask], z[f"{c}|fkl"][mask], z[f"{c}|pull"][mask]
            for cv in clips:
                k = lambda f: z[f"{c}|{cv:g}|{f}"][mask]
                rev = np.mean((pull > 0) & (k("pull_c") < 0))
                T.append(f"  {c:<18} {cv:>5g} {k('any').mean():9.3f} {k('n').mean():7.2f} {k('top1').mean():10.3f} "
                         f"{k('samp').mean():10.3f} {k('loss_c').sum() / fkl.sum():10.3f} "
                         f"{np.sqrt(k('gc2').sum() / g2.sum()):10.3f} "
                         f"{k('dot').sum() / np.sqrt(k('gc2').sum() * g2.sum()):6.3f} {pull.mean():10.4f} "
                         f"{k('pull_c').mean():10.4f} {rev:9.3f}")
        T.append(f"  gradient difference between contexts that survives the clip, sqrt(sum|Δg_c|^2 / sum|Δg|^2):")
        for a, b in pairs:
            full = z[f"{a}~{b}|full"][mask].sum()
            T.append(f"    {a + ' vs ' + b:<32} " + "  ".join(
                f"clip {cv:g}: {np.sqrt(z[f'{a}~{b}|{cv:g}'][mask].sum() / full):.3f}" for cv in clips)
                + f"   (unclipped |Δg| rms {np.sqrt(full / mask.sum()):.4f})")
        T.append("")

    block(np.ones_like(fin), "all positions")
    block(fin, "finished rollouts only")
    block(~fin, "truncated rollouts only")
    T.append("## by relative position (deciles), official clip 0.05: any_clip / loss_kept / |g_c|/|g| / top1_rev")
    for c in ctxs:
        cells = []
        for d in range(10):
            m = dec == d
            k = lambda f: z[f"{c}|0.05|{f}"][m]
            cells.append(f"{k('any').mean():.2f}/{k('loss_c').sum() / z[f'{c}|fkl'][m].sum():.2f}/"
                         f"{np.sqrt(k('gc2').sum() / z[f'{c}|g2'][m].sum()):.2f}/"
                         f"{np.mean((z[f'{c}|pull'][m] > 0) & (k('pull_c') < 0)):.2f}")
        T.append(f"  {c:<18} " + " ".join(f"{s:<20}" for s in cells))
    T.append("  share of all unclipped KL by decile: " + "  ".join(
        f"{c}: " + ",".join(f"{z[f'{c}|fkl'][dec == d].sum() / z[f'{c}|fkl'].sum():.2f}" for d in range(10))
        for c in ctxs))
    text = "\n".join(T) + "\n"
    (sd / "clipstats.txt").write_text(text)
    return text
