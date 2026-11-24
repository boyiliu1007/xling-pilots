import numpy as np
import pytest

from votedistill.pasmr import PIVOT_CLOSE, PIVOT_OPEN
from votedistill.pasmr.data import pam_target, split
from votedistill.pasmr.srl import (R_AGREE, R_DISAGREE, R_FORMAT, extract_pivot, parse_variant, pasmr_reward,
                                   token_returns)

PIV = f" {PIVOT_OPEN}Julie reads 12 pages. How many are left?{PIVOT_CLOSE}\nJulie alisoma kurasa 12."


def sw(answer, pivot=True):
    return (PIV if pivot else " Julie alisoma kurasa 12.") + f"\n</think>\n\nJibu ni $\\boxed{{{answer}}}$."


def en(answer):
    return f" Okay.\n</think>\n\nThe answer is $\\boxed{{{answer}}}$."


def test_extract_pivot():
    assert extract_pivot(PIV) == "Julie reads 12 pages. How many are left?"
    assert extract_pivot("no pivot here") is None
    assert extract_pivot(f"{PIVOT_OPEN}   {PIVOT_CLOSE}") is None
    assert extract_pivot(f"{PIVOT_OPEN}first{PIVOT_CLOSE} {PIVOT_OPEN}second{PIVOT_CLOSE}") == "first"


def test_reward_cases():
    assert pasmr_reward(sw(42), "stop", en(42), "stop")[0] == R_AGREE
    assert pasmr_reward(sw(42), "stop", en("42.0"), "stop")[0] == R_AGREE           # numeric equivalence
    assert pasmr_reward(sw(42), "stop", en(41), "stop")[0] == R_DISAGREE
    assert pasmr_reward(sw(42), "stop", en(42), "length")[0] == R_DISAGREE          # English unfinished: no A_En
    assert pasmr_reward(sw(42), "stop", None, None)[0] == R_DISAGREE
    assert pasmr_reward(sw(42, pivot=False), "stop", en(42), "stop")[0] == R_FORMAT  # no pivot
    assert pasmr_reward(sw(42), "length", en(42), "stop")[0] == R_FORMAT            # Swahili unfinished
    f, s = pasmr_reward(sw(42), "stop", en(42), "stop")
    assert s["agree"] and s["pivot_ok"] and not s["format_error"]


def test_token_returns_gamma1_and_kl():
    kl = np.array([0.1, 0.0, -0.2, 0.3])
    g = token_returns(1.0, kl, beta=0.01, gamma=1.0)
    expect = 1.0 - 0.01 * np.cumsum(kl[::-1])[::-1]       # G_i = F - beta * sum_{k>=i} kl_k
    assert np.allclose(g, expect)
    g9 = token_returns(1.0, np.zeros(3), beta=0.01, gamma=0.5)
    assert np.allclose(g9, [0.25, 0.5, 1.0])
    assert len(token_returns(1.0, np.zeros(0))) == 0


def test_split_disjoint_and_deterministic():
    a, b = split(range(2805))
    assert set(a).isdisjoint(b) and len(a) + len(b) == 2805 and len(a) == 1402
    assert (a, b) == split(range(2805))


def test_pam_target_format():
    t = pam_target("Kwa ombi, nitaanza kufikiria kwa Kiswahili.", "How many?", "Hatua 1.", "Jibu: $\\boxed{3}$")
    assert t.startswith("<think>\nKwa ombi, nitaanza kufikiria kwa Kiswahili. <translation_start>How many?<translation_end>\n")
    assert t.endswith("\n</think>\n\nJibu: $\\boxed{3}$")
    assert extract_pivot(t) == "How many?"


def test_variants_and_run_names():
    assert parse_variant("") == {"prompts_per_step": 128, "steps": 300}
    assert parse_variant("q4-steps2") == {"prompts_per_step": 4, "steps": 2}
    from votedistill.exp2.runs import final_step, parse_name
    assert parse_name("pasmr__gsm8k-mt3k__exact__s42")[0] == "pasmr"
    assert parse_name("pasmr__gsm8k-mt3k__exact__q4-steps2__s1")[0] == "pasmr"
    assert parse_name("pasmr-pam__gsm8k-mt3k__exact__s42")[0] == "pasmr-pam"
    assert final_step("pasmr__gsm8k-mt3k__exact__s42") == 300
    with pytest.raises(SystemExit):
        parse_name("pasmr__gsm8k-mt3k__exact__noclip__s42")


def test_grpo_pam_start_variant():
    from votedistill.exp2.grpo import INITS, parse_variant
    v = parse_variant("votelidref-lr5e-6-q16-steps300-initpasmrpam-len4096-rlhalf")
    assert v["kind"] == "votelidref" and v["init"] == "pasmrpam" and v["max_tokens"] == 4096 and v["rlhalf"]
    assert INITS["pasmrpam"]("gsm8k-mt3k", 42) == "pasmr-pam__gsm8k-mt3k__exact__s42"
    d = parse_variant("votelidref-lr5e-6-q16-steps300")                      # D unchanged
    assert d["max_tokens"] == 2048 and not d["rlhalf"] and d["init"] is None
    assert parse_variant("votelidref-lr5e-6-q16-steps300-initpasmrpam-rlhalf")["max_tokens"] == 2048
