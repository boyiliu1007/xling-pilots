from votedistill.exp2.runs import eval_steps, final_step, init_of, parse_name


def test_two_stage_names():
    n = "L1__copsd__exact__s1vote30-noclip__s43"
    assert parse_name(n) == ("L1", "copsd", "exact", 43, None)
    assert init_of(n) == ("L3-vote__copsd__exact__s43", 30)
    assert eval_steps(n)[0] == "step30" and eval_steps(n)[-1] == "step100"
    assert init_of("L3-gold__copsd__exact__s1gold30-noclip__s42") == ("L3-gold__copsd__exact__s42", 30)
    assert init_of("L3-vote__copsd__exact__s1sft30-noclip__s42") == ("selfsft-vote__copsd__exact__s42", 30)
    assert init_of("L3-vote__copsd__exact__noclip__s42") is None


def test_selfsft_name():
    n = "selfsft-vote__copsd__exact__s42"
    assert parse_name(n)[0] == "selfsft-vote"
    assert eval_steps(n) == ["step10", "step20", "step30"] and final_step(n) == 30
