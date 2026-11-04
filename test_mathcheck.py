from mathcheck import cluster_vote, equiv, extract_pred, parse_answer


def g(gold, text, thinking=False):
    s, p = extract_pred(text, thinking)
    return equiv(gold, parse_answer(gold), s, p)


def test_fraction_vs_decimal():
    assert g(r"\frac{1}{2}", r"so \boxed{0.5}")
    assert g("0.5", r"\boxed{\frac{1}{2}}")
    assert g(r"\frac{3\sqrt{3}}{2}", r"\boxed{\dfrac{3\sqrt{3}}{2}}")


def test_equivalent_expressions():
    assert g(r"2\sqrt{2}", r"\boxed{\sqrt{8}}")
    assert g("x+1", r"\boxed{1+x}")
    assert g("18", r"\boxed{18}")


def test_nested_braces_and_last_box():
    assert g(r"\frac{1}{3}", r"first \boxed{2} then \boxed{\frac{1}{3}}")
    s, _ = extract_pred(r"\boxed{\frac{\sqrt{3}}{2}}")
    assert s == r"\frac{\sqrt{3}}{2}"


def test_non_equivalent():
    assert not g(r"\frac{1}{3}", r"\boxed{0.5}")
    assert not g("x+1", r"\boxed{x-1}")
    assert not g("18", r"\boxed{19}")


def test_unparseable_never_correct():
    s, p = extract_pred("I cannot solve this.")
    assert s is None and p == []
    assert not g("18", "I cannot solve this.")
    assert not g("18", "")


def test_thinking_final_text():
    assert g("7", "<think>maybe \\boxed{5}</think>\nSo \\boxed{7}", thinking=True)
    assert not g("5", "<think>maybe \\boxed{5} ... (truncated)", thinking=True)


def test_cluster_vote():
    preds = [r"\frac{1}{2}", "0.5", "1/2", "3", "3", None, "abc?", "0.50"]
    S, P = [], []
    for t in preds:
        s, p = extract_pred(r"\boxed{" + t + "}") if t is not None else (None, [])
        S.append(s)
        P.append(p)
    rep, counts, share, tie = cluster_vote(S, P, 8)
    assert not tie and S[rep] == r"\frac{1}{2}" and share == 4 / 8
    rep, counts, share, tie = cluster_vote(["1", "2", "1", "2"], [parse_answer(x) for x in "1212"], 8)
    assert rep is None and tie and share == 2 / 8
    rep, counts, share, tie = cluster_vote([None] * 8, [[]] * 8, 8)
    assert rep is None and share == 0 and not tie
