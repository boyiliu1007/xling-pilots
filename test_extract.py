from extract import extract_answer, last_boxed, majority_vote, numeric_equal, parse_number
from stats import wilson


def test_basic():
    assert extract_answer(r"so \boxed{42}") == 42
    assert extract_answer(r"\boxed{-42}") == -42
    assert extract_answer(r"\boxed{3.5}") == 3.5
    assert extract_answer(r"\boxed{1,200}") == 1200
    assert extract_answer(r"\boxed{1{,}200}") == 1200
    assert extract_answer(r"\boxed{\text{42}}") == 42
    assert extract_answer(r"\boxed{\$18}") == 18
    assert extract_answer(r"\boxed{50\%}") == 50
    assert extract_answer(r"\boxed{42 \text{ dollars}}") == 42
    assert extract_answer(r"\boxed{18.00}") == 18 and isinstance(extract_answer(r"\boxed{18.00}"), int)


def test_nested_and_multiple():
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_answer(r"\boxed{\frac{1}{2}}") == 0.5
    assert extract_answer(r"\boxed{\text{\textbf{7}}}") == 7
    assert extract_answer(r"first \boxed{3} then \boxed{5}") == 5
    assert extract_answer(r"\boxed{80 + 150 = 230 \text{ miles}}") == 230
    assert extract_answer(r"\boxed{x = 2y + 3}") is None


def test_fallback_and_unparseable():
    assert extract_answer("The total is 1,250 apples, so 1,250.") == 1250
    assert extract_answer("She has -3.5 left") == -3.5
    assert extract_answer("I cannot determine this.") is None
    assert extract_answer(r"\boxed{x}") is None  # boxed present but non-numeric -> None, no fallback
    assert extract_answer(r"\boxed{3 and 5}") is None
    assert extract_answer("") is None
    assert parse_number("abc") is None


def test_votes_and_ties():
    maj, counts, share, tie = majority_vote([5, 5, 5, 3, 3, None, None, 7])
    assert maj == 5 and share == 3 / 8 and not tie and counts["5"] == 3
    maj, counts, share, tie = majority_vote([5, 5, 3, 3, None, None, None, None])
    assert maj is None and tie and share == 2 / 8
    maj, counts, share, tie = majority_vote([None] * 8)
    assert maj is None and share == 0 and not tie
    maj, _, share, tie = majority_vote([3, 3.0, 2, None])
    assert maj == 3 and share == 0.5 and not tie


def test_unparseable_never_correct():
    assert not numeric_equal(None, 5)
    assert not numeric_equal(None, None)
    assert numeric_equal(1200, 1200.0)


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
