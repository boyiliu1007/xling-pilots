"""The 1-5 score table of the LLM judge is part of the evaluation protocol: these tests pin it."""
import itertools

import pytest

from votedistill.judge import Judgment, pick_reference, qwk, score_from


def J(und="yes", steps="all_correct", sup="yes", gold="no"):
    return {"understood_problem": und, "steps": steps, "supports_answer": sup, "reaches_gold": gold}


@pytest.mark.parametrize("j, correct, score", [
    (J(), True, 5),
    (J(steps="minor_slip"), True, 4),
    (J(steps="major_errors"), True, 3),
    (J(und="partly"), True, 3),
    (J(steps="no_steps"), True, 2),
    (J(sup="no"), True, 2),
    (J(gold="yes", sup="no"), False, 3),            # right value in the work, wrong or missing box
    (J(steps="minor_slip"), False, 3),              # one slip from right
    (J(), False, 2),                                # correct steps but unfinished
    (J(und="partly", steps="major_errors"), False, 2),
    (J(steps="major_errors"), False, 2),
    (J(und="no", steps="major_errors"), False, 1),
    (J(steps="no_steps", sup="no"), False, 1),
])
def test_score_table(j, correct, score):
    assert score_from(j, correct) == score


def test_every_combination_scores_1_to_5_and_correct_never_below_wrong_twin():
    """Same checklist answers, final answer right vs wrong: right never scores lower, except the intended case of a
    right answer the work does not produce (2) vs an honest one-slip near miss (3)."""
    F = Judgment.model_fields
    keys = ["understood_problem", "steps", "supports_answer", "reaches_gold"]
    for combo in itertools.product(*(F[k].annotation.__args__ for k in keys)):
        j = dict(zip(keys, combo))
        right, wrong = score_from(j, True), score_from(j, False)
        assert 1 <= right <= 5 and 1 <= wrong <= 5
        unearned_vs_near_miss = (j["supports_answer"], j["reaches_gold"], j["understood_problem"], j["steps"]) == \
            ("no", "no", "yes", "minor_slip")
        assert wrong <= right or (unearned_vs_near_miss and (right, wrong) == (2, 3)), (j, right, wrong)


def test_pick_reference_uses_majority_and_skips_truncated():
    rec = {"outputs": ["x</think>sol5", "y</think>sol7a", "z</think>sol7b"], "preds": ["5", "7", "7"],
           "formatted": [True, True, True], "truncated": [False, True, False]}
    ref = pick_reference(rec)
    assert ref["answer"] == "7" and ref["text"] == "sol7b"


def test_qwk():
    assert qwk([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert qwk([1, 1, 5, 5], [5, 5, 1, 1]) < 0
