import numpy as np

from mcq import item_record, map_back, position_of, renorm, shift_order, vote
from stats import wilson


def test_roundtrip():
    for s in range(4):
        order = shift_order(s)
        assert sorted(order) == [0, 1, 2, 3]
        for o in range(4):
            assert order[position_of(o, s)] == o
    assert shift_order(0) == [0, 1, 2, 3]
    assert shift_order(1) == [1, 2, 3, 0]


def test_gold_visits_every_position():
    for gold in range(4):
        assert sorted(position_of(gold, s) for s in range(4)) == [0, 1, 2, 3]


def test_map_back():
    # model puts all mass on letter B (position 1) under shift 1 -> original option 2
    assert map_back([0, 1, 0, 0], 1) == [0, 0, 1, 0]
    # pure position bias "always A" maps to a different original option each shift
    picks = [int(np.argmax(map_back([1, 0, 0, 0], s))) for s in range(4)]
    assert picks == [0, 1, 2, 3]


def test_renorm():
    p, m = renorm([0.2, 0.1, 0.1, 0.1])
    assert abs(m - 0.5) < 1e-12 and abs(p[0] - 0.4) < 1e-12 and abs(sum(p) - 1) < 1e-12
    p, m = renorm([0, 0, 0, 0])
    assert m == 0 and p == [0.25] * 4


def test_vote_ties():
    assert vote([1, 1, 2, 3]) == 1
    assert vote([1, 1, 2, 2]) is None
    assert vote([0, 1, 2, 3]) is None
    assert vote([3, 3, 3, 3]) == 3


def test_debiased_and_record():
    # position-biased model: 0.7 on letter A, rest split; gold = 2
    mapped = [map_back([0.7, 0.1, 0.1, 0.1], s) for s in range(4)]
    r = item_record(mapped, [0, 0, 0, 0], gold=2)
    assert r["order_sensitive"] and r["mean_shift_accuracy"] == 0.25
    assert r["vote_answer"] is None and not r["vote_correct"]
    assert np.allclose(r["debiased_probs"], [0.25] * 4)
    # content-driven model: always prefers original option 2
    mapped = [[0.1, 0.1, 0.7, 0.1]] * 4
    r = item_record(mapped, [position_of(2, s) for s in range(4)], gold=2)
    assert not r["order_sensitive"] and r["debiased_correct"] and r["single_order_correct"] and r["vote_correct"]
    # averaging can flip the answer vs single order
    mapped = [[0.4, 0.35, 0.15, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.5, 0.1, 0.1], [0.45, 0.4, 0.1, 0.05]]
    r = item_record(mapped, [0, 0, 0, 0], gold=1)
    assert not r["single_order_correct"] and r["debiased_correct"]


def test_wilson_na():
    assert wilson(0, 0) is None
