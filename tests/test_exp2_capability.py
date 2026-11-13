from votedistill.exp2.capability import GROUPS, group_of, item_features


def _rec(item, preds, gold="18", maj_correct=False, mean_acc=0.0):
    return {"item": item, "preds": preds, "formatted": [p is not None for p in preds], "gold": gold,
            "maj_correct": maj_correct, "mean_acc": mean_acc}


def test_features_and_groups():
    en = _rec(0, ["18"] * 8 + ["17"] * 2 + [None] * 2)
    sw_fail = _rec(0, ["5"] * 3 + ["18.0"] * 2 + [None] * 7)
    f = item_features(en, sw_fail)
    assert f["en_share"] == 8 / 12 and f["en_maj"] == 18 and f["en_maj_correct"]
    assert f["sw_agree"] == 2 / 12 and not f["cross"]
    assert group_of(f).startswith("A")
    sw_ok = _rec(0, ["18.00"] * 4 + [None] * 8)
    assert group_of(item_features(en, sw_ok)).startswith("B")
    en_unsure = _rec(0, ["18"] * 6 + ["17"] * 6)          # tie -> no English majority
    f = item_features(en_unsure, sw_ok)
    assert f["en_share"] == 0 and f["en_maj"] is None and group_of(f).startswith("C")


def test_groups_partition():
    for es in (0.0, 0.5, 0.51, 1.0):
        for cross in (False, True):
            assert sum(rule({"en_share": es, "cross": cross}) for rule in GROUPS.values()) == 1
