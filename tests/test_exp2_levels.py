from votedistill.exp2.levels import TEACHER_CONTEXTS, answer_only, derangement, teacher_rows


def _rows(n=7):
    return [{"problem": f"q{i}", "solution": f"sol{i}", "Answer": str(i)} for i in range(n)]


def test_derangement_has_no_fixed_points_and_is_deterministic():
    for n in (2, 3, 10, 500):
        p = derangement(n)
        assert sorted(p) == list(range(n))
        assert all(p[i] != i for i in range(n))
        assert p == derangement(n)


def test_teacher_rows_reference_content():
    rows = _rows()
    r, f = teacher_rows(rows, "L0-same")
    assert f == dict(include_problem_en=False, include_reference_solution_en=False)
    r, f = teacher_rows(rows, "L1")
    assert f == dict(include_problem_en=True, include_reference_solution_en=False)
    r, f = teacher_rows(rows, "L3-gold")
    assert f["include_reference_solution_en"] and [x["solution"] for x in r] == [x["solution"] for x in rows]
    r, _ = teacher_rows(rows, "L2-gold")
    assert [x["solution"] for x in r] == [answer_only(x["Answer"]) for x in rows]
    r, _ = teacher_rows(rows, "L3-shuffled-gold")
    assert all(x["solution"] != y["solution"] for x, y in zip(r, rows))
    assert sorted(x["solution"] for x in r) == sorted(x["solution"] for x in rows)
    assert [x["problem"] for x in r] == [x["problem"] for x in rows]   # only the reference moves
    assert rows[0]["solution"] == "sol0"                                # input not mutated


def test_all_contexts_buildable():
    for c in TEACHER_CONTEXTS:
        teacher_rows(_rows(), c)
