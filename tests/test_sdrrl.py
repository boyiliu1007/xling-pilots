import random

from votedistill.sdrrl.codeswitch import code_switch, parse_tei
from votedistill.sdrrl.mathtext import mask_math, rebuild, split_document, unmask
from votedistill.sdrrl.parallel import instructions
from votedistill.sdrrl.references import agrees, select_reference, split_trace

DOC = """### Step 1: Let the weight of an apple be $ a $ and a pear be $ p $.

From the first equation:

$$
3a + p = 10l \\tag{1}
$$

So **7 plums** balance one pear. The answer is $\\boxed{7}$.

---
$$ x = 1 $$"""


def identity(pieces):
    return {p.text: p.text for p in pieces if p.kind == "text"}


def test_mask_roundtrip():
    masked, masks = mask_math(r"Since $\frac{3}{4}$ of 120 passed, we get \(90\) and \boxed{\frac{1}{2}}.")
    assert masked == "Since [M1] of 120 passed, we get [M2] and [M3]."
    assert unmask(masked, masks) == r"Since $\frac{3}{4}$ of 120 passed, we get \(90\) and \boxed{\frac{1}{2}}."
    assert unmask("Tangu [m 1] ya 120, [M3] na [M2].", masks).startswith(r"Tangu $\frac{3}{4}$ ya 120, \boxed{\frac{1}{2}}")


def test_unmask_detects_lost_or_duplicated_math():
    _, masks = mask_math("Let $a$ be $b$.")
    assert unmask("Acha [M1] iwe.", masks) is None
    assert unmask("Acha [M1] iwe [M1].", masks) is None
    assert unmask("Acha [M1] iwe [M2] na [M3].", masks) is None


def test_boxed_inside_inline_math_is_one_span():
    masked, masks = mask_math(r"The answer is $\boxed{7}$.")
    assert masked == "The answer is [M1]." and masks == [r"$\boxed{7}$"]


def test_split_document_rebuilds_and_protects_math():
    pieces = split_document(DOC)
    text, n_text, lost = rebuild(pieces, identity(pieces))
    assert lost == 0 and text == DOC.replace("**", "")
    sentences = [p.text for p in pieces if p.kind == "text"]
    assert sentences == ["Step 1: Let the weight of an apple be [M1] and a pear be [M2].", "From the first equation:",
                         "So 7 plums balance one pear.", "The answer is [M1]."]
    assert n_text == 4
    assert all("$" not in s and "\\" not in s for s in sentences)          # no LaTeX reaches the translator


def test_rebuild_fails_when_a_sentence_loses_math():
    pieces = split_document("Let $a$ be the weight. Then we are done.")
    tr = identity(pieces)
    tr["Let [M1] be the weight."] = "Acha iwe uzito."
    assert rebuild(pieces, tr) == (None, 2, 1)


def test_agrees():
    assert agrees(7, 7.0) and not agrees(7, 8) and not agrees(None, 7)
    assert agrees(r"\frac{1}{2}", "0.5") and agrees(" 18 ", "18") and not agrees("x+1", "x-1")


def test_select_reference():
    rec = {"raw_outputs": ["long " * 30 + r"\boxed{7}", r"short \boxed{7}", r"wrong \boxed{8}", r"cut \boxed{7}"],
           "preds": [7, 7, 8, 7], "finish_reasons": ["stop", "stop", "stop", "length"],
           "majority_answer": 7, "vote_share": 0.75, "is_tie": False}
    assert select_reference(rec) == r"short \boxed{7}"                        # shortest agreeing, untruncated
    assert select_reference(rec, pick="longest").startswith("long")
    assert select_reference(rec, min_share=0.875) is None
    assert select_reference({**rec, "is_tie": True}) is None


def test_split_trace():
    assert split_trace("<think>\nreason\n</think>\n\nThe answer is 7.", "7") == ("reason", "The answer is 7.")
    assert split_trace("plain solution", "7") == ("plain solution", None)


def test_parse_tei():
    xml = """<entry><form><orth>above</orth></form><sense><def>juu</def></sense></entry>
    <entry xml:id="abudu"><form xml:lang="sw"><orth>abudu</orth></form><sense><def>worship, revere (formal); adore</def>
    <note type="editor">ignore me</note></sense></entry>"""
    assert parse_tei(xml) == [("above", ["juu"]), ("abudu", ["worship", "revere", "adore"])]


def test_code_switch_keeps_math_and_counts():
    d = {"total": "jumla", "left": "kushoto", "apples": "matofaa"}
    text = r"Total apples: $\left(3\right)$ and \[ \text{total} \] with \left words left."
    out, found, swapped, total = code_switch(text, d, 1.0, random.Random(0))
    assert out == r"Jumla matofaa: $\left(3\right)$ and \[ \text{total} \] with \left words kushoto."
    assert (found, swapped, total) == (3, 3, 6)
    assert code_switch(text, d, 0.0, random.Random(0))[0] == text


def test_parallel_instructions():
    ex = instructions([("The fox jumps over the dog today.", "Mbweha anaruka juu ya mbwa leo.")], "swa", random.Random(0))
    assert [e["subset"] for e in ex] == ["mt_en2tgt", "mt_tgt2en", "completion"]
    assert ex[0]["user"].startswith("Translate the following sentence from English to Swahili.\n") and ex[0]["answer"].startswith("Mbweha")
    prefix = ex[2]["user"].split("\n", 1)[1]
    assert ex[2]["answer"].startswith(prefix) and 0 < len(prefix) < len(ex[2]["answer"])


def test_code_switch_skips_single_letters():
    out, found, swapped, total = code_switch("(A) $A$ na B", {"a": "of", "na": "and"}, 1.0, random.Random(0))
    assert out == "(A) $A$ and B" and (found, swapped, total) == (1, 1, 3)


def test_quality_scores_align_with_segments(tmp_path, monkeypatch):
    import json

    from votedistill.sdrrl import quality
    recs = [{"question_id": 3, "segments": [{"src": "a", "mt": "x"}, {"src": "b", "mt": "y"}]},
            {"question_id": 5, "segments": [{"src": "a", "mt": "x"}]}, {"question_id": 9, "segments": []}]
    (tmp_path / "translations.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(quality, "score_pairs", lambda pairs, *a: {("a", "x"): 0.9, ("b", "y"): 0.5})
    quality.run(tmp_path)
    out = [json.loads(l) for l in open(tmp_path / "qe_scores.jsonl")]
    assert [o["scores"] for o in out] == [[0.9, 0.5], [0.9], []]
    assert out[0]["mean"] == 0.7 and out[0]["min"] == 0.5 and out[2]["mean"] is None


def test_translation_verdict():
    from votedistill.sdrrl.build import translation_verdict
    sw = "Kwa hiyo jumla ya matunda ni kumi na mbili, na jibu la mwisho ni hili."
    ok = {"reasoning_tgt": sw}
    assert translation_verdict(ok, {"mean": 0.85}, 0.8, "swa") is None
    assert translation_verdict(ok, {"mean": 0.79}, 0.8, "swa") == "low_quality"
    assert translation_verdict(ok, None, 0.8, "swa") is None                         # --no_qe
    assert translation_verdict({"reasoning_tgt": None}, {"mean": 0.9}, 0.8, "swa") == "math_lost"
    assert translation_verdict({"reasoning_tgt": "So the total is twelve and we are done."}, {"mean": 0.9}, 0.8, "swa") == "off_target_language"


def test_sft_schedule_and_assistant_turn():
    from votedistill.sdrrl.build import assistant_turn
    from votedistill.sdrrl.train import schedule
    s = schedule(10, 2, 4, seed=0)
    assert [len(b) for b in s] == [4, 4, 2, 4, 4, 2] and sorted(sum(s[:3], [])) == list(range(10)) and s[:3] != s[3:]
    assert schedule(10, 2, 4, seed=0) == s
    assert assistant_turn("Prefix.", " reason \n", " final ") == "<think>\nPrefix. reason\n</think>\n\nfinal"


# ---------------------------------------------------------------- Experiment 2 integration (exp2/runs.py context "sdrrl")

def test_exp2_sdrrl_names():
    import pytest
    from votedistill.exp2.runs import parse_name, sdrrl_epochs
    assert parse_name("sdrrl__gsm8k-mt3k__exact__s42")[:4] == ("sdrrl", "gsm8k-mt3k", "exact", 42)
    assert parse_name("sdrrl__gsm8k-mt3k__exact__ep2__s43")[:4] == ("sdrrl", "gsm8k-mt3k", "exact", 43)
    assert sdrrl_epochs("sdrrl__gsm8k-mt3k__exact__s42") == 4
    assert sdrrl_epochs("sdrrl__gsm8k-mt3k__exact__ep2__s42") == 2
    with pytest.raises(SystemExit):
        parse_name("sdrrl__gsm8k-mt3k__exact__noclip__s42")       # a COPSD variant is not an SDRRL variant


def test_exp2_sdrrl_steps_come_from_config(tmp_path, monkeypatch):
    import json
    import pytest
    from votedistill.exp2 import runs
    monkeypatch.setattr(runs, "RUNS", tmp_path)
    n = "sdrrl__gsm8k-mt3k__exact__s42"
    with pytest.raises(SystemExit):                                  # not trained yet
        runs.final_step(n)
    (tmp_path / n).mkdir()
    (tmp_path / n / "config.json").write_text(json.dumps({"sft": {"final_step": 35}}))
    assert runs.final_step(n) == 35
    assert runs.eval_steps(n) == ["step10", "step20", "step30", "step35"]


def test_exp2_data_dirs_follow_base_model():
    from votedistill.exp2.levels import EXP2_RESULTS
    from votedistill.sdrrl import exp2_data_dir, exp2_votes
    assert exp2_data_dir("gsm8k-mt3k") == EXP2_RESULTS / "sdrrl" / "gsm8k-mt3k"
    assert exp2_votes("gsm8k-mt3k") == EXP2_RESULTS / "consensus" / "gsm8k3k_en_thinking.jsonl"


def test_gsm8k_question_rows_match_consensus_ids():
    import pytest
    from votedistill.exp2.trainsets import MT_FILES
    if not MT_FILES["gsm8k-mt3k"].exists():
        pytest.skip("gsm8k-mt3k translation not checked out")
    from votedistill.sdrrl.build import question_rows
    rows = question_rows("gsm8k-mt3k")
    assert len(rows) == 2805                                         # NLLB QC passes (Experiment 2)
    q, r = next(iter(rows.items()))
    assert isinstance(q, int) and r["problem"] and r["problem_translated"] and r["qid"] == q


def test_grpo_warm_start_variant():
    from votedistill.exp2.grpo import parse_variant
    from votedistill.exp2.runs import parse_name
    v = parse_variant("votelidref-lr5e-6-q16-steps300-initsdrrl")
    assert v["kind"] == "votelidref" and v["steps"] == 300 and v["init"] == "sdrrl" and v["lr"] == 5e-6
    assert parse_variant("votelidref-lr5e-6-q16-steps300")["init"] is None          # Jeff's runs unchanged
    assert parse_name("grpo__gsm8k-mt3k__exact__votelidref-lr5e-6-q16-steps300-initsdrrl__s42")[0] == "grpo"
