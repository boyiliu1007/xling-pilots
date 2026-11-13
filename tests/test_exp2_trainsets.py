import torch

from votedistill.copsd.train import unpad
from votedistill.exp2.runs import parse_name, run_name
from votedistill.exp2.trainsets import gsm8k_reference
from votedistill.exp2.translate import qc, split_sentences


def test_unpad_strips_only_right_padding():
    assert unpad(torch.tensor([5, 6, 7, 0, 0]), 0).tolist() == [5, 6, 7]
    assert unpad(torch.tensor([5, 6]), 0).tolist() == [5, 6]
    try:
        unpad(torch.tensor([5, 0, 7, 0]), 0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_gsm8k_reference():
    ans = "She has 3*4=<<3*4=12>>12 eggs.\nShe sells 12-2=<<12-2=10>>10.\n#### 1,000"
    ref = gsm8k_reference(ans)
    assert "<<" not in ref and ref.endswith("\\boxed{1000}.") and "3*4=12" in ref


def test_split_sentences_keeps_numbers():
    s = split_sentences("Tom has $3.50. He buys 2 apples! How many?\nAnswer now.")
    assert s == ["Tom has $3.50.", "He buys 2 apples!", "How many?", "Answer now."]


def test_qc():
    assert qc("Tom has 3 apples and 12 pears.", "Tom ana tufaha 3 na pea 12.")[0]
    assert qc("Tom has 3 apples and 12 pears.", "Tom ana tufaha tatu na pea 12.")[0]      # small number as a word
    ok, why = qc("Tom has 3 apples and 12 pears.", "Tom ana tufaha na pea 12.")              # number dropped
    assert not ok and why[0].startswith("numbers")
    assert not qc("Tom has 30 apples and 12 pears.", "Tom ana tufaha thelathini na pea 12.")[0]   # > WORD_MAX
    assert not qc("Tom has 3 apples.", "Tom ana tufaha 3 na 5.")[0]                          # invented number


def test_run_names_roundtrip():
    n = run_name("L3-vote", "gsm8k-mt", "exact", 7)
    assert parse_name(n) == ("L3-vote", "gsm8k-mt", "exact", 7)
