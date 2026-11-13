"""The clipped-gradient formula used by votedistill/exp2/clipstats.py matches autograd through copsd/loss.py."""
import torch

from votedistill.copsd.loss import generalized_jsd_loss


def test_clipped_gradient_formula():
    torch.manual_seed(0)
    T, c = 1.1, 0.05
    s = (torch.randn(5, 50) * 3).requires_grad_()
    t = torch.randn(5, 50) * 3
    for clip in (None, c):
        s.grad = None
        generalized_jsd_loss(s[None], t[None], beta=0.0, temperature=T, reduction="sum", token_clip=clip).backward()
        ls, lt = torch.log_softmax(s.detach() / T, -1), torch.log_softmax(t / T, -1)
        ps, pt = ls.exp(), lt.exp()
        if clip is None:
            g = ps - pt
        else:
            keep = ((pt * (lt - ls)) <= clip).float()
            g = -pt * keep + (pt * keep).sum(-1, keepdim=True) * ps
        assert torch.allclose(s.grad, g / T, atol=1e-6)
