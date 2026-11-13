import numpy as np
import torch

from votedistill.copsd.loss import generalized_jsd_loss
from votedistill.exp2 import screen


def test_token_stats_matches_copsd_loss_and_is_zero_for_identical_models():
    g = torch.Generator().manual_seed(0)
    L, V = 37, 50
    s, t = torch.randn(L, V, generator=g) * 3, torch.randn(L, V, generator=g) * 3
    y = torch.randint(0, V, (L,), generator=g).tolist()
    old = screen.CHUNK
    screen.CHUNK = 8                                       # exercise chunking
    try:
        d, f, fc = screen.token_stats(s, t, y, temp=1.1, clip=0.05)
        ref = generalized_jsd_loss(s[None], t[None], beta=0, temperature=1.1, reduction="none")[0].sum(-1)
        refc = generalized_jsd_loss(s[None], t[None], beta=0, temperature=1.1, reduction="none", token_clip=0.05)[0].sum(-1)
        assert np.allclose(f, ref.numpy(), atol=1e-5) and np.allclose(fc, refc.numpy(), atol=1e-5)
        ls, lt = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
        assert np.allclose(d, (ls[range(L), y] - lt[range(L), y]).numpy(), atol=1e-5)
        assert (f >= -1e-6).all()
        d0, f0, _ = screen.token_stats(s, s, y)
        assert np.abs(d0).max() < 1e-6 and np.abs(f0).max() < 1e-6
    finally:
        screen.CHUNK = old
