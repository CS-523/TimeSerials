"""Tests for ``src_control.models.state_space_nn``."""
from __future__ import annotations

import numpy as np
import torch

from src_control.models.state_space import LinearSS
from src_control.models.state_space_nn import SS_NN_Hybrid, YHead, init_hybrid_from_n4sid


def test_hybrid_forward_shape():
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=8, hidden=32, window=3)
    u = torch.randn(2, 12, 8)
    y_prev = torch.randn(2, 12, 4)
    y_pred = model(u, y_prev=y_prev, teacher_forcing=0.5)
    assert y_pred.shape == (2, 12, 4)


def test_hybrid_rollout_ar():
    y_ar = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=8, hidden=32, window=3).rollout(
        torch.randn(2, 12, 8)
    )
    assert y_ar.shape == (2, 12, 4)
    assert torch.isfinite(y_ar).all()


def test_hybrid_init_from_n4sid():
    A = np.eye(8) * 0.9
    B = np.zeros((8, 8))
    C = np.zeros((4, 8))
    D = np.zeros((4, 8))
    n4sid_model = LinearSS(A=A, B=B, C=C, D=D, n=8, m=8, p=4)
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=8)
    init_hybrid_from_n4sid(model, n4sid_model)
    assert torch.allclose(model.linear.A.data, torch.tensor(A, dtype=torch.float32))


def test_yhead_adaptive_window():
    yh = YHead(window=8)
    # Different window should still produce output
    out = yh(torch.randn(3, 4, 4))
    assert out.shape == (3,)
    out2 = yh(torch.randn(3, 8, 4))
    assert out2.shape == (3,)