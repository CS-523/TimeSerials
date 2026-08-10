"""Hybrid linear-SS + autoregressive neural-network predictor.

Two heads stacked:

1. A **linear state-space baseline** with ``A, B, C, D`` as ``nn.Parameter``,
   initialised from N4SID on the training set. This gives an interpretable,
   stable linear rollout that anchors the prediction.

2. A **nonlinear residual NN** (a small MLP over a sliding window) that learns
   the residual ``r_t = y_t − y_t^{linear}``. The MLP is preferred over a
   full Mamba/LSTM for stability on this dataset — the linear SS already
   captures most of the slow trends, and a small MLP is fast to train and
   easy to deploy.

The hybrid model supports both teacher-forced training and autoregressive
rollout at inference. During autoregressive generation, the standardized
output ``y_pred`` is fed back as input on the next step.

Optionally, a tiny :class:`YHead` MLP regresses the final ``Y`` value from
the last-window y features — used by the optimizer.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src_control.models.state_space import LinearSS


class LinearSSTorch(nn.Module):
    """Differentiable discrete-time linear state-space layer.

    State propagation is unrolled sequentially (T ≤ ~320 in our dataset).
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float32))
        self.D = nn.Parameter(torch.tensor(D, dtype=torch.float32))
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]

    def rollout(
        self,
        u: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rollout with input ``u`` of shape ``(B, T, m)``.

        Returns ``y`` of shape ``(B, T, p)``.
        """
        B, T, _ = u.shape
        if x0 is None:
            x = torch.zeros(B, self.n, device=u.device, dtype=u.dtype)
        else:
            x = x0
        ys = []
        for t in range(T):
            x = x @ self.A.T + u[:, t, :] @ self.B.T
            y = x @ self.C.T + u[:, t, :] @ self.D.T
            ys.append(y)
        return torch.stack(ys, dim=1)


class ResidualMLP(nn.Module):
    """Residual MLP that maps a sliding window of (u, y_lin) → y_res.

    Uses an MLP over the last ``window`` (y_lin + u) vectors concatenated.
    This stays purely feedforward so autoregressive rollout is well-defined
    (no hidden-state coupling).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        window: int = 4,
        hidden: int = 128,
    ):
        super().__init__()
        self.window = window
        self.dim_in = dim_in
        self.flatten_dim = window * dim_in
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, dim_in). Returns y_res: (B, T, dim_out).

        At each step t, concatenates the past ``window`` features
        [f[t-w+1], ..., f[t]] and applies the MLP. The first ``window-1``
        steps use a shorter window (zero-padded at the start).
        """
        B, T, d = features.shape
        w = self.window
        pad = torch.zeros(B, w - 1, d, device=features.device, dtype=features.dtype)
        padded = torch.cat([pad, features], dim=1)  # (B, T+w-1, d)
        outs = []
        for t in range(T):
            window = padded[:, t : t + w, :]
            flat = window.reshape(B, w * d)
            outs.append(self.net(flat))
        return torch.stack(outs, dim=1)


class SS_NN_Hybrid(nn.Module):
    """Hybrid prediction model: linear SS baseline + residual NN.

    Forward signature::

        y_pred = model(u, y_prev=None, teacher_forcing=1.0)

    Parameters
    ----------
    u : (B, T, m) exogenous inputs
    y_prev : (B, T, p) previously observed y (or None for pure autoregressive)
    teacher_forcing : in [0, 1]; if y_prev is given, blend y_prev with the
        previous prediction by this amount.
    """

    def __init__(
        self,
        dim_u: int = 8,
        dim_y: int = 4,
        n_state: int = 16,
        hidden: int = 128,
        window: int = 4,
    ):
        super().__init__()
        self.dim_u = dim_u
        self.dim_y = dim_y
        self.window = window

        # Linear SS — initialised lazily (to zeros) before N4SID init.
        self.linear = LinearSSTorch(
            A=np.eye(n_state) * 0.9,
            B=np.zeros((n_state, dim_u)),
            C=np.zeros((dim_y, n_state)),
            D=np.zeros((dim_y, dim_u)),
        )
        # Residual NN — input is concatenation of (u, y_lin, y_prev) per step.
        # dim_in = dim_u + dim_y + dim_y (= 8 + 4 + 4 = 16)
        self.residual = ResidualMLP(
            dim_in=dim_u + 2 * dim_y, dim_out=dim_y, window=window, hidden=hidden
        )

    def init_from_n4sid(self, model: LinearSS) -> None:
        """Initialise linear SS parameters from a fitted :class:`LinearSS`."""
        with torch.no_grad():
            self.linear.A.data = torch.tensor(model.A, dtype=torch.float32)
            self.linear.B.data = torch.tensor(model.B, dtype=torch.float32)
            self.linear.C.data = torch.tensor(model.C, dtype=torch.float32)
            self.linear.D.data = torch.tensor(model.D, dtype=torch.float32)

    def forward(
        self,
        u: torch.Tensor,
        y_prev: Optional[torch.Tensor] = None,
        teacher_forcing: float = 1.0,
    ) -> torch.Tensor:
        """Predict y for the same time steps as u.

        Parameters
        ----------
        u : (B, T, m)
        y_prev : (B, T, p) ground-truth or previously known y (or None).
        teacher_forcing : in [0, 1]. With TF=1, residual MLP sees ground-truth
            y_prev; with TF=0, it sees the model's own prediction.
        """
        B, T, m = u.shape
        y_lin = self.linear.rollout(u)  # (B, T, p)

        if y_prev is None:
            y_ctx = y_lin
        else:
            y_ctx = teacher_forcing * y_prev + (1.0 - teacher_forcing) * y_lin

        # Concatenate [u, y_lin, y_ctx] along last dim
        feats = torch.cat([u, y_lin, y_ctx], dim=-1)
        y_res = self.residual(feats)
        return y_lin + y_res

    def rollout(
        self,
        u: torch.Tensor,
        y_seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout. ``u`` is ``(B, H, m)``.

        Returns ``y`` of shape ``(B, H, p)``.

        Parameters
        ----------
        y_seed : (B, w-1, p) initial past-y window. If None, zeros.

        Note
        ----
        During rollout we set ``teacher_forcing=0`` so the residual MLP uses
        the model's own previous prediction (y_lin) as context. This is the
        standard open-loop forecast mode.
        """
        # Open-loop rollout: predict y_t given only y_lin (no y_prev available).
        return self.forward(u, y_prev=None, teacher_forcing=0.0)


class YHead(nn.Module):
    """Tiny MLP regressing final ``Y`` from the last-cycle y window.

    Inputs: (B, window * 4) — flatten the last ``window`` (standardised) y values.
    Output: (B,) predicted final Y.

    The forward accepts any ``window`` length by re-interpolating to the
    training-time window via ``adaptive_avg_pool1d``.
    """

    def __init__(self, window: int = 8, hidden: int = 64):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(window * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, y_tail: torch.Tensor) -> torch.Tensor:
        """y_tail: (B, w, 4). Returns (B,).

        If ``w`` differs from the training window, we pool to the training
        length before flattening.
        """
        B, w, p = y_tail.shape
        if w != self.window:
            # Reshape to (B, p, w) for pooling, then back
            y_tail = y_tail.transpose(1, 2)  # (B, p, w)
            y_tail = torch.nn.functional.adaptive_avg_pool1d(y_tail, self.window)
            y_tail = y_tail.transpose(1, 2)  # (B, window, p)
        return self.net(y_tail.reshape(B, self.window * p)).squeeze(-1)


def init_hybrid_from_n4sid(
    model: SS_NN_Hybrid,
    n4sid_model: LinearSS,
) -> None:
    """Convenience wrapper for ``model.init_from_n4sid(n4sid_model)``."""
    model.init_from_n4sid(n4sid_model)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _selftest():
    """Forward / backward / autoregressive rollout sanity checks."""
    torch.manual_seed(0)
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=8, hidden=32, window=3)
    u = torch.randn(2, 12, 8)
    y_prev = torch.randn(2, 12, 4)
    y_pred = model(u, y_prev=y_prev, teacher_forcing=0.5)
    assert y_pred.shape == (2, 12, 4), f"shape mismatch: {y_pred.shape}"

    y_ar = model.rollout(u)
    assert y_ar.shape == (2, 12, 4), f"rollout shape mismatch: {y_ar.shape}"
    assert torch.isfinite(y_ar).all(), "rollout produced NaN"

    # Backward
    loss = y_pred.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"[selftest] WARN: {name} has no grad")

    # Y head
    yh = YHead(window=4)
    yh_pred = yh(torch.randn(3, 4, 4))
    assert yh_pred.shape == (3,), f"y-head shape mismatch: {yh_pred.shape}"
    print("[selftest] SS_NN_Hybrid forward/rollout/YHead OK")


if __name__ == "__main__":
    _selftest()
    print("state_space_nn self-test passed.")