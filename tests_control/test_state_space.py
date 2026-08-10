"""Tests for ``src_control.models.state_space``."""
from __future__ import annotations

import numpy as np

from src_control.models.state_space import (
    LinearSS,
    kalman_filter,
    n4sid,
)


def test_n4sid_recovers_ar2():
    """N4SID should recover a simple AR(2) system to high accuracy."""
    rng = np.random.RandomState(0)
    A_true = np.array([[0.9, 0.1], [-0.2, 0.7]])
    B_true = np.array([[1.0], [0.5]])
    C_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    D_true = np.array([[0.0], [0.0]])
    n, m, p = 2, 1, 2

    T = 1500
    u = rng.randn(T, m)
    w = rng.randn(T, n) * 0.05
    v = rng.randn(T, p) * 0.05

    x = np.zeros((T, n))
    y = np.zeros((T, p))
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + B_true @ u[t - 1] + w[t]
        y[t] = C_true @ x[t] + D_true @ u[t] + v[t]

    model = n4sid(u, y, order=2, n_lags=8)
    err = np.linalg.norm(model.A - A_true, ord="fro")
    assert err < 0.1, f"A recovery error too high: {err:.4f}"
    assert model.n == 2 and model.m == 1 and model.p == 2


def test_kalman_with_mask():
    """Kalman filter with all observations should match state closely."""
    A = np.array([[0.95, 0.05], [-0.1, 0.9]])
    B = np.array([[0.5], [0.5]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    n, m, p = 2, 1, 2
    rng = np.random.RandomState(42)
    T = 200
    u = rng.randn(T, m) * 0.5
    x = np.zeros((T, n))
    y = np.zeros((T, p))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + B @ u[t - 1]
        y[t] = C @ x[t] + rng.randn(p) * 0.01

    x_filt, P_filt = kalman_filter(
        A, B, C, D, u, y, np.ones_like(y, dtype=bool),
        Q=np.eye(n) * 0.01, R=np.eye(p) * 0.01,
    )
    assert x_filt.shape == (T, n)
    assert P_filt.shape == (T, n, n)
    rmse = float(np.sqrt(np.mean((x_filt - x) ** 2)))
    assert rmse < 0.5, f"Kalman RMSE too high: {rmse:.4f}"


def test_linear_ss_rollout_shape():
    A = np.eye(3) * 0.9
    B = np.zeros((3, 4))
    C = np.zeros((2, 3))
    D = np.zeros((2, 4))
    model = LinearSS(A=A, B=B, C=C, D=D, n=3, m=4, p=2)
    u = np.random.randn(10, 4)
    y = model.rollout(u)
    assert y.shape == (10, 2)


def test_kalman_handles_missing_observations():
    """When y_mask is False everywhere, Kalman should reduce to prediction only."""
    A = np.array([[0.9]])
    B = np.array([[0.5]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    u = np.zeros((5, 1))
    y = np.zeros((5, 1))
    y_mask = np.zeros((5, 1), dtype=bool)  # no observations
    x_filt, _ = kalman_filter(A, B, C, D, u, y, y_mask,
                               Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
                               x0=np.array([1.0]))
    # State decays from 1.0 via A=0.9
    expected_first = 0.9 * 1.0
    assert np.isclose(x_filt[0, 0], expected_first, atol=1e-3)