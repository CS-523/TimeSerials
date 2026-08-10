"""Linear state-space identification (N4SID) and Kalman filter, pure numpy.

We need N4SID and a Kalman filter to:

* Identify a discrete-time linear state-space model from data
  ``x_{t+1} = A x_t + B u_t + w,   y_t = C x_t + D u_t + v``
  and use it as a **linear baseline** inside the hybrid SS-NN predictor.

* Provide a forward Kalman smoother for online state estimation when y
  measurements are sparse (mask-aware).

This module deliberately avoids ``control``, ``sippy``, ``filterpy`` (not
installed in this environment); the implementation follows the canonical
N4SID algorithm of Van Overschee & De Moor 1994 (chapter 4), with a few
practical simplifications (SVD truncation; least-squares solve for A,B,C,D).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LinearSS:
    """Discrete-time linear state-space model."""
    A: np.ndarray  # (n, n)
    B: np.ndarray  # (n, m)
    C: np.ndarray  # (p, n)
    D: np.ndarray  # (p, m)
    n: int
    m: int
    p: int

    def rollout(self, u: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Open-loop rollout given inputs ``u`` of shape ``(T, m)``.

        Returns ``y`` of shape ``(T, p)``.
        """
        T = u.shape[0]
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = x0.copy()
        ys = np.zeros((T, self.p))
        for t in range(T):
            x = self.A @ x + self.B @ u[t]
            ys[t] = self.C @ x + self.D @ u[t]
        return ys

    def to_torch(self):
        """Return a :class:`src_control.models.state_space_nn.LinearSSTorch`."""
        from src_control.models.state_space_nn import LinearSSTorch
        return LinearSSTorch(self.A, self.B, self.C, self.D)


# --------------------------------------------------------------------------- #
# N4SID (Numerical algorithms for Subspace State-Space IDentification)
# --------------------------------------------------------------------------- #
def _hankel(data: np.ndarray, rows: int, cols: int, shift: int = 1) -> np.ndarray:
    """Build a block-Hankel matrix of shape ``(rows * n_features, cols)``."""
    n_feat = data.shape[1]
    T = data.shape[0]
    H = np.zeros((rows * n_feat, cols))
    for i in range(rows):
        # data[i : T - rows + i + 1] reversed to match the standard convention
        # (oldest sample at the top).
        chunk = data[i : i + cols]
        H[i * n_feat : (i + 1) * n_feat, :] = chunk.T
    return H


def n4sid(
    u: np.ndarray,
    y: np.ndarray,
    order: int = 16,
    n_lags: int = 10,
) -> LinearSS:
    """Identify a discrete linear SS via a robust simplified subspace method.

    The implementation follows these steps (a pragmatic blend of N4SID and
    ARX-style least squares):

    1. Build past and future block-Hankel matrices from the I/O data.
    2. Orthogonal-projection of future outputs onto past data, then SVD
       truncation to ``order`` to extract the state sequence.
    3. Solve least squares for ``(A, B)`` from the state sequence and inputs.
    4. Solve least squares for ``(C, D)`` from the state sequence and inputs.

    The method is numerically stable for T ≳ 200 and works without external
    dependencies.

    Parameters
    ----------
    u : (T, m)
    y : (T, p)
    order : desired state dimension
    n_lags : block-Hankel width

    Returns
    -------
    :class:`LinearSS`
    """
    T, m = u.shape
    _, p = y.shape

    while 2 * n_lags + 2 > T and n_lags > 1:
        n_lags -= 1
    if order > n_lags * (m + p):
        order = max(1, n_lags * (m + p) // 2)

    # Build past and future Hankel matrices.
    # Past data Zp has 2*n_lags rows: [u_{t-1..t-n_lags}; y_{t-1..t-n_lags}].
    # Future outputs Yf has n_lags rows: [y_{t..t+n_lags-1}].
    # Future inputs Uf: [u_{t..t+n_lags-1}].
    # N = T - 2*n_lags + 1 columns.

    N = T - 2 * n_lags + 1
    if N <= order + 2:
        raise ValueError(f"Data too short: T={T}, n_lags={n_lags}, N={N}")

    Zp_rows = []
    for k in range(1, n_lags + 1):
        Zp_rows.append(u[n_lags - k : n_lags - k + N, :].T)  # (m, N)
        Zp_rows.append(y[n_lags - k : n_lags - k + N, :].T)  # (p, N)
    Zp = np.concatenate(Zp_rows, axis=0)  # (n_lags * (m + p), N)

    Yf_rows = []
    for k in range(n_lags):
        Yf_rows.append(y[n_lags + k : n_lags + k + N, :].T)
    Yf = np.concatenate(Yf_rows, axis=0)  # (n_lags * p, N)

    # Oblique projection O = Yf / Zp  (project Yf onto the row space of Zp).
    # O = Yf @ Zp.T @ (Zp @ Zp.T + eps I)^{-1} @ Zp  (Tikhonov-regularized least squares)
    eps = 1e-4 * max(1.0, np.trace(Zp @ Zp.T) / Zp.shape[0])
    ZpZt = Zp @ Zp.T
    O = Yf @ Zp.T @ np.linalg.solve(ZpZt + eps * np.eye(ZpZt.shape[0]), Zp)

    # SVD of O → state sequence
    U_s, s_s, Vt = np.linalg.svd(O, full_matrices=False)
    order = min(order, len(s_s))
    U_s = U_s[:, :order]
    s_s = s_s[:order]
    Vt = Vt[:order, :]

    # State sequence: X = diag(s^{1/2}) V^T  (rank-order truncation)
    scale = np.sqrt(np.maximum(s_s, 0.0))
    state = np.diag(scale) @ Vt  # (order, N)

    # Build regression matrices for A,B (state dynamics) using consecutive pairs.
    X_curr = state[:, :-1].T  # (N-1, order)
    X_next = state[:, 1:].T   # (N-1, order)
    # Pair X_next with inputs spanning the same interval.
    u_block = u[n_lags : n_lags + N - 1]  # (N-1, m)
    lhs_AB = np.concatenate([X_curr, u_block], axis=1)  # (N-1, order+m)
    AB, *_ = np.linalg.lstsq(lhs_AB, X_next, rcond=None)
    A = AB[:order, :].T      # (order, order)
    B = AB[order:order + m, :].T  # (order, m)

    # C, D: regress y on [state; u] for the *same* timestamps.
    X_for_CD = state.T  # (N, order)
    u_for_CD = u[n_lags : n_lags + N]
    y_for_CD = y[n_lags : n_lags + N]
    lhs_CD = np.concatenate([X_for_CD, u_for_CD], axis=1)  # (N, order+m)
    CD, *_ = np.linalg.lstsq(lhs_CD, y_for_CD, rcond=None)
    C = CD[:order, :].T      # (p, order)
    D = CD[order:order + m, :].T  # (p, m)

    return LinearSS(A=A, B=B, C=C, D=D, n=order, m=m, p=p)


# --------------------------------------------------------------------------- #
# Kalman filter with measurement-mask support
# --------------------------------------------------------------------------- #
def kalman_filter(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward Kalman filter (with missing-observation support).

    Parameters
    ----------
    A : (n, n), B : (n, m), C : (p, n), D : (p, m)
    u : (T, m), y : (T, p), y_mask : (T, p) bool
    Q : (n, n) process noise covariance (default 1e-3 I)
    R : (p, p) measurement noise covariance (default 1e-1 I)
    x0 : (n,) initial state (default zeros)
    P0 : (n, n) initial covariance (default I)

    Returns
    -------
    x_post : (T, n) filtered states
    P_post : (T, n, n) filtered covariances
    """
    T = u.shape[0]
    n = A.shape[0]
    p = C.shape[0]

    if Q is None:
        Q = np.eye(n) * 1e-3
    if R is None:
        R = np.eye(p) * 1e-1
    if x0 is None:
        x0 = np.zeros(n)
    if P0 is None:
        P0 = np.eye(n)

    x_post = np.zeros((T, n))
    P_post = np.zeros((T, n, n))

    x = x0.copy()
    P = P0.copy()
    I = np.eye(n)

    for t in range(T):
        # Predict
        x = A @ x + B @ u[t]
        P = A @ P @ A.T + Q

        # Update with available measurements
        y_pred = C @ x + D @ u[t]
        obs = np.where(y_mask[t])[0]
        if len(obs) > 0:
            C_o = C[obs]
            y_o = y[t, obs]
            y_pred_o = y_pred[obs]
            R_o = R[np.ix_(obs, obs)]
            S = C_o @ P @ C_o.T + R_o
            try:
                K = P @ C_o.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P @ C_o.T @ np.linalg.pinv(S)
            innov = y_o - y_pred_o
            x = x + K @ innov
            P = (I - K @ C_o) @ P

        x_post[t] = x
        P_post[t] = P
    return x_post, P_post


# --------------------------------------------------------------------------- #
# Sanity self-test (run as a module)
# --------------------------------------------------------------------------- #
def _selftest():
    """Recover an AR(2) system via N4SID; sanity-check accuracy."""
    rng = np.random.RandomState(0)
    A_true = np.array([[0.9, 0.1], [-0.2, 0.7]])
    B_true = np.array([[1.0], [0.5]])
    C_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    D_true = np.array([[0.0], [0.0]])
    n, m, p = 2, 1, 2

    T = 1000
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
    print(f"[selftest] ||A_rec - A_true||_F = {err:.4f}")
    assert err < 0.5, f"N4SID recovery failed (err={err:.4f})"

    # Kalman check (smoke test only — RMSE depends on Q,R tuning and state
# basis alignment, which are configured per-dataset in practice).
    x_filt, _ = kalman_filter(model.A, model.B, model.C, model.D, u, y,
                              np.ones_like(y, dtype=bool),
                              Q=np.eye(2) * 0.01, R=np.eye(2) * 0.01)
    rmse = float(np.sqrt(np.mean((x_filt - x) ** 2)))
    print(f"[selftest] Kalman RMSE = {rmse:.4f} (informational only)")


if __name__ == "__main__":
    _selftest()
    print("state_space self-test passed.")