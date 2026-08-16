"""Differentiable optimizer: choose future x3/x4/x6/x8 to maximize y4.

Two frozen ``SS_NN_Hybrid`` models are supplied by the caller:

* ``x_forecaster`` (``dim_y=8``) — predicts the "no-intervention" default
  future x1–x8 trajectory from a past context via ``forecast(...)``.
* ``y_predictor`` (``dim_y=4``) — maps a full x1–x8 sequence to y1–y4 via
  ``forward(...)``; y4 = ``output[..., 3]`` is the reward.

Only the future *controllable* inputs x3/x4/x6/x8 (columns 2/3/5/7) are free,
differentiable variables. The non-controllable x1/x2/x5/x7 follow the forecast
and are held fixed. This is the "free" vs "non-free" split: the four free
columns are the setpoints an operator can actually set, not the ones that are
merely predictable.

Everything runs in the *standardized* space the models were trained on.
Physical bounds are converted with the caller-supplied ``x_mean``/``x_scale``;
the caller is responsible for denormalizing results back to physical units.
"""
from __future__ import annotations

import numpy as np
import torch

# Controllable (decision) column indices: x3, x4, x6, x8 (0-based).
DECISION_IDX = (2, 3, 5, 7)
# Non-controllable (follow-the-forecast) column indices: x1, x2, x5, x7.
FIXED_IDX = (0, 1, 4, 6)
CONTROL_NAMES = ("x3", "x4", "x6", "x8")

# Physical bounds in raw space, ordered to match DECISION_IDX.
CONTROL_BOUNDS_RAW = np.array(
    [
        [0.0, 110.0],       # x3
        [26.0, 36.0],       # x4
        [5000.0, 50000.0],  # x6
        [0.0, 1500.0],      # x8
    ],
    dtype=np.float64,
)

# Column assembly order for the (H, 8) trajectory: each of the 8 x-columns is
# taken either from the free control tensor or the fixed forecast tensor.
# ``ctrl`` stores the position inside x_ctrl (ordered by DECISION_IDX); ``fixed``
# stores the actual column index inside fixed_future (which is 0..7).
_DECISION_POS = {j: k for k, j in enumerate(DECISION_IDX)}
_COL_PARTS = [
    ("ctrl", _DECISION_POS[j]) if j in _DECISION_POS else ("fixed", j)
    for j in range(8)
]


def standardized_bounds(x_mean, x_scale, control_bounds_raw=CONTROL_BOUNDS_RAW):
    """Convert physical control bounds to standardized bounds, shape ``(4, 2)``."""
    mean = np.asarray(x_mean, dtype=np.float64)[list(DECISION_IDX)]
    scale = np.asarray(x_scale, dtype=np.float64)[list(DECISION_IDX)]
    cb = np.asarray(control_bounds_raw, dtype=np.float64)
    lo = (cb[:, 0] - mean) / scale
    hi = (cb[:, 1] - mean) / scale
    return np.stack([lo, hi], axis=1)


def _assemble_future(x_ctrl, fixed_future):
    """Assemble ``(H, 8)`` future x from ``(H, 4)`` control + ``(H, 8)`` fixed.

    Uses ``torch.cat`` of per-column slices (rather than in-place assignment)
    so gradients flow back to ``x_ctrl`` while ``fixed_future`` stays detached.
    """
    cols = []
    for kind, pos in _COL_PARTS:
        if kind == "ctrl":
            cols.append(x_ctrl[:, pos : pos + 1])
        else:
            cols.append(fixed_future[:, pos : pos + 1])
    return torch.cat(cols, dim=1)


def maximize_y4(
    x_ctx,
    x_forecaster,
    y_predictor,
    x_mean,
    x_scale,
    horizon,
    y4_phase_mask=None,
    control_bounds_raw=None,
    n_starts=5,
    max_iter=200,
    lr=0.05,
    effort_penalty=0.0,
    seed=0,
    device=None,
):
    """Maximize mean future y4 over the controllable x3/x4/x6/x8.

    Parameters
    ----------
    x_ctx : torch.Tensor
        Past context ``(C, 8)`` in standardized x-space.
    x_forecaster, y_predictor : nn.Module
        Frozen models (see module docstring); must already be on ``device``.
    x_mean, x_scale : array-like
        ``(8,)`` scaler statistics used to map physical bounds into the
        standardized space.
    horizon : int
        Number of future steps ``H`` to optimize.
    y4_phase_mask : array-like of bool, optional
        ``(H,)`` mask marking the future steps that are real y4 sampling
        positions (the daily schedule). Reward is averaged over these steps
        only. ``None`` (default) = every step observed (legacy whole-window
        mean).
    control_bounds_raw : (4, 2) array-like, optional
        Physical-space lower/upper bounds for the controllable x3/x4/x6/x8,
        ordered to match ``DECISION_IDX``. ``None`` (default) uses the
        permissive ``CONTROL_BOUNDS_RAW``; pass the training-distribution range
        to keep the optimization in-distribution.
    n_starts : int
        Multi-start count; start 0 = forecast (clamped), the rest are uniform
        random within bounds.
    max_iter : int
        Optimizer steps per start (Adam).
    lr : float
        Adam learning rate.
    effort_penalty : float
        ``λ`` for the control-effort term ``λ·‖x_ctrl − x_default‖²``. Default
        0 disables it; larger values keep the trajectory near the forecast to
        avoid physically-unreachable jumps.
    seed : int
        Seed for the random multi-starts.

    Returns
    -------
    dict with keys (all normalized / standardized space):
        x_default_norm : (H, 8) forecast default trajectory
        x_opt_norm     : (H, 8) optimized trajectory (only ctrl cols changed)
        y4_default_norm: (H,)   y4 along the default trajectory
        y4_opt_norm    : (H,)   y4 along the optimized trajectory
        reward_default_norm : float = mean y4 over the observed sampling phase
        reward_opt_norm     : float = mean y4 over the observed sampling phase
            (falls back to the last step when no step is observed in the window)
    """
    if device is None:
        device = x_ctx.device
    x_ctx = x_ctx.to(device).float().detach()
    C = x_ctx.shape[0]

    # Which future steps are real y4 sampling positions (daily schedule). None
    # means "every step observed" (legacy whole-window mean).
    if y4_phase_mask is None:
        phase = torch.ones(horizon, dtype=torch.bool, device=device)
    else:
        phase = torch.as_tensor(y4_phase_mask, dtype=torch.bool, device=device)
        if phase.numel() != horizon:
            raise ValueError(
                f"y4_phase_mask length {phase.numel()} != horizon {horizon}"
            )

    def aligned_mean(y4):
        """Mean y4 over the observed phase; fall back to the last step if none."""
        return y4[phase].mean() if phase.any() else y4[-1]

    # Freeze both models; only the control tensor remains differentiable.
    for m in (x_forecaster, y_predictor):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # Default "no-intervention" future trajectory (autoregressive rollout).
    with torch.no_grad():
        x_fut_default = x_forecaster.forecast(
            x_ctx.unsqueeze(0), horizon, x_future_gt=None, teacher_forcing=0.0
        )[0]  # (H, 8)

    if control_bounds_raw is None:
        control_bounds_raw = CONTROL_BOUNDS_RAW
    bounds = standardized_bounds(x_mean, x_scale, control_bounds_raw)
    lo = torch.tensor(bounds[:, 0], dtype=torch.float32, device=device)  # (4,)
    hi = torch.tensor(bounds[:, 1], dtype=torch.float32, device=device)  # (4,)

    fixed_future = x_fut_default.detach().clone()  # (H, 8), held fixed
    x_default_ctrl = x_fut_default[:, DECISION_IDX].clone()  # (H, 4)

    def reward_of(x_ctrl):
        """Mean y4 over the observed sampling phase for control trajectory x_ctrl."""
        future = _assemble_future(x_ctrl, fixed_future)  # (H, 8)
        u = torch.cat([x_ctx, future], dim=0).unsqueeze(0)  # (1, C+H, 8)
        y = y_predictor(u, y_prev=None, teacher_forcing=0.0)  # (1, C+H, 4)
        return aligned_mean(y[0, C:, 3])

    rng = np.random.RandomState(seed)
    # Initialize the best with the "no-intervention" default so the optimizer
    # never returns a result worse than the baseline (gradient ascent from a
    # feasible warm-start must not regress).
    with torch.no_grad():
        reward_default = float(reward_of(x_default_ctrl).item())
    best = {"reward": reward_default, "x_ctrl": x_default_ctrl.clamp(lo, hi).detach().clone()}

    for k in range(n_starts):
        if k == 0:
            x_init = x_default_ctrl.clamp(lo, hi).clone()
        else:
            r = torch.tensor(rng.rand(horizon, 4), dtype=torch.float32, device=device)
            x_init = (lo + (hi - lo) * r).clone()
        x_init.requires_grad_(True)

        opt = torch.optim.Adam([x_init], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad()
            reward = reward_of(x_init)
            effort = ((x_init - x_default_ctrl) ** 2).mean()
            loss = -reward + effort_penalty * effort
            loss.backward()
            opt.step()
            with torch.no_grad():
                x_init.clamp_(lo, hi)

        with torch.no_grad():
            r_final = float(reward_of(x_init).item())
        if r_final > best["reward"]:
            best["reward"] = r_final
            best["x_ctrl"] = x_init.detach().clone()

    # Final y4 trajectories (default vs optimized) for reporting/plotting.
    with torch.no_grad():
        u_def = torch.cat([x_ctx, x_fut_default], dim=0).unsqueeze(0)
        y4_default = y_predictor(u_def, y_prev=None, teacher_forcing=0.0)[0, C:, 3]
        x_opt = _assemble_future(best["x_ctrl"], fixed_future)
        u_opt = torch.cat([x_ctx, x_opt], dim=0).unsqueeze(0)
        y4_opt = y_predictor(u_opt, y_prev=None, teacher_forcing=0.0)[0, C:, 3]

    return {
        "x_default_norm": x_fut_default.detach().cpu().numpy(),
        "x_opt_norm": x_opt.detach().cpu().numpy(),
        "y4_default_norm": y4_default.detach().cpu().numpy(),
        "y4_opt_norm": y4_opt.detach().cpu().numpy(),
        "reward_default_norm": float(aligned_mean(y4_default).item()),
        "reward_opt_norm": float(aligned_mean(y4_opt).item()),
    }
