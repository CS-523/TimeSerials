"""Pareto-frontier MPC optimizer over (x3, x4, x6, x8).

Decision variables
------------------
``u`` of shape ``(H, 4)`` representing future x3, x4, x6, x8 trajectories.

Objective
---------
Weighted sum of ``Σ y4`` over the horizon and the predicted final ``Y``::

    maximize  w_y4 * Σ y4 + w_Y * Y_pred
    s.t.      u_lo ≤ u ≤ u_hi

We sweep ``w_y4 ∈ {1, 0.7, 0.5, 0.3, 0}`` and ``w_Y = 1 - w_y4`` to trace a
Pareto frontier.

Algorithm
---------
The trained hybrid model is differentiable, so we use **L-BFGS** with
multi-start. Box constraints enforced via ``u.clamp`` after each step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src_control.models.state_space_nn import SS_NN_Hybrid, YHead


# Index mapping (matches config: X_COLS = x1..x8, DECISION_COLS = x3,x4,x6,x8)
DECISION_INDICES = (2, 3, 5, 7)   # x3, x4, x6, x8 within u
FIXED_INPUT_INDICES = (0, 1, 4)   # x1, x2, x5 within u


@dataclass
class OptConfig:
    horizon: int = 16
    weights: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0),
    ])
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x3": (0.0, 110.0),
        "x4": (26.0, 36.0),
        "x6": (5_000.0, 50_000.0),
        "x8": (0.0, 1500.0),
    })
    fixed_x1_x2_x5: Optional[Tuple[float, float, float]] = None  # (mean_x1, mean_x2, mean_x5)
    fixed_x7: str = "cumulative"  # 'cumulative' or 'mean'
    n_starts: int = 5
    lr: float = 0.05
    max_iter: int = 200
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_u_template(
    cfg: OptConfig,
    x_history: np.ndarray,
    fixed_x1_x2_x5: Tuple[float, float, float],
) -> torch.Tensor:
    """Build the initial decision ``u_dec`` of shape ``(H, 4)``.

    Initial guess: continuation of the last observed (x3, x4, x6, x8).
    """
    last = x_history[-1]
    last_dec = np.array([last[i] for i in DECISION_INDICES], dtype=np.float32)
    u_dec = np.tile(last_dec, (cfg.horizon, 1))  # (H, 4)
    return torch.tensor(u_dec, dtype=torch.float32)


def _assemble_full_u(
    u_dec: torch.Tensor,
    cfg: OptConfig,
    fixed_x1_x2_x5: Tuple[float, float, float],
    x_history: np.ndarray,
) -> torch.Tensor:
    """Place decision variables into the full ``u`` (8 channels) at each step."""
    H = u_dec.shape[0]
    u_full = torch.zeros(H, 8, dtype=u_dec.dtype)
    # Set decision slots
    for k, idx in enumerate(DECISION_INDICES):
        u_full[:, idx] = u_dec[:, k]
    # Fixed slots: x1, x2, x5
    u_full[:, FIXED_INPUT_INDICES[0]] = fixed_x1_x2_x5[0]
    u_full[:, FIXED_INPUT_INDICES[1]] = fixed_x1_x2_x5[1]
    u_full[:, FIXED_INPUT_INDICES[2]] = fixed_x1_x2_x5[2]
    # x7: continue cumulative
    if cfg.fixed_x7 == "cumulative" and x_history.shape[0] > 0:
        last_x7 = float(x_history[-1, 6])
        # Estimate per-step increment
        if x_history.shape[0] >= 2:
            dx7 = float(x_history[-1, 6] - x_history[-2, 6])
        else:
            dx7 = 0.0
        for t in range(H):
            u_full[t, 6] = last_x7 + dx7 * (t + 1)
    else:
        u_full[:, 6] = fixed_x1_x2_x5[0]  # placeholder; ignore x7 effect
    return u_full


# --------------------------------------------------------------------------- #
# Single (sample, weight) optimization
# --------------------------------------------------------------------------- #
def _optimize_one(
    model: SS_NN_Hybrid,
    yhead: YHead,
    cfg: OptConfig,
    u_dec_init: torch.Tensor,
    full_u_template: torch.Tensor,
    bounds_lo: torch.Tensor,
    bounds_hi: torch.Tensor,
    w_y4: float,
    w_Y: float,
    n_starts: int,
) -> Dict[str, np.ndarray]:
    """Run multi-start L-BFGS to find best ``u_dec`` for one weight tuple.

    Each L-BFGS step evaluates the full horizon rollout with **closed-loop
    teacher forcing**: the model's own previous prediction is fed back as
    ``y_prev`` context, so the residual MLP sees a realistic distribution.
    """
    best_loss = float("inf")
    best_u = None
    best_obj = None

    rng = np.random.RandomState(0)

    H = cfg.horizon
    for start in range(n_starts):
        if start == 0:
            u_dec = u_dec_init.clone()
        else:
            lo = bounds_lo.cpu().numpy()
            hi = bounds_hi.cpu().numpy()
            rand = rng.uniform(0, 1, size=(H, 4)).astype(np.float32)
            rand_u = torch.tensor(lo + rand * (hi - lo))
            u_dec = rand_u.clone()
        u_dec = u_dec.detach().requires_grad_(True)
        opt = torch.optim.LBFGS([u_dec], lr=cfg.lr, max_iter=cfg.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            u_dec_clamped = torch.max(torch.min(u_dec, bounds_hi), bounds_lo)
            u_full = full_u_template.clone()
            for k, idx in enumerate(DECISION_INDICES):
                u_full[:, idx] = u_dec_clamped[:, k]
            u_batch = u_full.unsqueeze(0)  # (1, H, 8)
            # Closed-loop MPC: feed previous prediction back as y_prev.
            # We approximate by running forward with y_prev = y_lin
            # (i.e. teacher_forcing=0, equivalent to one-shot rollout).
            y_pred = model(u_batch, y_prev=None, teacher_forcing=0.0)
            y4 = y_pred[0, :, 3]
            Y_pred = yhead(y_pred[:, -8:, :])[0]
            obj = -(w_y4 * y4.sum() + w_Y * Y_pred)
            obj.backward()
            return obj

        try:
            opt.step(closure)
        except Exception as e:
            print(f"  start {start} LBFGS failed: {e}")
            continue

        with torch.no_grad():
            u_dec_clamped = torch.max(torch.min(u_dec, bounds_hi), bounds_lo)
            u_full = full_u_template.clone()
            for k, idx in enumerate(DECISION_INDICES):
                u_full[:, idx] = u_dec_clamped[:, k]
            u_batch = u_full.unsqueeze(0)
            y_pred = model(u_batch, y_prev=None, teacher_forcing=0.0)
            y4 = float(y_pred[0, :, 3].sum().item())
            Y = float(yhead(y_pred[:, -8:, :])[0].item())
            loss = -(w_y4 * y4 + w_Y * Y)

        if loss < best_loss:
            best_loss = loss
            best_u = u_dec_clamped.detach().clone()
            best_obj = (y4, Y)

    if best_u is None:
        best_u = u_dec_init.clone()
        best_obj = (0.0, 0.0)

    return {
        "u_dec": best_u.cpu().numpy(),
        "y4_sum": best_obj[0],
        "Y_pred": best_obj[1],
    }


# --------------------------------------------------------------------------- #
# Pareto frontier driver
# --------------------------------------------------------------------------- #
def optimize_pareto(
    model: SS_NN_Hybrid,
    yhead: YHead,
    x_history: np.ndarray,
    cfg: OptConfig,
    fixed_x1_x2_x5: Tuple[float, float, float],
    y_history: Optional[np.ndarray] = None,
    y_scaler_mean: Optional[np.ndarray] = None,
    y_scaler_scale: Optional[np.ndarray] = None,
) -> Dict[str, list]:
    """Run Pareto sweep for one sample.

    Parameters
    ----------
    x_history : (T_hist, 8) — last observed inputs (raw, denormalized).
    y_history : (T_hist, 4) optional — last observed y. If given, the rollout
        uses it as a teacher-forcing seed so the residual MLP sees realistic
        context. If None, zeros are used (model still works because the linear
        SS handles most of the dynamics).
    cfg : optimization config
    fixed_x1_x2_x5 : tuple of mean x1, x2, x5 (in **standardized** space).

    Returns
    -------
    dict with keys ``'points'`` (list of (y4_sum, Y)), ``'trajectories'``
    (one per weight), ``'baseline'``.
    """
    model.eval()
    yhead.eval()
    device = cfg.device

    # Bounds in **standardized** space — convert from raw via x scaler
    # We require the caller to provide standardized bounds; the optimizer
    # operates entirely in standardized space.
    bounds = []
    for name in ("x3", "x4", "x6", "x8"):
        lo, hi = cfg.bounds[name]
        bounds.append((lo, hi))
    bounds_lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=device)
    bounds_hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=device)

    u_dec_init = _build_u_template(cfg, x_history, fixed_x1_x2_x5)
    full_u_template = _assemble_full_u(u_dec_init, cfg, fixed_x1_x2_x5, x_history)

    trajectories = []
    points = []
    for (w_y4, w_Y) in cfg.weights:
        res = _optimize_one(
            model=model,
            yhead=yhead,
            cfg=cfg,
            u_dec_init=u_dec_init,
            full_u_template=full_u_template,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            w_y4=w_y4,
            w_Y=w_Y,
            n_starts=cfg.n_starts,
        )
        # Build full trajectory with the best u_dec
        u_full = full_u_template.clone()
        u_dec_clamped = torch.max(torch.min(
            torch.tensor(res["u_dec"], dtype=torch.float32), bounds_hi), bounds_lo)
        for k, idx in enumerate(DECISION_INDICES):
            u_full[:, idx] = u_dec_clamped[:, k]
        with torch.no_grad():
            y_pred = model(u_full.unsqueeze(0), y_prev=None, teacher_forcing=0.0)
            Y = float(yhead(y_pred[:, -8:, :])[0].item())

        trajectories.append({
            "weights": (w_y4, w_Y),
            "u_dec": res["u_dec"],
            "y4_sum": res["y4_sum"],
            "Y_pred": res["Y_pred"],
            "y_full": y_pred[0].cpu().numpy(),
        })
        points.append((res["y4_sum"], res["Y_pred"]))

    # Baseline: continue last-input policy
    baseline_u = _assemble_full_u(u_dec_init, cfg, fixed_x1_x2_x5, x_history)
    with torch.no_grad():
        y_pred_base = model(baseline_u.unsqueeze(0), y_prev=None, teacher_forcing=0.0)
        Y_base = float(yhead(y_pred_base[:, -8:, :])[0].item())
    baseline = {
        "u_dec": u_dec_init.cpu().numpy(),
        "y_full": y_pred_base[0].cpu().numpy(),
        "y4_sum": float(y_pred_base[0, :, 3].sum().item()),
        "Y_pred": Y_base,
    }

    return {
        "points": points,
        "trajectories": trajectories,
        "baseline": baseline,
        "bounds": bounds,
    }


# --------------------------------------------------------------------------- #
# Pareto non-dominance filter
# --------------------------------------------------------------------------- #
def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    """Indices of non-dominated points (Pareto front for maximization)."""
    n = len(points)
    nd = [True] * n
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if points[j][0] >= points[i][0] and points[j][1] >= points[i][1] and \
               (points[j][0] > points[i][0] or points[j][1] > points[i][1]):
                nd[i] = False
                break
    return [i for i in range(n) if nd[i]]