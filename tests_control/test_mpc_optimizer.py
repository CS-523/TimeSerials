"""Tests for ``src_control.optimization.mpc_optimizer``."""
from __future__ import annotations

import numpy as np
import torch

from src_control.models.state_space_nn import SS_NN_Hybrid, YHead
from src_control.optimization.mpc_optimizer import (
    DECISION_INDICES,
    OptConfig,
    pareto_front,
)


def _build_random_model() -> tuple[SS_NN_Hybrid, YHead]:
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=8, hidden=16, window=3)
    yhead = YHead(window=8, hidden=16)
    return model, yhead


def test_pareto_front_trivial():
    """Single point is always on the Pareto front."""
    nd = pareto_front([(1.0, 2.0)])
    assert nd == [0]


def test_pareto_front_dominance():
    """One point dominates another when both coordinates are >= and one is >."""
    points = [(1.0, 1.0), (2.0, 2.0), (0.5, 3.0), (1.5, 1.5)]
    nd = pareto_front(points)
    # (2,2) and (0.5,3) are non-dominated
    assert set(nd) == {1, 2}


def test_optimize_pareto_smoke():
    """Run optimization with 1 start + tiny horizon on a random model."""
    model, yhead = _build_random_model()
    cfg = OptConfig(horizon=4, n_starts=1, max_iter=3, device="cpu")
    x_history = np.random.randn(20, 8).astype(np.float32)
    res = _run_pareto(model, yhead, x_history, cfg)
    assert "points" in res
    assert len(res["points"]) == len(cfg.weights)
    assert "baseline" in res
    # Each point is a (y4_sum, Y_pred) tuple.
    for p in res["points"]:
        assert len(p) == 2
        assert np.isfinite(p[0]) and np.isfinite(p[1])


def _run_pareto(model, yhead, x_history, cfg):
    """Local helper to import lazily and avoid module-level side effects."""
    from src_control.optimization.mpc_optimizer import optimize_pareto
    return optimize_pareto(
        model=model,
        yhead=yhead,
        x_history=x_history,
        cfg=cfg,
        fixed_x1_x2_x5=(0.0, 0.0, 0.0),
    )