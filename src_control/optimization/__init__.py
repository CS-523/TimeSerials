"""Optimization: choose future x3/x4/x6/x8 to maximize y4.

Exposes the differentiable reward-maximization routine :func:`maximize_y4`,
which composes two frozen ``SS_NN_Hybrid`` models (the x-forecaster for the
default trajectory, and the y4-predictor for the reward) over the controllable
inputs x3/x4/x6/x8 only.
"""
from .optimizer import (
    CONTROL_BOUNDS_RAW,
    CONTROL_NAMES,
    DECISION_IDX,
    FIXED_IDX,
    maximize_y4,
    standardized_bounds,
)

__all__ = [
    "maximize_y4",
    "standardized_bounds",
    "DECISION_IDX",
    "FIXED_IDX",
    "CONTROL_NAMES",
    "CONTROL_BOUNDS_RAW",
]
