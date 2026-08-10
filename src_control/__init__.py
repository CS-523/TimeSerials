"""src_control package — modern control theory pipeline for time_serials project.

Modules:
    data_loader        : CSV discovery and canonical parsing
    preprocess         : cleaning, missing-value handling, scaling, splitting
    analysis           : correlation / MI / PCA / lag analysis
    models             : linear state-space (N4SID/Kalman) and hybrid SS-NN
    optimization       : MPC and Pareto-frontier optimizer over (x3,x4,x6,x8)
    visualization      : plotting helpers
    utils              : metrics, seeding

This package is independent of the legacy ``src/`` directory at the repo root.
Use :func:`src_control.import_legacy.add_legacy_path` to import reusable
components from ``src/path_integrators.py`` without renaming or modifying them.
"""

__version__ = "0.1.0"