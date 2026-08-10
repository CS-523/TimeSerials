"""Add the repo's legacy ``src/`` directory to ``sys.path``.

The legacy directory contains :mod:`path_integrators` (with ``MambaLiteSSM`` and
other reusable blocks) and ``train_path_integrators_gemini`` (training-loop
pattern reference). We treat it as **read-only**: no symbol is renamed, no file
is modified.

Usage::

    from src_control.import_legacy import add_legacy_path
    add_legacy_path()                     # idempotent
    from path_integrators import MambaLiteSSM  # works
"""
from __future__ import annotations

import os
import sys

_LEGACY_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)
_LEGENCY_SRC_ALT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)


def add_legacy_path() -> str:
    """Insert the legacy ``src/`` directory at the front of ``sys.path``.

    Returns the absolute path actually inserted. Idempotent.
    """
    path = _LEGACY_SRC
    if not os.path.isdir(path):
        return ""
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def legacy_src_path() -> str:
    """Absolute path of the legacy ``src/`` directory (for diagnostics)."""
    return _LEGACY_SRC