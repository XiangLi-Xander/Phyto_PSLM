"""
IUPred3 wrapper for per-residue disorder score computation.

This module provides :func:`compute_iupred_scores`, which returns the
per-residue IUPred3 disorder profile as a 1-D float32 array using the
'long' mode with 'medium' Savitzky-Golay smoothing.

To avoid redundant disk I/O, the IUPred3 energy matrix and histogram
are cached in memory after the first read via monkey-patching of the
underlying ``iupred3_lib`` helpers.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IUPRED3_DIR = os.path.join(_PROJECT_ROOT, "iupred3")
if _IUPRED3_DIR not in sys.path:
    sys.path.insert(0, _IUPRED3_DIR)

import iupred3_lib  # noqa: E402

_MATRIX_CACHE: dict = {}
_HISTO_CACHE: dict = {}

_orig_read_matrix = iupred3_lib.read_matrix
_orig_read_histo = iupred3_lib.read_histo


def _cached_read_matrix(path: str):
    if path not in _MATRIX_CACHE:
        _MATRIX_CACHE[path] = _orig_read_matrix(path)
    return _MATRIX_CACHE[path]


def _cached_read_histo(path: str):
    if path not in _HISTO_CACHE:
        _HISTO_CACHE[path] = _orig_read_histo(path)
    return _HISTO_CACHE[path]


iupred3_lib.read_matrix = _cached_read_matrix
iupred3_lib.read_histo = _cached_read_histo

_cached_read_matrix(
    os.path.join(_IUPRED3_DIR, "data", "iupred2_long_energy_matrix")
)
_cached_read_histo(os.path.join(_IUPRED3_DIR, "data", "long_histogram"))


def compute_iupred_scores(
    sequence: str,
    mode: str = "long",
    smoothing: str = "medium",
) -> np.ndarray:
    """Compute per-residue IUPred3 disorder scores.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence (already truncated if necessary).
    mode : str, optional
        IUPred3 prediction mode (default: "long").
    smoothing : str, optional
        Savitzky-Golay smoothing level (default: "medium").

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(L,)``, where ``L = len(sequence)``.
    """
    L = len(sequence)
    if L == 0:
        return np.zeros(0, dtype=np.float32)
    scores, _ = iupred3_lib.iupred(sequence, mode=mode, smoothing=smoothing)
    return np.asarray(scores, dtype=np.float32)
