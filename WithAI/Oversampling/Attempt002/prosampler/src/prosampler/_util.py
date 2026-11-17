"""Small internal helpers shared by several modules."""
from __future__ import annotations

import numpy as np


def is_symmetric(h, rel_tol: float = 1e-9) -> bool:
    """True if ``h`` equals its time reversal (linear phase, type I/II), up to ``rel_tol`` of its peak."""
    h = np.asarray(h, dtype=float)
    return bool(np.max(np.abs(h - h[::-1])) <= rel_tol * np.max(np.abs(h)))


def trim_zero_ends(h, rel_tol: float = 1e-12) -> np.ndarray:
    """Drop leading / trailing taps that are numerically zero (such end taps are forced by the Nyquist condition)."""
    h = np.asarray(h, dtype=float)
    nz = np.nonzero(np.abs(h) > rel_tol * np.max(np.abs(h)))[0]
    return h[nz[0]:nz[-1] + 1] if len(nz) else h


def mirror_matrix(n: int, drop_ends: bool = False) -> np.ndarray:
    """
    Matrix ``S`` with ``g = S @ theta`` for a symmetric kernel of odd length ``n``.

    ``theta[k] = g[mid - k] = g[mid + k]`` with ``mid = (n - 1) // 2``, so ``theta`` has ``mid + 1`` entries
    (the centre tap and the right half).  With ``drop_ends=True`` the two end taps are not parametrised
    (they are fixed to zero), giving ``mid`` entries.
    """
    if n % 2 == 0:
        raise ValueError("symmetric parametrisation needs an odd length")
    mid = (n - 1) // 2
    m = mid if drop_ends else mid + 1
    S = np.zeros((n, m))
    for k in range(m):
        S[mid - k, k] = 1.0
        S[mid + k, k] = 1.0
    return S


def conv_rows(u, n_d: int, idx) -> np.ndarray:
    """Rows ``idx`` of the convolution matrix of ``u``: ``(conv_rows(u, n_d, idx) @ d)[m] == np.convolve(u, d)[idx[m]]``."""
    u = np.asarray(u, dtype=float)
    idx = np.asarray(idx)
    j = np.arange(n_d)
    C = np.zeros((len(idx), n_d))
    for m, p in enumerate(idx):
        i = p - j
        ok = (i >= 0) & (i < len(u))
        C[m, j[ok]] = u[i[ok]]
    return C
