"""Reference (non-polyphase) implementation of the oversampling chain, for testing and verification."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def reference_roundtrip(x, u, d, L: int, latency: int,
                        shaper: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    ``x -> zero-stuff by L -> u -> shaper -> d -> keep every L-th sample``, aligned by ``latency``.

    Straightforward and slow (full convolutions); meant as a ground truth, not for real-time use.
    With ``shaper=None`` (identity) and a perfect-reconstruction pair, the result equals ``x``.

    Parameters
    ----------
    latency : int
        Roundtrip latency in BASE-rate samples, i.e. the centre of ``u * d`` is at ``L * latency``.
    shaper : callable, optional
        Memoryless (or any) processing applied at the high rate between the two filters.
    """
    x = np.asarray(x, dtype=float)
    up = np.zeros(len(x) * L)
    up[::L] = x
    v = np.convolve(up, u)
    if shaper is not None:
        v = np.asarray(shaper(v), dtype=float)
    v = np.convolve(v, d)
    y = v[::L][latency:latency + len(x)]
    if len(y) < len(x):
        raise ValueError("latency too large for the filter lengths: output shorter than input")
    return y
