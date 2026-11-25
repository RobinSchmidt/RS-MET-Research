"""
Lowpass kernels that can serve as the (anti-imaging) upsampling kernel ``u``.

Any kernel works as a starting point for the partner design, these are just convenient ones.
Gain convention: the kernel is normalised to a DC gain (sum of taps) of ``gain``, default ``L`` -
the gain that compensates the zero-stuffing of an L-fold upsampler.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.signal import get_window

from .analysis import stopband_db
from .errors import SpecNotReached
from .spec import Spec

WindowLike = Union[str, tuple, np.ndarray, Callable[[int], np.ndarray]]


def kaiser_beta(atten_db: float) -> float:
    """Kaiser window beta for a given stopband attenuation (Kaiser's empirical formula)."""
    A = float(atten_db)
    if A > 50:
        return 0.1102 * (A - 8.7)
    if A >= 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    return 0.0


def kaiser_length_estimate(spec: Spec) -> int:
    """Kaiser's estimate of the (odd) kernel length for ``spec.atten_db`` over the spec's transition band."""
    if spec.atten_db is None:
        raise ValueError("spec.atten_db is required to estimate a length")
    n = int(np.ceil((spec.atten_db - 8.0) / (2.285 * spec.transition_width))) + 1
    return n | 1


def windowed_sinc(L: int, n_taps: int, window: WindowLike = "hann", gain: Optional[float] = None) -> np.ndarray:
    """
    Windowed-sinc lowpass with cutoff ``pi / L``, symmetric around its centre tap (linear phase).

    ``sinc(n / L)`` has zeros at all non-zero multiples of ``L``; the window keeps that property, so the
    result alone is already a Nyquist(L) kernel (``u[centre + L k] = delta[k]`` before gain normalisation).

    Parameters
    ----------
    window : str, tuple, ndarray or callable
        Anything accepted by :func:`scipy.signal.get_window` (e.g. ``"hann"``, ``("kaiser", 8.0)``),
        an array of length ``n_taps``, or a callable ``n -> array``.
    gain : float, optional
        DC gain (sum of taps).  Default ``L``.
    """
    if n_taps < 1:
        raise ValueError("n_taps must be positive")
    if callable(window):
        win = np.asarray(window(n_taps), dtype=float)
    elif isinstance(window, np.ndarray):
        win = window.astype(float)
    else:
        win = get_window(window, n_taps, fftbins=False)
    if len(win) != n_taps:
        raise ValueError(f"window has length {len(win)}, expected {n_taps}")
    n = np.arange(n_taps) - (n_taps - 1) / 2.0
    u = np.sinc(n / L) * win
    return u * ((L if gain is None else gain) / u.sum())


def kaiser_sinc(L: int, n_taps: int, atten_db: Optional[float] = None, beta: Optional[float] = None,
                gain: Optional[float] = None) -> np.ndarray:
    """Kaiser-windowed sinc; give either ``beta`` or ``atten_db`` (from which beta is derived)."""
    if (beta is None) == (atten_db is None):
        raise ValueError("give exactly one of beta and atten_db")
    b = kaiser_beta(atten_db) if beta is None else beta
    return windowed_sinc(L, n_taps, ("kaiser", b), gain)


def design_kaiser_sinc(spec: Spec, gain: Optional[float] = None, max_factor: float = 6.0) -> np.ndarray:
    """
    Shortest odd-length Kaiser-sinc whose stopband (from ``spec.stopband_rad``) is below ``-spec.atten_db``.

    Starts below the Kaiser length estimate and grows in steps of 2.  Raises
    :class:`~prosampler.errors.SpecNotReached` if not met within ``max_factor`` times the estimate.
    """
    if spec.atten_db is None:
        raise ValueError("spec.atten_db is required")
    g = spec.L if gain is None else gain
    est = kaiser_length_estimate(spec)
    n = max(5, int(0.7 * est)) | 1
    while n <= max_factor * est:
        u = kaiser_sinc(spec.L, n, atten_db=spec.atten_db, gain=g)
        if stopband_db(u, spec, gain=g) <= -spec.atten_db:
            return u
        n += 2
    raise SpecNotReached(f"no Kaiser-sinc up to {int(max_factor * est)} taps reaches {spec.atten_db} dB for {spec}")
