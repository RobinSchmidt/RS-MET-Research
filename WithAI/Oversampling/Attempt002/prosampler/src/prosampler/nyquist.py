"""
The perfect-reconstruction (Nyquist) condition and helpers to check it.

With ``h = u * d`` (convolution) and the centre of ``h`` at ``n0 = L * latency``, the oversampling
roundtrip ``zero-stuff -> u -> d -> decimate`` is the identity delayed by ``latency`` base-rate
samples if and only if

    h[n0 + L k] = delta[k]      for all integers k      (h is a Nyquist(L) / L-th band filter),

given the gain convention that ``u * d`` has a passband gain of ``L`` (``u``: L, ``d``: 1).
"""
from __future__ import annotations

import numpy as np

from .errors import VerificationError
from .process import reference_roundtrip


def nyquist_error(u, d, L: int, latency: int) -> float:
    """
    Largest deviation of ``(u*d)[L*latency + L k]`` from ``delta[k]`` over all ``k``.

    Zero (up to roundoff) means the roundtrip is an exact identity delayed by ``latency`` base-rate samples.
    """
    h = np.convolve(np.asarray(u, float), np.asarray(d, float))
    n0 = int(L) * int(latency)
    if not 0 <= n0 < len(h):
        return 1.0                                   # the centre tap is not even inside u*d
    idx = np.arange(n0 % L, len(h), L)
    return float(np.max(np.abs(h[idx] - (idx == n0))))


def symmetric_latency(n_u: int, n_d: int, L: int) -> int:
    """
    Roundtrip latency (base-rate samples) of a linear-phase pair with odd lengths ``n_u`` and ``n_d``.

    The centre of ``u * d`` is ``(n_u + n_d - 2) / 2`` and must be a multiple of ``L`` so that the latency
    is a whole number of base-rate samples.  Raises ``ValueError`` (with the nearest valid ``n_d``) if not.
    """
    if n_u % 2 == 0 or n_d % 2 == 0:
        raise ValueError("linear-phase kernels need odd lengths")
    n0 = (n_u + n_d - 2) // 2
    if n0 % L:
        raise ValueError(
            f"n_u + n_d - 2 must be a multiple of 2*L = {2 * L} (centre of u*d must be a multiple of L); "
            f"for n_u = {n_u} the nearest valid n_d are {partner_length(n_u, L, n0 // L)} and "
            f"{partner_length(n_u, L, n0 // L + 1)}")
    return n0 // L


def partner_length(n_u: int, L: int, latency: int) -> int:
    """Length ``n_d`` that gives the requested ``latency`` for a linear-phase pair: ``2 L latency + 2 - n_u``."""
    return 2 * L * latency + 2 - n_u


def verify_roundtrip(u, d, L: int, latency: int, tol: float = 1e-9, n_samples: int = 4096,
                     seed: int = 0, raise_on_fail: bool = True) -> float:
    """
    Run white noise through the reference chain with an identity waveshaper and return ``max|y - x|``.

    Raises :class:`~prosampler.errors.VerificationError` if the error exceeds ``tol`` (unless
    ``raise_on_fail=False``).
    """
    x = np.random.default_rng(seed).standard_normal(n_samples)
    err = float(np.max(np.abs(reference_roundtrip(x, u, d, L, latency) - x)))
    if raise_on_fail and not err <= tol:
        raise VerificationError(f"identity roundtrip error {err:.3e} exceeds tolerance {tol:.1e}")
    return err
