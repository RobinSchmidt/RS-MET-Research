"""The :class:`FilterPair` result container."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from . import analysis
from ._util import is_symmetric, trim_zero_ends
from .nyquist import nyquist_error, verify_roundtrip
from .spec import Spec


def _stored(h: np.ndarray) -> int:
    e = trim_zero_ends(h)
    return (len(e) + 1) // 2 if is_symmetric(e) else len(e)


@dataclass
class FilterPair:
    """
    A perfect-reconstruction pair of upsampling kernel ``u`` and downsampling kernel ``d``.

    Convention: ``u`` has a DC gain of ``L`` (it compensates the zero-stuffing) and ``d`` has a DC gain
    of 1.  The chain ``zero-stuff -> u -> (processing) -> d -> keep every L-th sample`` returns the input
    delayed by ``latency`` base-rate samples when the processing is the identity.

    Attributes
    ----------
    u, d : ndarray
        The kernels.
    L : int
        Oversampling factor.
    latency : int
        Roundtrip latency in base-rate samples (the centre of ``u * d`` is at ``L * latency``).
    passband : float, optional
        Passband edge as a fraction of the original Nyquist frequency (needed for :meth:`metrics`).
    method : str
        How the pair was produced (provenance).
    info : dict
        Free-form extra information from the design routine.
    """

    u: np.ndarray
    d: np.ndarray
    L: int
    latency: int
    passband: float | None = None
    method: str = "unknown"
    info: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.u = np.array(self.u, dtype=float)
        self.d = np.array(self.d, dtype=float)
        if self.u.ndim != 1 or self.d.ndim != 1:
            raise ValueError("u and d must be 1-D arrays")
        if int(self.L) != self.L or self.L < 2:
            raise ValueError("L must be an integer >= 2")
        self.L, self.latency = int(self.L), int(self.latency)
        if self.latency < 0:
            raise ValueError("latency must be >= 0")

    # ------------------------------------------------------------------ basic properties
    @property
    def n_u(self) -> int:
        return len(self.u)

    @property
    def n_d(self) -> int:
        return len(self.d)

    @property
    def nyquist_error(self) -> float:
        """Deviation from the Nyquist(L) condition; ~1e-16 for an exact pair."""
        return nyquist_error(self.u, self.d, self.L, self.latency)

    @property
    def is_matched(self) -> bool:
        """True if ``u`` and ``d`` are the same kernel (one set of coefficients serves both filters)."""
        return self.n_u == self.n_d and bool(np.array_equal(self.u, self.d))

    @property
    def shaper_level(self) -> float:
        """
        Signal level at the nonlinearity relative to the input level, ``sum(u) / L``.

        1 for the pair convention (``u`` has DC gain ``L``); ``1/sqrt(L)`` for a matched kernel.  With a nonlinearity
        between the filters multiply by ``1 / shaper_level`` before it and by ``shaper_level`` after it (or use
        ``reference_roundtrip(..., compensate_levels=True)``) to make the shaper see the input level.  The identity
        roundtrip is the same either way.
        """
        return float(np.sum(self.u)) / self.L

    @property
    def stored_coeffs(self) -> int:
        """Coefficients that must be stored, exploiting symmetry, dropping zero end taps and sharing ``u = d``."""
        return _stored(self.u) if self.is_matched else _stored(self.u) + _stored(self.d)

    @property
    def macs_per_sample(self) -> int:
        """Multiplications per base-rate sample (polyphase, symmetry NOT exploited), zero end taps dropped."""
        return len(trim_zero_ends(self.u)) + len(trim_zero_ends(self.d))

    @property
    def spec(self) -> Spec:
        if self.passband is None:
            raise ValueError("this FilterPair has no passband set")
        return Spec(self.L, self.passband)

    # ------------------------------------------------------------------ operations
    def verify(self, tol: float = 1e-9, **kwargs) -> float:
        """Identity-roundtrip check on white noise; raises ``VerificationError`` if the error exceeds ``tol``."""
        return verify_roundtrip(self.u, self.d, self.L, self.latency, tol=tol, **kwargs)

    def swapped(self) -> "FilterPair":
        """
        The same pair with the roles exchanged: ``d`` (rescaled to gain ``L``) becomes the upsampler and
        ``u`` (rescaled to gain 1) the downsampler.

        The product ``u * d`` and therefore the identity roundtrip are unchanged.  With a nonlinear
        processing step between the filters the results are NOT identical, because the first filter
        determines what the nonlinearity sees.
        """
        gd = float(np.sum(self.d))
        if abs(gd) < 1e-12:
            raise ValueError("cannot swap: d has (almost) zero DC gain")
        c = self.L / gd
        return FilterPair(self.d * c, self.u / c, self.L, self.latency, self.passband,
                          method=f"{self.method}+swapped", info=dict(self.info))

    def metrics(self, n_grid: int = 4000) -> Dict[str, float]:
        """Stopband attenuation, passband ripple and transition-band peaks of both kernels (dB)."""
        s = self.spec
        gu, gd = float(np.sum(self.u)), float(np.sum(self.d))
        return {
            "u_stop_db": analysis.stopband_db(self.u, s, gu, n_grid),
            "d_stop_db": analysis.stopband_db(self.d, s, gd, n_grid),
            "u_ripple_db": analysis.passband_ripple_db(self.u, s, n_grid),
            "d_ripple_db": analysis.passband_ripple_db(self.d, s, n_grid),
            "u_transition_peak_db": analysis.transition_peak_db(self.u, s, gu, n_grid),
            "d_transition_peak_db": analysis.transition_peak_db(self.d, s, gd, n_grid),
            "nyquist_error": self.nyquist_error,
        }

    def __repr__(self) -> str:
        return (f"FilterPair(L={self.L}, n_u={self.n_u}, n_d={self.n_d}, latency={self.latency}, "
                f"nyquist_error={self.nyquist_error:.1e}, method={self.method!r})")
