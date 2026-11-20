"""
Prescribed transition-band roll-off for the oversampling filters.

Normalise the transition band to ``x in [0, 1]`` (``x = 0`` at the passband edge, ``x = 1`` at the stopband edge; the
original Nyquist frequency is at ``x = 1/2``).  The Nyquist(L) condition forces the PRODUCT of the two zero-phase
amplitudes, ``r(x) = A_u A_d / L``, to be point-symmetric: ``r(x) + r(1 - x) = 1`` (up to the tiny stopband images).
Any ``r`` with that property is allowed, e.g.

* ``"cosine"``  ``r = cos(pi x / 2)^2``  (raised cosine; smooth at both band edges, taps decay like 1/n^3),
* ``"linear"``  ``r = 1 - x``            (kinks at the band edges, taps decay like 1/n^2),
* ``"kaiser"``  ``r = 1 - F(x)`` where ``F`` is the running integral of the main lobe of the spectrum of a Kaiser window
  with parameter ``beta`` (this is what a Kaiser-windowed sinc does by itself).

``1 - x^p`` is NOT point-symmetric for ``p != 1`` and therefore not an admissible product.

The split between the filters is an exponent ``alpha`` in ``[0, 1]``:  ``A_u / L = r^alpha``,  ``A_d = r^(1 - alpha)``.
``alpha = 1/2`` is the matched (root-Nyquist) case, both filters at -3 dB at the original Nyquist frequency; ``alpha = 1``
means ``u`` carries the whole roll-off (a windowed-sinc ``u`` is close to that, and ``d`` then needs a compensating bump).
Both factors are monotone for any ``alpha``, so a pair that follows them has no bump.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("cosine", "linear", "kaiser")


def kaiser_integral_profile(beta: float, n: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """``(x, r)`` of the Kaiser-window main-lobe integral shape (``r(0) = 1``, ``r(1) = 0``, point-symmetric)."""
    nu0 = np.sqrt(beta ** 2 + np.pi ** 2)                    # first zero of the Kaiser window spectrum
    nu = np.linspace(-nu0, nu0, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_in, q_out = np.sqrt(np.maximum(beta ** 2 - nu ** 2, 0.0)), np.sqrt(np.maximum(nu ** 2 - beta ** 2, 0.0))
        W = np.where(np.abs(nu) < beta, np.sinh(q_in) / q_in, np.sin(q_out) / q_out)
    W[~np.isfinite(W)] = 1.0
    F = np.concatenate([[0.0], np.cumsum(0.5 * (W[1:] + W[:-1]) * np.diff(nu))])
    F /= F[-1]
    return np.linspace(0.0, 1.0, n), 1.0 - F


@dataclass(frozen=True)
class TransitionShape:
    """
    Prescribed transition roll-off: product shape ``kind``, split ``alpha`` and absolute tolerance ``tol``.

    ``tol`` is the half-width of the tube around the target (amplitudes relative to each filter's DC gain).
    """

    kind: str = "cosine"
    alpha: float = 0.5
    tol: float = 0.03
    beta: float = 4.0                  # only for kind="kaiser"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not self.tol > 0:
            raise ValueError("tol must be positive")

    def product(self, x) -> np.ndarray:
        """Target of ``A_u A_d / L`` at normalised transition positions ``x``."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        if self.kind == "cosine":
            return np.cos(0.5 * np.pi * x) ** 2
        if self.kind == "linear":
            return 1.0 - x
        xs, r = kaiser_integral_profile(self.beta)
        return np.interp(x, xs, r)

    def u_target(self, x) -> np.ndarray:
        """Target of ``A_u / L``."""
        return self.product(x) ** self.alpha

    def d_target(self, x) -> np.ndarray:
        """Target of ``A_d``."""
        return self.product(x) ** (1.0 - self.alpha)
