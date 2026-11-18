"""
Design specification: oversampling factor, passband edge and (optionally) stopband attenuation.

Conventions used throughout prosampler
--------------------------------------
* ``L`` is the integer oversampling factor.  Filters run at the HIGH rate (L times the base rate).
* Frequencies are in radians per HIGH-rate sample, ``w`` in ``[0, pi]``.  The original Nyquist
  frequency is therefore ``pi / L``.
* ``passband`` is the passband edge as a FRACTION of the original Nyquist frequency, i.e. the edge is
  at ``passband * pi / L`` (``0 < passband < 1``).  At a base rate of 44.1 kHz, ``passband = 0.9``
  means 19.8 kHz.
* The stopband starts at ``(2 - passband) * pi / L``, the mirror image of the passband edge around
  the original Nyquist frequency.  This is forced by the Nyquist(L) condition (the transition band
  of ``u * d`` must be antisymmetric around ``pi / L``), so it is not a free parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Frequency grid on [0, pi] with boolean masks for the three bands."""

    w: np.ndarray
    passband: np.ndarray
    stopband: np.ndarray
    transition: np.ndarray


@dataclass(frozen=True)
class Spec:
    """
    Specification of an oversampling filter design.

    Parameters
    ----------
    L : int
        Oversampling factor (>= 2).
    passband : float
        Passband edge as a fraction of the original Nyquist frequency, ``0 < passband < 1``.
    atten_db : float, optional
        Required stopband attenuation in dB (positive number), relative to the passband gain.
        Only needed by routines that choose filter lengths automatically.
    """

    L: int
    passband: float
    atten_db: Optional[float] = None

    def __post_init__(self):
        if int(self.L) != self.L or self.L < 2:
            raise ValueError(f"L must be an integer >= 2, got {self.L!r}")
        object.__setattr__(self, "L", int(self.L))
        if not 0.0 < self.passband < 1.0:
            raise ValueError(
                f"passband must be a fraction of the original Nyquist frequency in (0, 1), got {self.passband!r}")
        if self.atten_db is not None and not self.atten_db > 0:
            raise ValueError(f"atten_db must be positive (e.g. 80 for -80 dB), got {self.atten_db!r}")

    @property
    def passband_edge(self) -> float:
        """Passband edge in rad per high-rate sample."""
        return self.passband * np.pi / self.L

    @property
    def stopband_edge(self) -> float:
        """Stopband edge in rad per high-rate sample, ``(2 - passband) * pi / L``."""
        return (2.0 - self.passband) * np.pi / self.L

    @property
    def transition_width(self) -> float:
        """Width of the transition band in rad per high-rate sample."""
        return self.stopband_edge - self.passband_edge

    def grid(self, n: int = 3000) -> Grid:
        """
        Uniform frequency grid of about ``n`` points on [0, pi] with passband / stopband / transition masks.

        The exact passband and stopband edge frequencies are always included (the stopband peak of a good design
        sits right at its edge, so a grid that misses it would under-report the peak by an amount that depends on
        the grid size).
        """
        w = np.unique(np.concatenate([np.linspace(0.0, np.pi, n), [self.passband_edge, self.stopband_edge]]))
        pas = w <= self.passband_edge
        stp = w >= self.stopband_edge
        return Grid(w=w, passband=pas, stopband=stp, transition=~(pas | stp))


def passband_from_hz(f_pass_hz: float, fs_hz: float) -> float:
    """
    Convenience: passband edge in Hz -> fraction of the original Nyquist frequency ``fs_hz / 2``.

    ``fs_hz`` is the BASE sample rate (before oversampling).
    """
    if not 0 < f_pass_hz < fs_hz / 2:
        raise ValueError("need 0 < f_pass_hz < fs_hz / 2")
    return float(f_pass_hz / (fs_hz / 2.0))
