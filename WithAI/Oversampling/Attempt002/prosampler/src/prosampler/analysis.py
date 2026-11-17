"""Frequency-domain measurements of filter kernels (all relative to a given passband gain)."""
from __future__ import annotations

import numpy as np

from .spec import Spec


def freqz(h, w) -> np.ndarray:
    """Complex frequency response of FIR kernel ``h`` at radian frequencies ``w`` (direct evaluation)."""
    h = np.asarray(h, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.exp(-1j * np.outer(w, np.arange(len(h)))) @ h


def stopband_db(h, spec: Spec, gain: float = 1.0, n_grid: int = 4000) -> float:
    """Peak magnitude in the stopband in dB, relative to ``gain`` (use L for u, 1 for d)."""
    g = spec.grid(n_grid)
    return float(20 * np.log10(np.max(np.abs(freqz(h, g.w[g.stopband]))) / gain + 1e-300))


def passband_ripple_db(h, spec: Spec, n_grid: int = 4000) -> float:
    """Peak-to-peak magnitude variation over the passband in dB (max/min)."""
    g = spec.grid(n_grid)
    m = np.abs(freqz(h, g.w[g.passband]))
    return float(20 * np.log10(m.max() / m.min()))


def transition_peak_db(h, spec: Spec, gain: float = 1.0, n_grid: int = 4000) -> float:
    """Peak magnitude between the passband and stopband edges in dB, relative to ``gain``."""
    g = spec.grid(n_grid)
    return float(20 * np.log10(np.max(np.abs(freqz(h, g.w[g.transition]))) / gain + 1e-300))


def group_delay_passband(h, spec: Spec, n_grid: int = 4000) -> float:
    """Mean group delay over the passband in (high-rate) samples."""
    g = spec.grid(n_grid)
    w = g.w[g.passband][1:]
    n = np.arange(len(h))
    H, dH = freqz(h, w), freqz(np.asarray(h) * n, w)
    return float(np.mean(np.real(dH / H)))


def band_energy_matrix(n_taps: int, w_lo: float, w_hi: float) -> np.ndarray:
    """
    Matrix ``Q`` with ``h^T Q h = integral_{w_lo}^{w_hi} |H(w)|^2 dw`` for any real kernel ``h`` of length ``n_taps``.

    ``Q[i, j] = integral cos((i - j) w) dw = (sin((i-j) w_hi) - sin((i-j) w_lo)) / (i - j)`` and ``w_hi - w_lo`` on the
    diagonal.  With ``w_hi = pi`` it is the stopband-energy matrix of the old KKT design script.
    """
    k = np.arange(n_taps)
    diff = k[:, None] - k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (np.sin(diff * w_hi) - np.sin(diff * w_lo)) / diff
    Q[diff == 0] = w_hi - w_lo
    return Q
