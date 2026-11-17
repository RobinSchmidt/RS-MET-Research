"""
Optional plots (need matplotlib: ``pip install prosampler[plots]``).

Every function accepts an optional ``ax`` (or ``axes``) to draw into and returns the matplotlib object(s), so
figures can be composed freely.  Frequencies are shown as a fraction of the ORIGINAL Nyquist frequency
(1.0 = Nyquist of the base rate; the oversampled Nyquist frequency is L).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .analysis import freqz
from .pair import FilterPair


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install prosampler[plots]") from e
    return plt


def plot_responses(pair: FilterPair, ax=None, n_grid: int = 4000, floor_db: float = -160.0, max_freq: Optional[float] = None):
    """
    Magnitude responses of ``u`` and ``d`` (each relative to its own DC gain) with the passband and stopband shaded.

    ``max_freq`` (in units of the original Nyquist frequency) limits the x axis; default is up to ``L`` (everything).
    """
    plt = _plt()
    ax = ax if ax is not None else plt.subplots(figsize=(8, 4))[1]
    L = pair.L
    w = np.linspace(0, np.pi, n_grid)
    x = w * L / np.pi
    for h, label in ((pair.u, "u"), (pair.d, "d")):
        mag = np.abs(freqz(h, w)) / abs(np.sum(h))
        ax.plot(x, np.maximum(20 * np.log10(mag + 1e-300), floor_db), label=f"{label} ({len(h)} taps)", lw=1)
    if pair.passband is not None:
        ax.axvspan(0, pair.passband, color="C2", alpha=0.12, label="passband")
        ax.axvspan(2 - pair.passband, L, color="C3", alpha=0.08, label="stopband")
    ax.set_xlim(0, max_freq if max_freq is not None else L)
    ax.set_ylim(floor_db, 10)
    ax.set_xlabel("frequency / original Nyquist")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_kernels(pair: FilterPair, axes=None):
    """Impulse responses of ``u`` and ``d`` (two stacked axes)."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=False)
    for ax, h, label in zip(axes, (pair.u, pair.d), ("u", "d")):
        ax.plot(np.arange(len(h)), h, lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("tap")
    return axes


def plot_path(pair: FilterPair, ax=None):
    """Stopband attenuation of ``u`` and ``d`` along the path recorded by :func:`prosampler.rebalance`."""
    if "path" not in pair.info:
        raise ValueError("this pair has no rebalance path (info['path']); run rebalance() first")
    plt = _plt()
    ax = ax if ax is not None else plt.subplots(figsize=(8, 4))[1]
    nu, nd, su, sd = (np.array(c) for c in zip(*pair.info["path"]))
    ax.plot(nu, su, label="u stopband")
    ax.plot(nu, sd, label="d stopband")
    ax.plot(nu, np.maximum(su, sd), "k--", lw=0.8, label="worse of the two")
    ax.set_xlabel("n_u   (n_d = total - n_u)")
    ax.set_ylabel("stopband peak (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
