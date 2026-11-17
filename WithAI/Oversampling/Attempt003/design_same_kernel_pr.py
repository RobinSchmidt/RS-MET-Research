#!/usr/bin/env python3
"""
Design one symmetric FIR kernel u for both up/down sampling (u=d)
such that upsample->u->u->downsample is PR (identity up to delay).

Requires: numpy, scipy
    pip install numpy scipy
"""

import argparse
import numpy as np
from scipy.signal import firwin, upfirdn
from scipy.optimize import minimize, NonlinearConstraint


def full_from_unique(a: np.ndarray) -> np.ndarray:
    """
    a[0] = center tap, a[k] = taps at +/-k
    returns odd-length symmetric FIR u
    """
    M = len(a) - 1
    N = 2 * M + 1
    u = np.empty(N, dtype=float)
    u[M] = a[0]
    for k in range(1, M + 1):
        u[M - k] = a[k]
        u[M + k] = a[k]
    return u


def unique_from_full(u: np.ndarray) -> np.ndarray:
    M = (len(u) - 1) // 2
    a = np.empty(M + 1, dtype=float)
    a[0] = u[M]
    a[1:] = u[M + 1 :]
    return a


def pr_moments(u: np.ndarray, L: int) -> np.ndarray:
    """
    Returns [c0, c1, ..., cR], where:
      c0 = sum_n u[n]^2
      ck = sum_n u[n] u[n-kL], k>=1
    PR requires c0=1 and ck=0 for k>=1.
    """
    N = len(u)
    R = (N - 1) // L
    c = np.zeros(R + 1, dtype=float)
    c[0] = np.dot(u, u)
    for k in range(1, R + 1):
        lag = k * L
        c[k] = np.dot(u[lag:], u[:-lag])
    return c


def lowpass_target_mag(f: np.ndarray, fp: float, fs: float) -> np.ndarray:
    """
    f in [0,1], where 1 = Nyquist.
    Cosine transition from fp to fs.
    """
    t = np.zeros_like(f)
    t[f <= fp] = 1.0
    tr = (f > fp) & (f < fs)
    if np.any(tr):
        x = (f[tr] - fp) / (fs - fp)
        t[tr] = 0.5 * (1.0 + np.cos(np.pi * x))
    return t


def design_same_kernel_pr(
    L=4,
    span=12,
    pass_frac=0.90,
    stop_frac=1.10,
    nfft=16384,
    w_stop=8.0,
    w_pass=1.0,
    maxiter=1500,
):
    """
    L         : oversampling factor
    span      : half-length in original-rate samples
                N = 2*span*L + 1 (odd, symmetric, delay aligns to L-grid)
    pass_frac : passband edge as fraction of ideal cutoff (ideal cutoff = 1/L Nyquist units)
    stop_frac : stopband start as fraction of ideal cutoff
    
    If optimization fails, increase span first (more taps gives the constraints room).
    """
    if span < 1:
        raise ValueError("span must be >= 1")

    N = 2 * span * L + 1
    fp = pass_frac / L
    fs = stop_frac / L

    if not (0 < fp < fs < 1):
        raise ValueError("Need 0 < pass_frac/L < stop_frac/L < 1")

    # Initial symmetric lowpass guess near ideal anti-imaging cutoff
    u0 = firwin(N, cutoff=1.0 / L, window=("kaiser", 8.6), scale=False)
    u0 /= np.sqrt(np.dot(u0, u0))  # satisfy c0 roughly
    a0 = unique_from_full(u0)

    f = np.linspace(0.0, 1.0, nfft // 2 + 1)
    tgt = lowpass_target_mag(f, fp, fs)
    w = np.where(f >= fs, w_stop, w_pass)

    def objective(a):
        u = full_from_unique(a)
        U = np.fft.rfft(u, n=nfft)
        mag = np.abs(U)
        e = w * (mag - tgt)
        return np.mean(e * e)

    # Exact PR constraints (numerically): c0=1, ck=0
    R = (N - 1) // L
    target = np.zeros(R + 1, dtype=float)
    target[0] = 1.0
    nlc = NonlinearConstraint(
        lambda a: pr_moments(full_from_unique(a), L),
        lb=target,
        ub=target,
    )

    res = minimize(
        objective,
        a0,
        method="SLSQP",
        constraints=[nlc],
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not res.success:
        raise RuntimeError(
            f"Optimization failed: {res.message}\n"
            f"Try larger span, looser pass/stop edges, or different weights."
        )

    u = full_from_unique(res.x)
    return u, res


def verify_roundtrip(u: np.ndarray, L: int, Nsig=48000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(Nsig)

    y_hi = upfirdn(u, x, up=L)
    y = upfirdn(u, y_hi, down=L)

    delay_lo = (len(u) - 1) // L
    y_aligned = y[delay_lo : delay_lo + Nsig]
    err = y_aligned - x

    h = np.convolve(u, u)
    c = len(u) - 1
    R = (len(u) - 1) // L
    pr_samples = np.array([h[c + m * L] for m in range(-R, R + 1)])
    m0 = R
    off = np.max(np.abs(np.delete(pr_samples, m0))) if len(pr_samples) > 1 else 0.0

    print(f"L={L}, taps={len(u)}, roundtrip delay={delay_lo} input-rate samples")
    print(f"RMS error:     {np.sqrt(np.mean(err**2)):.3e}")
    print(f"Max abs error: {np.max(np.abs(err)):.3e}")
    print(f"h[c] (target 1): {pr_samples[m0]:.12f}")
    print(f"max |h[c+mL]|, m!=0 (target 0): {off:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4, help="oversampling factor")
    p.add_argument("--span", type=int, default=12, help="half-length in input-rate samples")
    p.add_argument("--pass-frac", type=float, default=0.90, help="pass edge / ideal cutoff")
    p.add_argument("--stop-frac", type=float, default=1.10, help="stop edge / ideal cutoff")
    p.add_argument("--w-stop", type=float, default=8.0, help="stopband objective weight")
    p.add_argument("--w-pass", type=float, default=1.0, help="passband objective weight")
    p.add_argument("--nfft", type=int, default=16384)
    p.add_argument("--out", type=str, default="u_same_kernel.txt")
    args = p.parse_args()

    u, res = design_same_kernel_pr(
        L=args.L,
        span=args.span,
        pass_frac=args.pass_frac,
        stop_frac=args.stop_frac,
        nfft=args.nfft,
        w_stop=args.w_stop,
        w_pass=args.w_pass,
    )

    np.savetxt(args.out, u)
    print(f"Saved kernel to {args.out}")
    print(f"Objective: {res.fun:.6e}, iterations: {res.nit}")

    verify_roundtrip(u, args.L)


if __name__ == "__main__":
    main()