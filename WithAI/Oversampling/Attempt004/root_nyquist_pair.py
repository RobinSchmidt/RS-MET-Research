"""
Design of a matched FIR up/downsampling filter pair u[n], d[n] such that
upsample -> (identity processing) -> downsample is transparent, i.e. a pure
delay. Approach:

1. Design an overall M-th band (Nyquist) lowpass prototype h[n] of length
   Lu + Ld - 1 via windowed sinc (cutoff = pi/M guarantees zeros at n = k*M).
2. Lift H(w) if needed so it is nonnegative, then take a spectral factor
   u[n] (minimum phase). Set d[n] = u[-n] (time-reversed, i.e. the matched
   filter). Then |U(w) D(w)| = |H(w)|, so the roundtrip magnitude response
   equals the Nyquist prototype's: flat passband, linear phase overall.
"""

import numpy as np
from scipy.signal import firwin, minimum_phase, lfilter, upfirdn


def design_nyquist_pair(M, Lu, Ld, lift=1e-12):
    """
    Design upsampling kernel u[n] and downsampling kernel d[n] for
    oversampling factor M, such that up->identity->down is transparent.

    Parameters
    ----------
    M  : int   oversampling factor
    Lu : int   length of the upsampling (anti-imaging) kernel
    Ld : int   length of the downsampling (anti-aliasing) kernel

    Constraints: u and d are a matched pair d[n] = u[-n] from one spectral
    factor, so Lu == Ld is required. The cascade center tap must fall on a
    multiple of M (decimation phase), i.e. the length L must satisfy
    L = k*M + 1 for integer k. The roundtrip delay is (L-1)/M input samples.

    Returns
    -------
    u, d : ndarray, ndarray
    """
    if Lu != Ld:
        raise ValueError("Lu == Ld is required for an exact spectral-factor pair")
    # 1. Nyquist (M-th band) prototype, total length of the cascade:
    Lh = Lu + Ld - 1
    c = (Lh - 1) // 2                # center tap index of the cascade
    if c % M != 0:
        # decimation phase must hit the center tap: c must be a multiple of M
        raise ValueError(
            f"Require kernel length L = k*M + 1, got (L-1) % M = {c % M}")
    # Windowed sinc with cutoff pi/M: zeros at (n-c) = k*M hold analytically.
    n = np.arange(Lh)
    h = (1.0 / M) * np.sinc((n - c) / M) * np.kaiser(Lh, beta=8.6)

    # 2. Make the zero-phase amplitude A(w) = Re(e^{jwc} H(w)) nonnegative so
    #    a spectral factor exists (lifting delta at center keeps the zeros).
    Nfft = 16 * Lh
    W = np.fft.rfft(h, Nfft)
    wk = np.pi * np.arange(len(W)) / (len(W) - 1)
    Amin = np.real(W * np.exp(1j * wk * c)).min()
    if Amin < 0:
        h[c] += -Amin + lift

    # 3. Spectral factorization: u minimum-phase, d = time-reversed u.
    #    Accuracy of the two methods varies with M/L; try both, keep the
    #    factor whose convolution best preserves the Nyquist zeros.
    def residual(f):
        f = np.pad(f, (0, Lu - len(f))) if len(f) < Lu else f[:Lu]
        hh = np.convolve(f, f[::-1])
        off = hh[::M].copy()
        off[c // M] = 0
        return np.max(np.abs(off)), f
    cands = [residual(minimum_phase(h, method=m, n_fft=2**16))
             for m in ("homomorphic", "hilbert")]
    u = min(cands, key=lambda t: t[0])[1].copy()
    d = u[::-1].copy()

    # 5. Normalize so the cascade center tap is exactly 1: the roundtrip is
    #    then a pure delay of c/M input samples (c = Lu + Ld - 2).
    hc = np.convolve(u, d)[c]
    u /= np.sqrt(hc)
    d /= np.sqrt(hc)
    return u, d


def roundtrip(x, M, u, d):
    """upsample -> identity -> downsample"""
    y = upfirdn(u, x, up=M)          # upsample + anti-imaging filter
    y = upfirdn(d, y, down=M)        # anti-aliasing filter + downsample
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for M, Lu, Ld in [(2, 31, 31), (4, 65, 65), (4, 129, 129), (8, 129, 129)]:
        u, d = design_nyquist_pair(M, Lu, Ld)
        x = rng.standard_normal(2000)
        y = roundtrip(x, M, u, d)
        D = (Lu + Ld - 2) // (2 * M)    # expected roundtrip delay (input samples)
        err = np.max(np.abs(y[D:D + len(x)] - x))
        h = np.convolve(u, d)
        c = (Lu + Ld - 2) // 2
        off = h[np.arange(len(h)) % M == 0]
        off[c // M] = 0                 # exclude the center tap
        print(f"M={M}, Lu={Lu}, Ld={Ld}: delay={D}, "
              f"max|y-x|={err:.3e}, max off-center h[kM]={np.max(np.abs(off)):.3e}")
