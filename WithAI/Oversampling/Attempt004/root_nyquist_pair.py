"""
Design of a matched FIR up/downsampling filter pair u[n], d[n] such that
upsample -> (identity processing) -> downsample is transparent, i.e. a pure
delay. Approach:

1. Design an overall M-th band (Nyquist) lowpass prototype h[n] of length
   Lu + Ld - 1 via windowed sinc (cutoff = pi/M guarantees zeros at n = k*M).
2. Lift H(w) if needed so its zero-phase amplitude is nonnegative, then
   take a spectral factor u[n]. Two flavors:
   - phase="min":    u minimum phase, d[n] = u[-n] (matched filter).
   - phase="linear": u symmetric (zero phase), u[n] = u[-n], so d = u.
In both cases |U(w) D(w)| = |H(w)|, so the roundtrip magnitude response
equals the Nyquist prototype's: flat passband, linear phase overall.
"""

import numpy as np
from scipy.signal import minimum_phase, upfirdn


def _symmetric_spectral_root(h, L, n_fft=2**16):
    """
    Symmetric (zero-phase) length-L kernel u with u*u ~= h, obtained by
    truncating the inverse DFT of sqrt(A(w)), where A(w) is the zero-phase
    amplitude of h. This is the only symmetric root consistent with a good
    magnitude response: U(w) is real, so U(w)^2 = A(w) forces
    U(w) = sqrt(A(w)) up to signs at stopband nulls, and the exact square
    root has infinite length. Truncation leaves a residual in the Nyquist
    taps of u*u of order 1/L (the sqrt kernel inherits the windowed-sinc's
    ~1/n tail decay), i.e. the roundtrip is transparent only to about
    -50 dB for typical lengths, improving linearly with L. (Enforcing the
    taps exactly via Newton iterations is possible in principle but wrecks
    the magnitude response -- the tap-constraint Jacobian is nearly
    singular in the symmetric parametrization.)
    """
    c = L - 1                        # center tap of u*u (h's center)
    W = np.fft.rfft(h, n_fft)
    wk = 2 * np.pi * np.arange(len(W)) / n_fft
    A = np.real(W * np.exp(1j * wk * c))
    uf = np.fft.irfft(np.sqrt(np.maximum(A, 0)), n_fft)
    k = (L - 1) // 2
    return np.concatenate([uf[n_fft - k:], uf[:k + 1]])


def design_nyquist_pair(M, Lu, Ld, lift=1e-12, phase="min"):
    """
    Design upsampling kernel u[n] and downsampling kernel d[n] for
    oversampling factor M, such that up->identity->down is transparent.

    Parameters
    ----------
    M     : int   oversampling factor
    Lu    : int   length of the upsampling (anti-imaging) kernel
    Ld    : int   length of the downsampling (anti-aliasing) kernel
    phase : "min"    -> u minimum phase, d = time-reversed u (matched pair).
                        Roundtrip identity to ~1e-7 or better. Recommended.
            "linear" -> u symmetric, d = u (both kernels identical, no
                        reversal). Optimal per-kernel magnitude response
                        (sqrt of the prototype's), but the roundtrip
                        identity holds only to ~1e-3 (about -50 dB,
                        improving ~linearly with L): an exact symmetric
                        square root of the prototype does not exist in
                        finite length, and the truncated root's residual
                        decays slowly (~1/L).

    Constraints: u and d come from one spectral factor, so Lu == Ld is
    required. The cascade center tap must fall on a multiple of M
    (decimation phase), i.e. the length L must satisfy L = k*M + 1 for
    integer k. The roundtrip delay is (L-1)/M input samples.

    Returns
    -------
    u, d : ndarray, ndarray
    """
    if Lu != Ld:
        raise ValueError("Lu == Ld is required for an exact spectral-factor pair")
    if phase not in ("min", "linear"):
        raise ValueError('phase must be "min" or "linear"')
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

    # 3. Spectral factorization.
    if phase == "linear":
        u = _symmetric_spectral_root(h, Lu)
        d = u.copy()                 # symmetric: no reversal needed
    else:
        # u minimum phase, d = time-reversed u. Accuracy of the two methods
        # varies with M/L; try both, keep the factor whose convolution best
        # preserves the Nyquist zeros.
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
    for M, Lu, Ld, ph in [(2, 31, 31, "min"), (4, 65, 65, "min"),
                          (4, 129, 129, "min"), (8, 129, 129, "min"),
                          (2, 31, 31, "linear"), (4, 65, 65, "linear"),
                          (4, 129, 129, "linear"), (8, 129, 129, "linear")]:
        u, d = design_nyquist_pair(M, Lu, Ld, phase=ph)
        x = rng.standard_normal(2000)
        y = roundtrip(x, M, u, d)
        D = (Lu + Ld - 2) // (2 * M)    # expected roundtrip delay (input samples)
        err = np.max(np.abs(y[D:D + len(x)] - x))
        h = np.convolve(u, d)
        c = (Lu + Ld - 2) // 2
        off = h[np.arange(len(h)) % M == 0]
        off[c // M] = 0                 # exclude the center tap
        sym = np.max(np.abs(u - u[::-1]))
        print(f"M={M}, L={Lu}, {ph:6s}: delay={D}, max|y-x|={err:.3e}, "
              f"max off-center h[kM]={np.max(np.abs(off)):.3e}, "
              f"u symmetry err={sym:.1e}")
