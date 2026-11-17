#!/usr/bin/env python3
import numpy as np
from scipy.signal import firwin, upfirdn

def design_pr_pair(L=4, span=12, cutoff_frac=0.90, window="kaiser", beta=8.6):
    """
    Design u[n], d[n] for upsample-by-L / downsample-by-L roundtrip.

    L           : oversampling factor
    span        : half-length in original-rate samples (taps = 2*span*L+1)
    cutoff_frac : fraction of ideal cutoff (ideal is 1/L of high-rate Nyquist)
    """
    # Choose odd tap count with center delay exactly on an L-grid.
    # This makes roundtrip delay an integer number of original-rate samples.
    num_taps = 2 * span * L + 1

    # Anti-imaging lowpass for upsampler (high-rate normalized freq, Nyquist=1.0)
    cutoff = cutoff_frac / L
    u = firwin(num_taps, cutoff=cutoff, window=(window, beta) if window == "kaiser" else window)

    # Initial anti-alias candidate: matched filter
    g = u[::-1].copy()

    # Build exact PR constraints on decimated samples of h = u * d:
    # h[delay + mL] = delta[m]
    Nd = num_taps
    Nu = num_taps
    h_len = Nu + Nd - 1
    delay = (Nu - 1 + Nd - 1) // 2  # high-rate samples

    m_min = int(np.ceil(-delay / L))
    m_max = int(np.floor((h_len - 1 - delay) / L))
    ms = np.arange(m_min, m_max + 1)

    A = np.zeros((len(ms), Nd))
    b = np.zeros(len(ms))

    for r, m in enumerate(ms):
        n = delay + m * L
        # h[n] = sum_j d[j] * u[n-j]
        j_lo = max(0, n - (Nu - 1))
        j_hi = min(Nd - 1, n)
        js = np.arange(j_lo, j_hi + 1)
        A[r, js] = u[n - js]
        if m == 0:
            b[r] = 1.0

    # Project g onto affine set {d | A d = b}
    # d = g + A^T (A A^T)^(-1) (b - A g)
    AAT = A @ A.T
    d = g + A.T @ np.linalg.pinv(AAT, rcond=1e-12) @ (b - A @ g)

    return u, d, delay

def verify_roundtrip(u, d, L=4, N=48000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)

    y_hi = upfirdn(u, x, up=L)      # upsample + anti-imaging
    z_hi = y_hi                      # identity "waveshaper"
    y = upfirdn(d, z_hi, down=L)     # anti-alias + downsample

    # Integer roundtrip delay in original-rate samples
    total_delay_hi = (len(u) - 1 + len(d) - 1) // 2
    total_delay_lo = total_delay_hi // L

    y_aligned = y[total_delay_lo:total_delay_lo + N]
    err = y_aligned - x

    # Also check exact PR constraint residual directly
    # A*d-b should be near machine precision if design succeeded.
    h = np.convolve(u, d)
    dec = h[total_delay_hi::L]
    # center element of dec corresponds to m=0
    # (sequence starts at m_min, so just measure off-peak + peak)
    peak = np.max(np.abs(dec))
    off = np.max(np.abs(np.delete(dec, np.argmax(np.abs(dec))))) if len(dec) > 1 else 0.0

    print(f"L={L}, taps={len(u)}")
    print(f"Roundtrip delay (original-rate samples): {total_delay_lo}")
    print(f"RMS error: {np.sqrt(np.mean(err**2)):.3e}")
    print(f"Max abs error: {np.max(np.abs(err)):.3e}")
    print(f"Composite decimated peak (should be ~1): {peak:.12f}")
    print(f"Composite decimated off-peak (should be ~0): {off:.3e}")

if __name__ == "__main__":
    L = 4
    u, d, delay_hi = design_pr_pair(L=L, span=12, cutoff_frac=0.90, window="kaiser", beta=8.6)
    verify_roundtrip(u, d, L=L)

    # Save kernels if desired:
    np.savetxt("u_kernel.txt", u)
    np.savetxt("d_kernel.txt", d)