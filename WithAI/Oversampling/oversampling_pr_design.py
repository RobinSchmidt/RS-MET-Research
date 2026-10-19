import numpy as np

# ---------------- adjustable parameters ----------------
L        = 4          # oversampling factor
N_U      = 8 * L + 1  # length of upsampling kernel u (odd -> linear phase)
N_D      = 16 * L + 1 # length of downsampling kernel d (odd); keep N_D > ~2*N_U/... see check below
KAISER_BETA = 8.0     # window shape for u (higher -> more stopband attenuation)
TRANSITION  = 0.2     # relative transition width around pi/L (0..1) for the design of d
# --------------------------------------------------------


def design_u(L, n_taps, beta):
    """Kaiser-windowed sinc lowpass, cutoff pi/L, DC gain L (compensates zero-stuffing)."""
    n = np.arange(n_taps) - (n_taps - 1) / 2
    u = np.sinc(n / L) * np.kaiser(n_taps, beta)
    return u * (L / np.sum(u))


def stopband_matrix(n_taps, w_stop):
    """Q[i,j] = integral_{w_stop}^{pi} cos((i-j) w) dw  (stopband energy of a symmetric-in-time FIR)."""
    k = np.arange(n_taps)
    diff = k[:, None] - k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (np.sin(diff * np.pi) - np.sin(diff * w_stop)) / diff
    Q[diff == 0] = np.pi - w_stop
    return Q


def design_d(u, L, n_taps, transition):
    """
    Minimize stopband energy of d subject to the exact Nyquist(L) condition on h = u*d:
        h[c + L*n] = delta[n]
    Solved via the KKT system of an equality-constrained quadratic program.
    """
    n_u = len(u)
    n_h = n_u + n_taps - 1
    c = (n_h - 1) // 2                      # center of h (n_h is odd for odd lengths)

    # Constraint rows: positions c + L*n inside h
    idx = np.arange(c % L, n_h, L)
    rhs = (idx == c).astype(float)
    if len(idx) >= n_taps:
        raise ValueError(f"Too many constraints ({len(idx)}) for N_D={n_taps}; increase N_D or reduce N_U.")

    # C[m, j] = u[idx[m] - j]  (convolution matrix rows)
    C = np.zeros((len(idx), n_taps))
    for m, p in enumerate(idx):
        j = np.arange(n_taps)
        i = p - j
        ok = (i >= 0) & (i < n_u)
        C[m, j[ok]] = u[i[ok]]

    # Stopband of d: images of the original band start at 2*pi/L - w_pass
    w_pass = (np.pi / L) * (1 - transition)
    w_stop = 2 * np.pi / L - w_pass
    Q = stopband_matrix(n_taps, w_stop) + 1e-12 * np.eye(n_taps)

    # KKT: [Q C^T; C 0] [d; lambda] = [0; rhs]
    m = len(idx)
    K = np.block([[Q, C.T], [C, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.concatenate([np.zeros(n_taps), rhs]))
    return sol[:n_taps]


def roundtrip(x, u, d, L):
    """upsample -> (identity waveshaper) -> downsample; compensates the total delay."""
    up = np.zeros(len(x) * L)
    up[::L] = x
    up = np.convolve(up, u)
    # <-- waveshaper would go here -->
    y = np.convolve(up, d)
    delay = (len(u) - 1) // 2 + (len(d) - 1) // 2
    return y[delay::L][: len(x)]


if __name__ == "__main__":
    u = design_u(L, N_U, KAISER_BETA)
    d = design_d(u, L, N_D, TRANSITION)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    y = roundtrip(x, u, d, L)
    print(f"L={L}, len(u)={len(u)}, len(d)={len(d)}")
    print("max roundtrip error:", np.max(np.abs(y - x)))

    # Frequency responses (stopband attenuation, DC gain)
    def mag_db(h, w):
        return 20 * np.log10(np.abs(np.exp(-1j * np.outer(w, np.arange(len(h)))) @ h) + 1e-300)

    w = np.linspace(0, np.pi, 4096)
    w_stop = 2 * np.pi / L - (np.pi / L) * (1 - TRANSITION)
    sb = w >= w_stop
    print(f"u: DC gain {np.sum(u):.4f} (should be {L}), worst stopband {mag_db(u, w)[sb].max():.1f} dB")
    print(f"d: DC gain {np.sum(d):.4f}, worst stopband {mag_db(d, w)[sb].max():.1f} dB")