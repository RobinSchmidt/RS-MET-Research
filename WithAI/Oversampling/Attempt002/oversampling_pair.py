"""
FIR up/down-sampling filter pair (u, d) for oversampling with an EXACTLY transparent roundtrip.

    x -> zero-stuff by L -> u -> [nonlinear processing] -> d -> keep every L-th sample

If the processing is the identity, the chain returns  y[m] = x[m - K]  (exact up to float
roundoff) iff h = u * d is a Nyquist(L) filter with its centre at n0 = L*K:

        h[n0 + L*k] = delta[k]       for all integers k.

Gain convention: u has passband gain L, d has passband gain 1.

How the pair is built
---------------------
1. u is chosen first (a good anti-imaging lowpass):
     mode 'linear'     u = Kaiser-windowed sinc               -> exactly linear phase
     mode 'lowlatency' u = minimum-phase "root" of a sinc     -> small group delay (see note below)
2. d is then found by an equality-constrained least-squares problem.  For FIXED u the
   Nyquist condition is LINEAR in d, so it is imposed exactly as a hard constraint (null-space
   method).  The remaining freedom is used to minimise
        passband error  | U(w) D(w) - L e^{-j w n0} |^2   +   stopband energy |D(w)|^2 .
3. The length of d (linear mode) or the latency K (lowlatency mode) is searched so that BOTH
   filters meet the requested stopband attenuation.

IMPORTANT finding about 'lowlatency' (read before using it)
-----------------------------------------------------------
u alone does get a small group delay (minimum phase), but the ROUNDTRIP latency K is set by the
Nyquist condition together with a bounded-gain d, and at full specification it is NOT smaller than
in linear mode (it is within about +-15 % of it in the tests below).  If d is allowed unbounded gain
between passband and stopband edge, much smaller K "meets" the stopband spec, but d then has
gains of 100 dB or more there and is useless; that is why max_transition_gain_db is enforced.
If you pass `latency=K` explicitly you get the best attenuation reachable at that latency (it
degrades quickly: e.g. L=4, p=0.9: ~-75 dB at K=100, ~-44 dB at K=80, ~-20 dB at K=45).
The real levers on latency are a wider transition band (smaller `passband`) and lower `atten_db`.

Note: d comes out about twice as long as u.  That is structural: the Nyquist constraints use up
about (len(u)+len(d))/L of d's degrees of freedom, so for a short d (e.g. L = 2) there is nothing
left to shape its stopband.

Parameters
----------
L           oversampling factor (integer >= 2)
atten_db    required stopband attenuation of u and d (dB)
passband    passband edge as a fraction of the ORIGINAL Nyquist frequency, 0 < passband < 1
            (0.9 -> 19.8 kHz at fs = 44.1 kHz).  The stopband starts at (2 - passband), the
            mirror image around the original Nyquist frequency (forced by the Nyquist condition).
mode        'linear' | 'lowlatency'
"""
import numpy as np
from scipy.linalg import lstsq, toeplitz
from scipy.signal import minimum_phase, windows


# ------------------------------------------------------------------ helpers
def _kaiser_beta(A):
    if A > 50:
        return 0.1102 * (A - 8.7)
    if A >= 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    return 0.0


def _freqz(h, w):
    return np.exp(-1j * np.outer(w, np.arange(len(h)))) @ h


def _grid(L, passband, n_grid):
    """Frequency grid on [0, pi] (rad/sample at the HIGH rate) with passband / stopband masks."""
    w = np.linspace(0.0, np.pi, n_grid)
    return w, w <= passband * np.pi / L, w >= (2.0 - passband) * np.pi / L


def _kaiser_sinc(L, A, N):
    """Kaiser-windowed sinc with cutoff pi/L. Nyquist(L) by construction."""
    n = np.arange(N) - (N - 1) / 2
    return np.sinc(n / L) * windows.kaiser(N, _kaiser_beta(A))


def _stop_db(h, w, stp, gain):
    return 20 * np.log10(np.max(np.abs(_freqz(h, w)[stp])) / gain)


def _group_delay_mean(h, w, pas):
    n = np.arange(len(h))
    H, dH = _freqz(h, w[pas]), _freqz(h * n, w[pas])
    return float(np.mean(np.real(dH / H)))


def _design_u(L, A, passband, mode, w, stp):
    """Smallest u of the requested type whose stopband meets `A`."""
    dw = (2.0 - passband) * np.pi / L - passband * np.pi / L
    n_est = int(np.ceil((A - 8) / (2.285 * dw))) | 1
    scale = 0.8
    for _ in range(400):
        if mode == "linear":
            N = int(n_est * scale) | 1
            u = _kaiser_sinc(L, A, N)
        else:
            # linear-phase sinc with 2A dB, then its minimum-phase square root (A dB, half length)
            NH = int(2 * n_est * scale) | 1
            u = minimum_phase(_kaiser_sinc(L, 2 * A, NH), method="homomorphic", n_fft=1 << 16)
        u = u * (L / u.sum())
        if _stop_db(u, w, stp, L) <= -A:
            return u
        scale *= 1.01
    raise RuntimeError("could not design u for the requested specification")


def _solve_d(u, Nd, L, n0, w, pas, stp, stop_weight, trn=None, trn_weight=0.0):
    """
    Length-Nd filter d minimising  passband error + stopband energy
    subject to the EXACT Nyquist condition (u*d)[n0 + L k] = delta[k].
    """
    Nu = len(u)
    n_h = Nu + Nd - 1
    idx = np.arange(n0 % L, n_h, L)
    col = np.concatenate([u, np.zeros(Nd - 1)])
    row = np.zeros(Nd)
    row[0] = u[0]
    C = toeplitz(col, row)[idx, :]
    e = (idx == n0).astype(float)
    # SVD-based particular solution + null space, with iterative refinement so that the
    # Nyquist constraints hold to machine precision even when C is ill-conditioned
    Uc, sv, Vt = np.linalg.svd(C, full_matrices=True)
    r = int(np.sum(sv > sv[0] * 1e-14))
    Vr, Ur = Vt[:r].T, Uc[:, :r]
    d0 = Vr @ ((Ur.T @ e) / sv[:r])
    for _ in range(4):
        d0 = d0 - Vr @ ((Ur.T @ (C @ d0 - e)) / sv[:r])
    Z = Vt[r:].T
    U = _freqz(u, w)
    F = np.exp(-1j * np.outer(w, np.arange(Nd)))
    tgt = L * np.exp(-1j * w[pas] * n0) / U[pas]
    Fp, Fs = F[pas], F[stp]
    blocks = [Fp.real, Fp.imag, stop_weight * Fs.real, stop_weight * Fs.imag]
    n_zero = 2 * Fs.shape[0]
    if trn is not None and trn_weight > 0 and trn.any():
        Ft = F[trn]                       # keeps the don't-care band from blowing up
        blocks += [trn_weight * Ft.real, trn_weight * Ft.imag]
        n_zero += 2 * Ft.shape[0]
    M = np.vstack(blocks)
    rhs = np.concatenate([tgt.real, tgt.imag, np.zeros(n_zero)])
    y = lstsq(M @ Z, rhs - M @ d0)[0]
    return d0 + Z @ y


def _evaluate(u, d, L, n0, w, pas, stp, trn):
    U, D = _freqz(u, w) / L, _freqz(d, w)
    H = U * D * np.exp(1j * w * n0)

    def rip(X):
        return 20 * np.log10(np.max(np.abs(X[pas])) / np.min(np.abs(X[pas])))

    return dict(
        u_stop_db=20 * np.log10(np.max(np.abs(U[stp]))),
        d_stop_db=20 * np.log10(np.max(np.abs(D[stp]))),
        u_pass_ripple_db=rip(U),
        d_pass_ripple_db=rip(D),
        roundtrip_pass_err_db=20 * np.log10(np.max(np.abs(H[pas] - 1)) + 1e-300),
        d_transition_peak_db=20 * np.log10(np.max(np.abs(D[trn]))),
    )


# ------------------------------------------------------------------ main API
def design_oversampling_pair(L, atten_db, passband, mode="linear", latency=None,
                             d_length_factor=2.0, stop_weight=1.0,
                             max_pass_ripple_db=None, max_transition_gain_db=3.0,
                             transition_weight=None, n_grid=3000, verbose=False):
    """
    Returns (u, d, info).

    latency          (mode='lowlatency') roundtrip latency K in base-rate samples.
                     None = smallest K that meets the specification.
    d_length_factor  (mode='lowlatency') len(d) = factor * len(u).  Raised automatically if the
                     specification cannot be met.
    max_pass_ripple_db  accepted passband ripple of u and d (dB); None = max(0.01 dB, 4x the
                     stopband level, i.e. 4 * 8.686 * 10**(-atten_db/20)).
    max_transition_gain_db  accepted peak gain of d between passband and stopband edge (dB).
    transition_weight  weight of the transition-band energy of d in the least-squares objective.
                     None = 0 for 'linear', 0.3 for 'lowlatency' (there it is essential: without it d
                     can have enormous gain between passband and stopband edge).
    info             dict: K (roundtrip latency in base-rate samples, y[m] = x[m-K]), lengths,
                     achieved stopband attenuation / passband ripple, Nyquist error, ...
    """
    assert mode in ("linear", "lowlatency") and L >= 2 and 0 < passband < 1
    if transition_weight is None:
        transition_weight = 0.0 if mode == "linear" else 0.3
    if max_pass_ripple_db is None:
        max_pass_ripple_db = max(0.01, 4 * 8.686 * 10 ** (-atten_db / 20))
    w, pas, stp = _grid(L, passband, n_grid)
    trn = ~(pas | stp)
    u = _design_u(L, atten_db, passband, mode, w, stp)
    Nu = len(u)

    def attempt(K, Nd):
        n0 = L * K
        d = _solve_d(u, Nd, L, n0, w, pas, stp, stop_weight, trn, transition_weight)
        m = _evaluate(u, d, L, n0, w, pas, stp, trn)
        good = (m["d_stop_db"] <= -atten_db and m["d_transition_peak_db"] <= max_transition_gain_db and m["d_pass_ripple_db"] <= max_pass_ripple_db
                and m["u_pass_ripple_db"] <= max_pass_ripple_db)
        if verbose:
            print(f"  K={K:4d} Nu={Nu} Nd={Nd}: d_stop={m['d_stop_db']:7.1f} dB  ok={good}")
        return d, m, good

    def search_K(Nd_of_K, K_lo, K_hi):
        """smallest K in (K_lo, K_hi] that works (monotone assumption, bisection)"""
        d, m, good = attempt(K_hi, Nd_of_K(K_hi))
        if not good:
            return None
        best = (K_hi, d, m)
        while K_hi - K_lo > 1:
            mid = (K_lo + K_hi) // 2
            d, m, good = attempt(mid, Nd_of_K(mid))
            if good:
                K_hi, best = mid, (mid, d, m)
            else:
                K_lo = mid
        return best

    if mode == "linear":
        # n0 = L*K is the exact centre of u*d  ->  Nd = 2*L*K + 2 - Nu  (both filters symmetric)
        Nd_of_K = lambda K: 2 * L * K + 2 - Nu
        K_lo = int(np.ceil((Nu - 1) / L))
        K_hi = int(np.ceil((Nu * 2.6) / L))
        for _ in range(4):
            best = search_K(Nd_of_K, K_lo, K_hi)
            if best:
                break
            K_lo, K_hi = K_hi, int(K_hi * 1.5)
        else:
            raise RuntimeError("specification not reachable (try a smaller passband or attenuation)")
    else:
        factor = d_length_factor
        while True:
            Nd = int(np.ceil(factor * Nu))
            if latency is not None:
                d, m, good = attempt(int(latency), Nd)
                best = (int(latency), d, m)
                break
            K_hi = max(2, int(0.35 * (Nu + Nd - 1) / L))   # centre must stay well inside u*d
            best = search_K(lambda K: Nd, 1, K_hi)
            if best:
                break
            factor *= 1.25
            if factor > 6:
                raise RuntimeError("specification not reachable in lowlatency mode")
    K, d, m = best

    # exactness of the Nyquist condition
    h = np.convolve(u, d)
    sel = h[(L * K) % L::L]
    nyq_err = float(np.max(np.abs(sel - (np.arange(len(sel)) == (L * K) // L))))
    info = dict(mode=mode, L=L, K=K, N_u=len(u), N_d=len(d), nyquist_error=nyq_err, **m)
    info["u_group_delay_passband"] = _group_delay_mean(u, w, pas)       # high-rate samples
    info["d_group_delay_passband"] = _group_delay_mean(d, w, pas)       # high-rate samples
    return u, d, info


# ------------------------------------------------------------------ reference implementation
def roundtrip(x, u, d, L, K, shaper=lambda s: s):
    """Straightforward (non-polyphase) oversampling chain. Returns y aligned so that y ~= x."""
    up = np.zeros(len(x) * L)
    up[::L] = x
    v = np.convolve(shaper(np.convolve(up, u)), d)
    return v[::L][K:K + len(x)]


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for L, A, p in [(2, 90, 0.9), (4, 100, 0.9), (8, 120, 0.85)]:
        for mode in ("linear", "lowlatency"):
            t = time.time()
            u, d, info = design_oversampling_pair(L, A, p, mode)
            K = info["K"]
            x = rng.standard_normal(6000)
            y = roundtrip(x, u, d, L, K)
            err = np.max(np.abs(y - x))
            print(f"L={L} A={A} p={p} {mode:10s} K={K:3d} len(u)={info['N_u']:4d} len(d)={info['N_d']:4d} "
                  f"stop u/d={info['u_stop_db']:.1f}/{info['d_stop_db']:.1f} dB "
                  f"ripple u/d={info['u_pass_ripple_db']:.4f}/{info['d_pass_ripple_db']:.4f} dB "
                  f"nyq={info['nyquist_error']:.1e} roundtrip={err:.1e}  ({time.time()-t:.1f}s)")
