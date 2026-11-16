"""
Matched oversampling filter pair  u = d = g  with an EXACTLY transparent roundtrip.

    x -> zero-stuff by L -> g -> [nonlinear processing] -> g -> keep every L-th sample

With the identity in the middle the chain returns y[m] = x[m - K] (up to float roundoff) iff
h = g * g is a Nyquist(L) filter with its centre at n0 = L*K:

        h[n0 + L*k] = delta[k]       for all integers k.

Convention: g has DC gain sqrt(L), so that h has the gain L required by that condition (the
upsampler's zero-stuffing then needs no extra scaling, the roundtrip gain is exactly 1).

Design
------
g is symmetric (linear phase) with N = L*K + 1 taps (N must be odd, i.e. L*K even).  Then
n0 = N - 1 = L*K and the roundtrip latency is K base-rate samples.

Unlike the (u, d) design in oversampling_pair.py, the Nyquist condition is QUADRATIC in g, so it
is not a linear (KKT) equality constraint any more.  We minimise the stopband energy of g

        E(g) = integral_{(2-passband) pi/L}^{pi} |G(w)|^2 dw  =  g^T Q g

subject to the K+1 quadratic equality constraints (g*g)[n0 + L n] = delta[n], n = 0..K
(the constraints for n < 0 are identical by symmetry).  SLSQP finds the optimum; a final
Gauss-Newton projection onto the constraint set makes the Nyquist condition hold to ~1e-17.

The two end taps are exactly zero (forced by the constraint n = K), so the effective length is N-2.
KNOWN LIMITATION: the optimisation is nonconvex and SLSQP is fragile on it (rank-deficient
subproblems, because the outermost constraints involve products of tiny taps).  In the tests below it
works reliably up to about 60 dB stopband attenuation.  Beyond that it can stall: e.g. for L=4,
passband 0.9 the attenuation plateaus at about -74 dB although a one-off run reached -83 dB at
K=130, so the plateau is the solver, not the design.  The result is always exactly Nyquist (the
projection guarantees that); only its stopband may be worse than what is achievable.
Degrees of freedom: (N-1)/2 unknowns - K constraints = (L-2)*K/2.
For L = 2 this is ZERO: the solutions are isolated points of a square nonlinear system, so the
optimisation has nothing to optimise and this module's Kaiser-sinc start only finds bad ones
(about -8 dB for N = 95, p = 0.8).  CORRECTION of an earlier claim: that does NOT mean no good
solution exists.  Continuing from a jointly optimised (u, d) pair to equal lengths and then tying
u = d with a penalty found one at about -69 dB (N = 95, p = 0.8); but that solution is badly
conditioned (Jacobian condition number ~3e12), and projecting it onto an exact u = d solution in
double precision only reached a Nyquist error of ~5e-10 and about -52 dB.  For L = 2 prefer the
(u, d) pair from oversampling_pair.py.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import windows


def stopband_matrix(n_taps, w_stop):
    """Q[i,j] = integral_{w_stop}^{pi} cos((i-j) w) dw, so that  g^T Q g = stopband energy of g."""
    k = np.arange(n_taps)
    diff = k[:, None] - k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (np.sin(diff * np.pi) - np.sin(diff * w_stop)) / diff
    Q[diff == 0] = np.pi - w_stop
    return Q


def _mirror_matrix(N):
    """
    S with g = S @ theta for symmetric g (theta[k] = g[mid-k] = g[mid+k]).  The two end taps
    g[0] = g[N-1] are NOT parametrised: the last Nyquist constraint (g*g)[2N-2] = g[N-1]^2 = 0
    forces them to be exactly zero (and has zero gradient there, which would make the
    optimisation problem degenerate).
    """
    mid = (N - 1) // 2
    S = np.zeros((N, mid))
    for k in range(mid):
        S[mid - k, k] = 1.0
        S[mid + k, k] = 1.0
    return S


def _beta(A):
    if A > 50:
        return 0.1102 * (A - 8.7)
    if A >= 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    return 0.0


def design_matched(L, K, passband, kaiser_beta=4.0, g_init=None, n_rescale=10):
    """
    Symmetric g with N = L*K+1 taps, DC gain sqrt(L), g*g exactly Nyquist(L), minimal stopband energy.
    Returns g.   (L*K must be even; L >= 3 for a useful result.)

    g_init   optional shorter symmetric solution (same L, same passband).  It is zero-padded to
             length N, which keeps it EXACTLY feasible, so the result can only be better.  This
             makes the stopband energy monotone in K and is what makes long designs converge.
    """
    N = L * K + 1
    if N % 2 == 0:
        raise ValueError("L*K must be even (N = L*K+1 must be odd for a symmetric g)")
    idx = (N - 1) + L * np.arange(K)          # n = K is implied by the zero end taps
    tgt = (np.arange(K) == 0).astype(float)
    S = _mirror_matrix(N)
    Qt = S.T @ stopband_matrix(N, (2.0 - passband) * np.pi / L) @ S
    mid = (N - 1) // 2

    def r(th):
        g = S @ th
        return np.convolve(g, g)[idx] - tgt

    def J(th):
        g = S @ th
        Jg = np.zeros((K, N))
        j = np.arange(N)
        for m, p in enumerate(idx):
            i = p - j
            ok = (i >= 0) & (i < N)
            Jg[m, j[ok]] = 2 * g[i[ok]]
        return Jg @ S

    def polish(th, iters=40):                 # Gauss-Newton projection onto {r = 0}
        for _ in range(iters):
            rr = r(th)
            if np.max(np.abs(rr)) < 1e-16:
                break
            th = th - np.linalg.lstsq(J(th), rr, rcond=None)[0]
        return th

    if g_init is not None:
        pad = (N - len(g_init)) // 2
        g0 = np.pad(np.asarray(g_init, float), pad)
    else:
        n = np.arange(N) - mid
        g0 = np.sinc(n / L) * windows.kaiser(N, kaiser_beta)
        g0 *= np.sqrt(L) / g0.sum()
    if g_init is not None:                    # break the degeneracy of the zero-padded outer taps
        g0 = g0 + 1e-7 * np.sqrt(L) * np.random.default_rng(0).standard_normal(N) * (np.abs(g0) < 1e-12)
        g0 = 0.5 * (g0 + g0[::-1])
    th = polish(np.array([g0[mid - k] for k in range(mid)]))
    # The objective can be ~1e-10 or smaller (100 dB), far below SLSQP's absolute tolerances, so
    # it is re-scaled to O(1) before every restart.
    rng = np.random.default_rng(1)
    stalls = 0
    for _ in range(n_rescale):
        f0 = 0.5 * th @ Qt @ th
        sc = 1.0 / f0
        start = th if stalls == 0 else polish(th + 1e-6 * np.sqrt(f0) * rng.standard_normal(len(th)))
        res = minimize(lambda t: sc * 0.5 * t @ Qt @ t, start, jac=lambda t: sc * (Qt @ t), method="SLSQP",
                       constraints=[dict(type="eq", fun=r, jac=J)], options=dict(maxiter=300, ftol=1e-14))
        th_new = polish(res.x)
        if 0.5 * th_new @ Qt @ th_new < 0.999 * f0:      # keep only real improvements
            th, stalls = th_new, 0
        else:
            stalls += 1                                  # SLSQP is fragile here: retry once perturbed
            if stalls >= 3:
                break
    g = S @ th
    return g


def measure(g, L, passband, n_grid=6000):
    """Stopband peak (dB, relative to passband gain), passband ripple (dB), Nyquist error."""
    N = len(g)
    K = (N - 1) // L
    w = np.linspace(0, np.pi, n_grid)
    G = np.abs(np.exp(-1j * np.outer(w, np.arange(N))) @ g) / np.sqrt(L)
    sb, pb = w >= (2 - passband) * np.pi / L, w <= passband * np.pi / L
    h = np.convolve(g, g)
    idx = (N - 1) + L * np.arange(-K, K + 1)
    return dict(stop_db=20 * np.log10(G[sb].max()),
                pass_ripple_db=20 * np.log10(G[pb].max() / G[pb].min()),
                nyquist_error=float(np.max(np.abs(h[idx] - (idx == N - 1)))))


def design_matched_for_spec(L, atten_db, passband, K_max=600, growth=1.08, verbose=False):
    """
    Shortest g (smallest K, to within one bisection step) whose optimised design reaches `atten_db`.
    Each length is warm-started from the next shorter failing design, so the search is cheap.
    """
    step = 2 if L % 2 else 1                  # keep L*K even
    dw = (2.0 - passband) * np.pi / L - passband * np.pi / L
    K = max(step * 2, int(np.ceil(0.5 * (atten_db - 8) / (2.285 * dw) / L)))
    K += K % step
    beta = min(_beta(atten_db), 6.0)

    def attempt(K, g_init):
        g = design_matched(L, K, passband, kaiser_beta=beta, g_init=g_init)
        m = measure(g, L, passband)
        if verbose:
            print(f"  K={K} N={len(g)}: stop={m['stop_db']:.1f} dB nyq={m['nyquist_error']:.0e}", flush=True)
        return g, m

    lo, g_lo = None, None                     # largest failing design so far
    while True:
        g, m = attempt(K, g_lo)
        if m["stop_db"] <= -atten_db:
            break
        lo, g_lo = K, g
        K2 = int(np.ceil(K * growth / step)) * step
        K = max(K2, K + step)
        if K > K_max:
            raise RuntimeError("specification not reachable within K_max")
    hi, best = K, (K, g, m)
    while lo is not None and hi - lo > step:  # bisect between the last failure and the first success
        mid = (lo + hi) // 2
        mid -= mid % step
        if mid <= lo or mid >= hi:
            break
        gm, mm = attempt(mid, g_lo)
        if mm["stop_db"] <= -atten_db:
            hi, best = mid, (mid, gm, mm)
        else:
            lo, g_lo = mid, gm
    K, g, m = best
    return g, dict(K=K, N=len(g), **m)


def roundtrip(x, g, L, K, shaper=lambda s: s):
    """Straightforward (non-polyphase) oversampling chain with u = d = g."""
    up = np.zeros(len(x) * L)
    up[::L] = x
    v = np.convolve(shaper(np.convolve(up, g)), g)
    return v[::L][K:K + len(x)]


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for L, A, p in [(4, 40, 0.8), (4, 60, 0.85), (3, 60, 0.85), (8, 60, 0.85)]:
        t = time.time()
        g, info = design_matched_for_spec(L, A, p)
        x = rng.standard_normal(5000)
        err = np.max(np.abs(roundtrip(x, g, L, info["K"]) - x))
        print(f"L={L} A={A} p={p}: K={info['K']} N={info['N']} stop={info['stop_db']:.1f} dB "
              f"ripple={info['pass_ripple_db']:.4f} dB nyq={info['nyquist_error']:.1e} roundtrip={err:.1e} "
              f"({time.time()-t:.0f}s)", flush=True)
