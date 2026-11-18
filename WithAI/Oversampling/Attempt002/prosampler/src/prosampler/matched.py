"""
Matched kernel ``g`` with ``u = d = g`` and an exactly transparent roundtrip (pipeline: "from-scratch matched").

The chain  ``zero-stuff -> g -> (processing) -> g -> keep every L-th``  is an identity iff ``g * g`` is Nyquist(L)
with its centre at ``L * K``.  Unlike the (u, d) pair this is a QUADRATIC constraint in ``g``, so the problem is a
nonconvex NLP: minimise the stopband energy of ``g`` subject to ``(g*g)[n0 + L k] = delta[k]``.

Convention: ``g`` has DC gain ``sqrt(L)`` so that ``g * g`` has gain ``L``.  A matched :class:`FilterPair` therefore
holds ``u = d = g`` with gain ``sqrt(L)`` each (the product gain ``L`` is what matters).

Design facts
------------
* ``g`` is symmetric with ``N = L*K + 1`` taps (``L*K`` even), latency ``K``.  The two end taps are forced to
  zero by the last constraint, so the effective length is ``N - 2``.
* Degrees of freedom: ``(L - 2) K / 2``.  For ``L = 2`` this is ZERO: solutions are isolated points of a square
  system, and the Kaiser-sinc start only finds bad ones.  That does not mean no good solution exists (one at about
  -69 dB was found by continuation from a (u, d) pair) but it is badly conditioned (Jacobian condition ~3e12) and
  could not be projected to an exact ``u = d`` solution better than ~5e-10 / -52 dB.  For ``L = 2`` prefer
  :func:`~prosampler.design_pair`.
* The optimiser (SLSQP + Gauss-Newton projection) finds LOCAL minima, so the result depends on the start (Kaiser-sinc
  with window parameter ``beta``) and is not monotone in ``K``: for ``L=4, passband=0.9`` a cold start gives -82.7 dB
  at K=130 but -74.5 dB at K=140, while ``beta=2`` gives -85.6 dB at K=140.  ``design_matched`` therefore tries several
  starts (``betas``) and keeps the best; warm-starting alone can get trapped (a spec-driven 80 dB search that only
  warm-started stayed near -75 dB up to K=596).  The result is always exactly Nyquist (the projection guarantees
  it); only the stopband may be worse than what is achievable.  The spec-driven search bisects on K and so finds
  a length that meets the spec, not necessarily the shortest (a shorter K can be better than a longer one).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.signal import windows

from ._util import mirror_matrix
from .analysis import band_energy_matrix
from .errors import SpecNotReached
from .kernels import kaiser_beta
from .pair import FilterPair
from .spec import Spec


def optimise_matched(L: int, K: int, passband: float, beta: float = 4.0, g_init=None, n_rescale: int = 10) -> np.ndarray:
    """
    Symmetric ``g`` with ``N = L*K + 1`` taps, DC gain ``sqrt(L)``, ``g*g`` exactly Nyquist(L), minimal stopband energy.

    ``g_init`` is an optional shorter symmetric solution (same ``L`` and passband).  It is zero-padded, which keeps it
    exactly feasible, so the result can only be better; this makes the stopband energy monotone in ``K`` and is what
    makes long designs converge.
    """
    N = L * K + 1
    if N % 2 == 0:
        raise ValueError("L*K must be even (N = L*K+1 must be odd for a symmetric g)")
    idx = (N - 1) + L * np.arange(K)               # the constraint n = K is implied by the zero end taps
    tgt = (np.arange(K) == 0).astype(float)
    S = mirror_matrix(N, drop_ends=True)
    Qt = S.T @ band_energy_matrix(N, (2.0 - passband) * np.pi / L, np.pi) @ S
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

    def polish(th, iters=40):                      # Gauss-Newton projection onto {r = 0}
        for _ in range(iters):
            rr = r(th)
            if np.max(np.abs(rr)) < 1e-16:
                break
            th = th - np.linalg.lstsq(J(th), rr, rcond=None)[0]
        return th

    if g_init is not None:
        g0 = np.pad(np.asarray(g_init, float), (N - len(g_init)) // 2)
        g0 = g0 + 1e-7 * np.sqrt(L) * np.random.default_rng(0).standard_normal(N) * (np.abs(g0) < 1e-12)
        g0 = 0.5 * (g0 + g0[::-1])                 # break the degeneracy of the zero-padded outer taps
    else:
        n = np.arange(N) - mid
        g0 = np.sinc(n / L) * windows.kaiser(N, beta)
        g0 *= np.sqrt(L) / g0.sum()
    th = polish(np.array([g0[mid - k] for k in range(mid)]))

    # The objective can be ~1e-10 or smaller, far below SLSQP's absolute tolerances: rescale to O(1) per restart.
    rng = np.random.default_rng(1)
    stalls = 0
    for _ in range(n_rescale):
        f0 = 0.5 * th @ Qt @ th
        sc = 1.0 / f0
        start = th if stalls == 0 else polish(th + 1e-6 * np.sqrt(f0) * rng.standard_normal(len(th)))
        res = minimize(lambda t: sc * 0.5 * t @ Qt @ t, start, jac=lambda t: sc * (Qt @ t), method="SLSQP",
                       constraints=[dict(type="eq", fun=r, jac=J)], options=dict(maxiter=300, ftol=1e-14))
        th_new = polish(res.x)
        if 0.5 * th_new @ Qt @ th_new < 0.999 * f0:        # keep only real improvements
            th, stalls = th_new, 0
        else:
            stalls += 1                                      # SLSQP is fragile here: retry once perturbed
            if stalls >= 3:
                break
    return S @ th


def _make_pair(g, L, K, passband, meta) -> FilterPair:
    return FilterPair(g, g, L, K, passband, method="matched[from-scratch]", info=meta)


def design_matched(L: int, passband: float, atten_db: Optional[float] = None, *, K: Optional[int] = None,
                   g_init=None, beta: Optional[float] = None, betas: Optional[Sequence[float]] = None, K_max: int = 600, growth: float = 1.08,
                   max_pass_ripple_db: Optional[float] = None, strict: bool = False,
                   verbose: bool = False) -> FilterPair:
    """
    Matched pair ``u = d = g`` designed from scratch.

    * **Length-driven**: give ``K`` (``g`` then has ``L*K + 1`` taps, ``L*K`` must be even).
    * **Spec-driven**: give ``atten_db``; the smallest ``K`` reaching it is searched (growth, then bisection, each
      length warm-started from the previous failing design).

    The problem has many local minima and the result depends on the starting point (Kaiser-sinc with window parameter
    ``beta``; also not monotone in ``K``).  ``betas`` is a list of cold-start betas of which the best result is kept
    (together with the warm start from ``g_init`` / the previous length); default: ``(beta,)``, or ``(2, 4, 6)`` when
    ``atten_db > 60``.  ``beta`` is a shorthand for ``betas=(beta,)``.

    The returned :class:`FilterPair` has ``u`` and ``d`` equal to ``g`` (DC gain ``sqrt(L)`` each) and ``latency = K``.
    ``info`` holds ``K``, ``N``, ``meets_spec``.  Emits a warning for ``L = 2`` (see module docs) and when the spec
    is not reached (``strict=True`` raises :class:`~prosampler.SpecNotReached` instead).
    """
    if L == 2:
        warnings.warn("matched design with L=2 has zero degrees of freedom and gives poor, ill-conditioned "
                      "solutions; prefer design_pair", stacklevel=2)
    spec = Spec(L, passband, atten_db)
    step = 2 if L % 2 else 1                        # keep L*K even
    if betas is None:
        if beta is not None:
            betas = (beta,)
        elif atten_db is not None and atten_db > 60:
            betas = (2.0, 4.0, 6.0)
        else:
            betas = (min(kaiser_beta(atten_db), 6.0) if atten_db is not None else 4.0,)
    ripple_limit = None
    if atten_db is not None:
        ripple_limit = max_pass_ripple_db if max_pass_ripple_db is not None else max(0.01, 4 * 8.686 * 10 ** (-atten_db / 20))

    def measure(g, K):
        fp = _make_pair(g, L, K, passband, dict(K=K, N=len(g), atten_db=atten_db))
        m = fp.metrics(n_grid=6000)
        fp.info.update(stop_db=m["u_stop_db"], pass_ripple_db=m["u_ripple_db"])
        ok = atten_db is not None and m["u_stop_db"] <= -atten_db and m["u_ripple_db"] <= ripple_limit
        fp.info["meets_spec"] = bool(ok) if atten_db is not None else None
        if verbose:
            print(f"  K={K} N={len(g)}: stop={m['u_stop_db']:.1f} dB nyq={m['nyquist_error']:.0e}", flush=True)
        return fp

    def solve(K, g_warm):
        """Best of the cold starts (one per beta) and the warm start; keeps the lowest stopband."""
        cands = [measure(optimise_matched(L, K, passband, b), K) for b in betas]
        if g_warm is not None:
            cands.append(measure(optimise_matched(L, K, passband, betas[0], g_warm), K))
        return min(cands, key=lambda fp: fp.info["stop_db"])

    def fail(fp, msg):
        full = f"{msg} (stop {fp.info['stop_db']:.1f} dB, ripple {fp.info['pass_ripple_db']:.4f} dB)"
        if strict:
            raise SpecNotReached(full)
        warnings.warn(full, stacklevel=3)

    # ------------------------------------------------------------------ length-driven
    if K is not None:
        if (L * K) % 2:
            raise ValueError("L*K must be even")
        fp = solve(K, g_init)
        if fp.info["meets_spec"] is False:
            fail(fp, f"matched design with K={K} does not meet {atten_db} dB")
        return fp

    # ------------------------------------------------------------------ spec-driven
    if atten_db is None:
        raise ValueError("give atten_db (spec-driven) or K (length-driven)")
    dw = spec.transition_width                                     # rad per high-rate sample
    K0 = max(step * 2, int(np.ceil(0.5 * (atten_db - 8) / (2.285 * dw) / L)))
    K0 += K0 % step
    lo, g_lo, K_ = None, g_init, K0
    while True:
        fp = solve(K_, g_lo)
        if fp.info["meets_spec"]:
            break
        lo, g_lo = K_, fp.u
        K_ = max(int(np.ceil(K_ * growth / step)) * step, K_ + step)
        if K_ > K_max:
            fail(fp, f"specification not reachable within K_max={K_max}")
            return fp
    hi, best = K_, fp
    while lo is not None and hi - lo > step:
        mid = (lo + hi) // 2
        mid -= mid % step
        if mid <= lo or mid >= hi:
            break
        fm = solve(mid, g_lo)
        if fm.info["meets_spec"]:
            hi, best = mid, fm
        else:
            lo, g_lo = mid, fm.u
    return best
