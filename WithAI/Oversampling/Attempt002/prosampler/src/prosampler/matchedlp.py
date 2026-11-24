"""
Minimax refinement and shortest-length search for a single matched kernel ``g`` (``u = d = g``).

The chain ``zero-stuff -> g -> (processing) -> g -> keep every L-th`` is an identity iff ``g * g`` is Nyquist(L) with its
centre at ``L * K`` (see :mod:`prosampler.matched`).  Here the stopband *peak* of ``g`` is minimised under a passband
ripple bound and a cap on the bump, by sequential linear programming exactly as in :mod:`prosampler.jointlp`, but for
one kernel: the zero-phase amplitude ``A(w) = a(w) @ theta`` is linear in the symmetric kernel parameters ``theta``
(stopband peak, passband ripple, bump cap), only the Nyquist condition ``conv(g, g)[L k + (N-1)] = delta[k]`` is
quadratic.  It is linearised at the current feasible point, a trust region is added, the LP is solved and the step is
projected back onto the exact constraint set by Gauss-Newton.

Gains are relative to the nominal DC gain ``sqrt(L)`` (the exact DC gain of ``g`` is only approximately ``sqrt(L)``).

:func:`search_matched` finds the shortest ``K`` that meets a specification.  Unlike :func:`~prosampler.search_pair` there
is no split to scan, only the length; the best peak is a nearly linear function of ``K``, so a secant search with a bracket
needs only a handful of lengths.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ._util import conv_rows, mirror_matrix
from .errors import ProsamplerError, SpecNotReached
from .jointlp import DEFAULT_RIPPLE_DB, _lp
from .matched import optimise_matched
from .minimax import _amp_matrix
from .pair import FilterPair
from .spec import Spec


def optimise_matched_slp(pair, *, transition_max_db="ripple", iterations: int = 40, trust: float = 0.05,
                         ripple_db=DEFAULT_RIPPLE_DB, tube_weight: float = 1.0, n_grid: int = 3000,
                         min_improve: float = 1e-4, verbose: bool = False):
    """
    Minimax refinement of a matched pair ``u = d = g`` by sequential linear programming.

    Parameters
    ----------
    pair : FilterPair
        A matched pair as returned by :func:`~prosampler.design_matched` (``u = d = g`` with ``N = L K + 1`` taps,
        latency ``K``, ``g * g`` exactly Nyquist).
    transition_max_db : float, "ripple" or None
        Cap on the gain of ``g`` (relative to ``sqrt(L)``) between the passband and stopband edges.  Default ``"ripple"``:
        half the allowed peak-to-peak passband ripple.  ``None``: no cap.  Elastic (a start may violate it).
    ripple_db : float or None
        Allowed peak-to-peak passband ripple of ``g`` in dB (as in ``metrics()``); default 0.1 dB.  ``None``: twice the
        start's ripple (at least 0.001 dB).  Elastic: a start with more ripple is driven down to the bound.
    iterations, trust, tube_weight, n_grid, min_improve, verbose :
        As in :func:`~prosampler.optimise_joint_slp`.

    Returns
    -------
    FilterPair
        A new matched pair (same length and latency, exactly Nyquist); ``info["slp"]`` holds the history
        ``(stop_db, violation, trust)`` per accepted step.
    """
    L = pair.L
    u0, d0 = np.asarray(pair.u, float), np.asarray(pair.d, float)
    if len(u0) != len(d0) or not np.allclose(u0, d0, atol=1e-12):
        raise ProsamplerError("optimise_matched_slp needs a matched pair (u = d); use optimise_joint_slp for pairs")
    g0 = 0.5 * (u0 + u0[::-1])
    N, K = len(g0), pair.latency
    if N != L * K + 1 or N % 2 == 0:
        raise ProsamplerError(f"a matched kernel needs N = L*K + 1 taps (N odd); got N = {N}, L = {L}, K = {K}")
    spec = Spec(L, pair.passband, None)
    sL = np.sqrt(L)
    S = mirror_matrix(N, drop_ends=True)
    m = S.shape[1]
    mid = (N - 1) // 2
    th = g0[mid - np.arange(m)]
    idx = (N - 1) + L * np.arange(K)
    e = (np.arange(K) == 0).astype(float)

    def resid(t):
        g = S @ t
        return np.convolve(g, g)[idx] - e

    def jac(t):
        return 2.0 * conv_rows(S @ t, N, idx) @ S

    def project(t):
        for _ in range(30):
            R = resid(t)
            if np.max(np.abs(R)) < 1e-15:
                break
            t = t - np.linalg.lstsq(jac(t), R, rcond=None)[0]
        return t

    th = project(th)
    grid = spec.grid(n_grid)
    w_pass, w_stop, w_tr = grid.w[grid.passband], grid.w[grid.stopband], grid.w[grid.transition]
    w_fine = np.linspace(spec.stopband_edge, np.pi, 8 * n_grid)
    A_f, A_p, A_t = (_amp_matrix(N, w, S) for w in (w_fine, w_pass, w_tr))

    def peak(t):
        return float(np.max(np.abs(A_f @ t)) / sL)

    def ripple_of(t):
        r = A_p @ t / sL
        return 20 * np.log10(r.max() / r.min())

    if ripple_db is None:
        ripple_db = max(2 * ripple_of(th), 1e-3)
    lo, hi = 10 ** (-ripple_db / 40), 10 ** (ripple_db / 40)
    if isinstance(transition_max_db, str):
        if transition_max_db != "ripple":
            raise ValueError('transition_max_db must be a number, None or "ripple"')
        transition_max_db = ripple_db / 2
    gmax = None if transition_max_db is None else 10 ** (transition_max_db / 20)
    rip_soft = 0.01 if ripple_of(th) > 1.05 * ripple_db else 0.0          # elastic only if the start violates the bound

    def violation(t):
        r = A_p @ t / sL
        v = 100 * max(r.max() - hi, lo - r.min())
        if gmax is not None:
            v = max(v, np.max(A_t @ t / sL) - gmax)
        return float(max(v, 0.0))

    def merit(t):
        return peak(t) + tube_weight * violation(t)

    cur = merit(th)
    hist = []
    rho, stall = trust, 0
    hist.append((20 * np.log10(peak(th)), violation(th), rho))
    extra = np.array([])
    nv = m + 2
    for it in range(iterations):
        sigma = peak(th)
        mag = np.abs(A_f @ th) / sL
        loc = np.nonzero((mag >= np.roll(mag, 1)) & (mag >= np.roll(mag, -1)) & (mag > 0.5 * sigma))[0]
        extra = np.unique(np.concatenate([extra, w_fine[loc[(loc > 0) & (loc < len(mag) - 1)]]]))
        ws = np.unique(np.concatenate([w_stop, extra]))
        A_s = _amp_matrix(N, ws, S)
        rows, rhs = [], []

        def add(block, s_coef, xi_coef, rhs_vec):
            M = np.zeros((len(rhs_vec), nv))
            M[:, :m], M[:, m], M[:, m + 1] = block, s_coef, xi_coef
            rows.append(M)
            rhs.append(rhs_vec)

        for sg in (1.0, -1.0):
            add(sg * A_s, -sL * sigma, 0.0, -sg * (A_s @ th))
        add(A_p, 0.0, -sL * rip_soft, sL * hi - A_p @ th)
        add(-A_p, 0.0, -sL * rip_soft, -sL * lo + A_p @ th)
        if gmax is not None:
            add(A_t, 0.0, -sL, sL * gmax - A_t @ th)
        A_eq = np.zeros((K, nv))
        A_eq[:, :m] = jac(th)
        r_ = rho * np.max(np.abs(th))
        bounds = [(-r_, r_)] * m + [(10 ** (-160 / 20) / sigma, None), (0.0, None)]
        cost = np.zeros(nv)
        cost[m], cost[m + 1] = 1.0, tube_weight / sigma
        res = _lp(cost, np.vstack(rows), np.concatenate(rhs), A_eq, -resid(th), bounds)
        if res.status != 0 or res.x is None:
            rho *= 0.5
            if rho < 1e-6:
                break
            continue
        d_th = res.x[:m]
        new_th = project(th + d_th)
        new = merit(new_th)
        if new < cur * (1 - 1e-9):
            gain = (cur - new) / cur
            touched = np.max(np.abs(d_th)) / max(r_, 1e-300) > 0.95
            th, cur = new_th, new
            hist.append((20 * np.log10(peak(th)), violation(th), rho))
            if touched:
                rho = min(rho * 1.5, 0.3)
            stall = stall + 1 if gain < min_improve else 0
            if verbose:
                print(f"  it {it:2d}: stop {hist[-1][0]:.1f} dB, violation {hist[-1][1]:.4f}, rho {rho:.3f}")
            if stall >= 3:
                break
        else:
            rho *= 0.5
            if rho < 1e-6:
                break
    g = S @ th
    return FilterPair(g, g, L, K, pair.passband, method=f"{pair.method}+matchedslp",
                      info={**pair.info, "slp": dict(history=hist, ripple_db=ripple_db, transition_max_db=transition_max_db)})


def _fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _run_matched(args):
    """One candidate: LS start of length K, then SLP.  Returns ``(fp, worst_db, bump_db, ripple_db, ok)`` or ``None``."""
    L, passband, atten_db, K, beta, r_ref, cap, iterations = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = optimise_matched(L, K, passband, beta)
            start = FilterPair(g, g, L, K, passband, method="matched[from-scratch]", info=dict(K=K, N=len(g)))
            fp = optimise_matched_slp(start, ripple_db=r_ref, transition_max_db=cap, iterations=iterations)
        m = fp.metrics()
    except Exception:
        return None
    worst, bump, rip = m["u_stop_db"], m["u_transition_peak_db"], m["u_ripple_db"]
    ok = (worst <= -atten_db and rip <= r_ref * 1.02 + 1e-4 and (cap is None or bump <= cap + 0.02)
          and m["nyquist_error"] < 1e-12)
    return fp, worst, bump, rip, bool(ok)


def search_matched(L: int, passband: float, atten_db: float, *, transition_max_db="ripple",
                   ripple_db: float | None = DEFAULT_RIPPLE_DB, betas=(4.0,), iterations: int = 30,
                   K_start: int | None = None, K_max: int = 600, max_probes: int = 12, n_jobs: int = 1,
                   progress=True):
    """
    Shortest matched kernel ``g`` (``u = d = g``) meeting ``atten_db`` with the given passband ripple and bump cap.

    For each probed length ``K`` (``g`` has ``N = L K + 1`` taps) a least-squares start from
    :func:`~prosampler.optimise_matched` is refined by :func:`optimise_matched_slp`.  The best stopband peak is a nearly
    linear function of ``K``, so the search probes one length, jumps by secant steps to where the line reaches the
    specification (0.3 dB margin), keeps a bracket (longest failing / shortest passing ``K``) and ends when they are one
    step apart (step 1 for even ``L``, 2 for odd ``L``, so that ``L K`` stays even).

    Parameters
    ----------
    L, passband, atten_db :
        Oversampling factor, passband edge (fraction of the original Nyquist frequency) and required stopband attenuation.
    transition_max_db, ripple_db :
        As in :func:`optimise_matched_slp` (defaults: bump cap ``"ripple"`` = half the ripple, ripple 0.1 dB).  ``ripple_db``
        must be a number here.
    betas : sequence of float
        Kaiser parameters of the cold starts of the least-squares step; the best result of the SLP over these is kept
        (the problem has local minima; ``n_jobs`` runs them in parallel).
    iterations : int
        SLP iterations per candidate.
    K_start : int, optional
        First length to probe (default: the Kaiser estimate of the length).
    K_max, max_probes : int
        Limits on the largest ``K`` and on the number of probed lengths.
    n_jobs : int
        Worker processes (parallel over ``betas`` within a length); ``-1``: all cores.
    progress : bool or callable
        Print a line per candidate to stderr, or call ``progress(dict)`` with ``probe, K, N, beta, worst_db, bump_db,
        ripple_db, meets, elapsed``.

    Returns
    -------
    FilterPair
        The shortest matched pair found that meets the specification; ``info["search_matched"]`` holds the table of
        all candidates and the settings.  Raises :class:`~prosampler.SpecNotReached` if no probed length met it.
        Local optima scatter the result by a few dB, so this is the shortest length *found*.
    """
    if ripple_db is None:
        raise ValueError("search_matched needs an absolute ripple_db")
    if L == 2:
        warnings.warn("matched design with L=2 has zero degrees of freedom; prefer design_pair", stacklevel=2)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    cap = ripple_db / 2 if isinstance(transition_max_db, str) and transition_max_db == "ripple" else transition_max_db
    if isinstance(cap, str):
        raise ValueError('transition_max_db must be a number, None or "ripple"')
    spec = Spec(L, passband, atten_db)
    step = 2 if L % 2 else 1
    dw = spec.transition_width
    slope0 = 2.285 * dw * L                                     # Kaiser: dB gained per unit of K
    K0 = max(2 * step, int(np.ceil(2.2 * (atten_db - 8) / (2.285 * dw) / L))) if K_start is None else int(K_start)      # Kaiser estimate x 2.2
    K0 += K0 % step
    target = -atten_db - 0.3
    pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 and len(betas) > 1 else None
    t0, table, probes = time.time(), [], {}
    best = None

    def probe(K, number):
        nonlocal best
        args = [(L, passband, atten_db, K, b, ripple_db, cap, iterations) for b in betas]
        if pool is None:
            results = ((b, _run_matched(a)) for b, a in zip(betas, args))
        else:
            futs = {pool.submit(_run_matched, a): b for b, a in zip(betas, args)}
            results = ((futs[f], f.result()) for f in as_completed(futs))
        best_w, found = None, None
        for b, res in results:
            if res is None:
                continue
            fp, worst, bump, rip, ok = res
            table.append(dict(K=K, N=len(fp.u), beta=b, worst_db=worst, bump_db=bump, ripple_db=rip, meets=ok))
            info = dict(probe=number, K=K, N=len(fp.u), beta=b, worst_db=worst, bump_db=bump, ripple_db=rip, meets=ok,
                        elapsed=time.time() - t0)
            if callable(progress):
                progress(info)
            elif progress:
                print(f"[probe {number} | {_fmt_time(info['elapsed'])} elapsed] K {K} (N {info['N']}) beta {b}: worst "
                      f"{worst:.1f} dB, bump {bump:+.3f} dB, ripple {rip:.4f} dB, {'meets' if ok else 'misses'}",
                      file=sys.stderr, flush=True)
            if best_w is None or worst < best_w:
                best_w = worst
            if ok and (found is None or worst < found.metrics()["u_stop_db"]):
                found = fp
        if best_w is None:
            return None
        probes[K] = (best_w, found is not None)
        if found is not None and (best is None or K < best.latency):
            best = found
        return probes[K]

    try:
        K, fail_hi, pass_lo = K0, None, None                    # fail_hi: largest failing K, pass_lo: smallest passing K
        for number in range(1, max_probes + 1):
            r = probe(K, number)
            if r is None or not r[1]:
                fail_hi = K if fail_hi is None else max(fail_hi, K)
            else:
                pass_lo = K if pass_lo is None else min(pass_lo, K)
            if fail_hi is not None and pass_lo is not None and pass_lo - fail_hi <= step:
                break
            pts = sorted((k, v[0]) for k, v in probes.items())
            if fail_hi is not None and pass_lo is not None and fail_hi in probes and pass_lo in probes:
                (k1, g1), (k2, g2) = (fail_hi, probes[fail_hi][0]), (pass_lo, probes[pass_lo][0])
            elif len(pts) >= 2:                                    # the two probes closest to the target
                (k1, g1), (k2, g2) = sorted(sorted(pts, key=lambda t: abs(t[1] - target))[:2])
            else:
                k1, g1, k2, g2 = pts[0][0], pts[0][1], pts[0][0] + 1, pts[0][1] - slope0
            if k2 != k1 and g2 < g1:                              # dB per K (negative slope)
                nk = k1 + (target - g1) * (k2 - k1) / (g2 - g1)
            else:
                nk = pts[0][0] + (target - pts[0][1]) / (-slope0)
            nk = int(round(nk / step)) * step
            lo_lim = (fail_hi + step) if fail_hi is not None else step * 2
            hi_lim = (pass_lo - step) if pass_lo is not None else K_max
            if pass_lo is None and fail_hi is not None:
                nk = max(nk, fail_hi + step)
            if fail_hi is None and pass_lo is not None:
                nk = min(nk, pass_lo - step)
            nk = max(lo_lim, min(hi_lim, nk))
            if hi_lim < lo_lim or nk in probes or nk > K_max:
                break
            K = nk
    finally:
        if pool is not None:
            pool.shutdown()
    if best is None:
        raise SpecNotReached(f"no matched kernel up to K = {max(probes) if probes else K0} met {atten_db} dB "
                             f"(ripple {ripple_db} dB, cap {cap} dB)")
    best.info["search_matched"] = dict(table=table, ripple_db=ripple_db, transition_max_db=cap, betas=tuple(betas),
                                       iterations=iterations, K_start=K0, elapsed_s=time.time() - t0)
    return best
