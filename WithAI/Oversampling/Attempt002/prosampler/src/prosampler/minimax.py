"""
Minimax (Chebyshev) design of the partner kernel ``d`` for a given ``u``, and an alternating joint version.

For symmetric, centred kernels the zero-phase amplitude is linear in the kernel parameters ``theta``,

    A(w) = sum_n x[n] cos(w (n - mid)) = a(w)^T theta ,

and so are the exact Nyquist(L) conditions ``C theta = e`` (for a FIXED partner kernel).  Therefore

    minimise  t   subject to   C theta = e,   |A(w)| <= G t  on the stopband

(``G`` is the nominal DC gain: 1 for ``d``, ``L`` for ``u``) is a linear program: convex, one global optimum, no local
minima.  It minimises the worst-case stopband magnitude (the attenuation figure that is quoted) instead of the
stopband energy, and the optimum is equiripple.  The LP is symmetric in the two kernels: it designs ``d`` given ``u``
and just as well ``u`` given ``d``.

Optional extra (still linear) constraints:

* ``transition_max_db``: ``|A(w)| <= G 10**(x/20)`` between the passband and stopband edges (bounds the bump of the
  kernel at the passband edge),
* ``ripple_db``: ``A(w) / G`` stays within ``+-x`` dB of 1 in the passband (each filter's own flatness),
* ``passband_flat_db`` (partner design only): the PRODUCT ``A_d(w) A_u(w) / L`` stays within ``+-x`` dB of 1.

The equalities are eliminated through the nullspace of ``C`` (``theta = theta0 + Z y``), so they hold exactly, and
only inequalities are left for the LP solver (HiGHS).  The frequency grid is refined Remez-style: after each solve
the true stopband peaks between grid points are added and the LP is solved again.

:func:`optimise_joint_minimax` alternates ``d``-steps and ``u``-steps.  Each step keeps the current kernels feasible,
so both stopband peaks are non-increasing; the joint problem itself is bilinear, so the limit is a local optimum.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ._util import conv_rows, is_symmetric, mirror_matrix
from .errors import ProsamplerError
from .nyquist import nyquist_error
from .spec import Spec


def _amp_matrix(n: int, w: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Rows ``a(w)`` with ``A(w) = a(w) @ theta`` for a symmetric odd-length kernel ``g = T @ theta``."""
    return np.cos(np.outer(w, np.arange(n) - (n - 1) / 2)) @ T


def _nyquist_nullspace(other, n_x: int, L: int, n0: int):
    """Symmetric parametrisation of the kernel ``x`` coupled to ``other``: ``T, C, e``, particular ``theta0``, basis ``Z``."""
    n_h = len(other) + n_x - 1
    T = mirror_matrix(n_x)
    idx = np.arange(n0 % L, n_h, L)
    idx = idx[idx <= n0]                              # rows beyond the centre duplicate those before it
    C = conv_rows(other, n_x, idx) @ T
    e = (idx == n0).astype(float)
    U_, sv, Vt = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(sv > sv[0] * 1e-14)) if sv.size and sv[0] > 0 else 0
    Vr, Ur = Vt[:rank].T, U_[:, :rank]
    th0 = Vr @ ((Ur.T @ e) / sv[:rank])
    for _ in range(4):
        th0 = th0 - Vr @ ((Ur.T @ (C @ th0 - e)) / sv[:rank])
    return T, C, e, th0, Vt[rank:].T


def _solve_kernel(other, n_x: int, spec: Spec, n0: int, gain: float, *, transition_max_db=None, ripple_db=None,
                  product_flat_db=None, n_grid=3000, max_exchange=8, tol=1e-3, floor_db=-160.0,
                  info=None) -> np.ndarray:
    """Minimax-optimal symmetric kernel ``x`` (``n_x`` taps) such that ``other * x`` is exactly Nyquist(L)."""
    L = spec.L
    T, C, e, th0, Z = _nyquist_nullspace(other, n_x, L, n0)
    if Z.shape[1] == 0:
        raise ProsamplerError("no degrees of freedom left after the Nyquist constraints (try a longer kernel)")
    nz = Z.shape[1]
    grid = spec.grid(n_grid)
    w_stop = grid.w[grid.stopband]
    sigma = gain * (10 ** (-spec.atten_db / 20) if spec.atten_db else 1e-4)   # scale of the LP variable t = s * sigma

    pass_rows = []                                   # (A_matrix, lo, hi) with lo <= A @ theta <= hi
    if ripple_db is not None:
        pass_rows.append((_amp_matrix(n_x, grid.w[grid.passband], T),
                          gain * 10 ** (-ripple_db / 20), gain * 10 ** (ripple_db / 20)))
    if product_flat_db is not None:
        wp = grid.w[grid.passband]
        a_other = _amp_matrix(len(other), wp, np.eye(len(other))) @ other
        pass_rows.append((_amp_matrix(n_x, wp, T) * (a_other / L)[:, None],
                          10 ** (-product_flat_db / 20), 10 ** (product_flat_db / 20)))
    A_trans = A_trans_fine = None
    if transition_max_db is not None and grid.transition.any():
        A_trans = _amp_matrix(n_x, grid.w[grid.transition], T)
        gmax = gain * 10 ** (transition_max_db / 20)
        wt = np.linspace(spec.passband_edge, spec.stopband_edge, 2 * n_grid)
        A_trans_fine = _amp_matrix(n_x, wt, T)

    w_fine = np.linspace(spec.stopband_edge, np.pi, 8 * n_grid)
    A_fine = _amp_matrix(n_x, w_fine, T)
    cost = np.zeros(nz + 1)
    cost[-1] = 1.0
    th, s_val, it = th0, np.nan, 0
    for it in range(1, max_exchange + 1):
        As = _amp_matrix(n_x, w_stop, T)
        rows, rhs = [], []
        for sg in (1.0, -1.0):                                              # sg * A <= sigma * s
            rows.append(np.hstack([sg * As @ Z, -sigma * np.ones((len(w_stop), 1))]))
            rhs.append(-sg * As @ th0)
        if A_trans is not None:                                             # |A| <= gmax
            for sg in (1.0, -1.0):
                rows.append(np.hstack([sg * A_trans @ Z, np.zeros((len(A_trans), 1))]))
                rhs.append(gmax - sg * A_trans @ th0)
        for Ap, lo, hi in pass_rows:                                        # lo <= A <= hi
            rows.append(np.hstack([Ap @ Z, np.zeros((len(Ap), 1))]))
            rhs.append(hi - Ap @ th0)
            rows.append(np.hstack([-Ap @ Z, np.zeros((len(Ap), 1))]))
            rhs.append(-lo + Ap @ th0)
        bounds = [(None, None)] * nz + [(10 ** (floor_db / 20) * gain / sigma, None)]
        A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)
        for method in ("highs", "highs-ipm", "highs-ds"):                   # HiGHS occasionally gives up: try the others
            res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method,
                          options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
            if res.status == 0 and res.x is not None:
                break
        else:
            raise ProsamplerError(f"minimax LP failed: {res.message}")
        th = th0 + Z @ res.x[:nz]
        s_val = res.x[-1]
        if not 0.2 < s_val < 5:                       # keep t = sigma * s of order sigma: the solver tolerance is absolute
            sigma, s_val = sigma * s_val, 1.0
        # exchange: add the true local maxima of |A| on the stopband that exceed the current level
        mag = np.abs(A_fine @ th)
        loc = np.nonzero((mag >= np.roll(mag, 1)) & (mag >= np.roll(mag, -1)))[0]
        loc = loc[(loc > 0) & (loc < len(mag) - 1)]
        bad = loc[mag[loc] > sigma * s_val * (1 + tol)]
        if len(bad) == 0:
            break
        w_stop = np.unique(np.concatenate([w_stop, w_fine[bad]]))

    # a bound that cannot be met can make the LP solver return a numerically meaningless point: verify on a fine grid
    if sigma * s_val > 1e-8 * gain and np.max(np.abs(A_fine @ th)) > sigma * s_val * (1 + 10 * tol):
        raise ProsamplerError("minimax design did not converge (stopband peak above the LP level)")
    if A_trans is not None and np.max(np.abs(A_trans_fine @ th)) > gmax * (1 + 1e-3) + 1e-9:
        raise ProsamplerError(f"transition_max_db = {transition_max_db} cannot be met for this design "
                              f"(try a larger bound)")
    for _ in range(3):                                                      # exact Nyquist to machine precision
        res_c = C @ th - e
        if np.max(np.abs(res_c)) < 1e-16:
            break
        th = th - np.linalg.lstsq(C, res_c, rcond=None)[0]
    x = T @ th
    if info is not None:
        info.update(stop_db=float(20 * np.log10(sigma * s_val / gain + 1e-300)), iterations=it)
    return x


def _check_centred(u, n_d, L, latency):
    from .partner import resolve_latency                                    # (avoid a circular import)
    latency = resolve_latency(u, n_d, L, latency)
    n0 = L * latency
    if not (is_symmetric(u) and len(u) % 2 == 1 and n_d % 2 == 1 and 2 * n0 == len(u) + n_d - 2):
        raise ProsamplerError("the minimax design needs symmetric odd-length kernels and the centred latency")
    return latency, n0


def design_partner_minimax(u, spec: Spec, n_taps: int, latency=None, *, transition_max_db=None,
                           passband_flat_db=None, n_grid: int = 3000, max_exchange: int = 8,
                           tol: float = 1e-3, floor_db: float = -160.0, info: dict | None = None) -> np.ndarray:
    """
    Partner ``d`` (``n_taps`` taps) of the symmetric, centred kernel ``u`` with minimal worst-case stopband magnitude.

    Same contract as :func:`prosampler.design_partner` (exact Nyquist(L) with centre ``L * latency``), but the
    objective is the stopband PEAK.  ``u`` must be symmetric with odd length, ``n_taps`` odd and the latency centred.
    ``floor_db`` keeps the LP level from going below this value (a longer ``d`` cannot be exploited numerically below
    about -160 dB, and a floor keeps the LP well-posed).
    ``info`` (a dict) receives ``stop_db`` (peak relative to DC gain 1 on the final exchange grid) and ``iterations``.
    """
    u = np.asarray(u, dtype=float)
    L, n_d = spec.L, int(n_taps)
    latency, n0 = _check_centred(u, n_d, L, latency)
    if transition_max_db is not None:
        edge_gain = L / abs(_amp_matrix(len(u), np.array([spec.passband_edge]), np.eye(len(u))) @ u)[0]
        if 10 ** (transition_max_db / 20) < edge_gain * (1 + 1e-4):
            raise ProsamplerError(
                f"transition_max_db = {transition_max_db} is infeasible: d must already have gain "
                f"{20 * np.log10(edge_gain):.4f} dB at the passband edge to compensate u (use a larger bound)")
    d = _solve_kernel(u, n_d, spec, n0, 1.0, transition_max_db=transition_max_db,
                      product_flat_db=passband_flat_db, n_grid=n_grid, max_exchange=max_exchange, tol=tol,
                      floor_db=floor_db, info=info)
    err = nyquist_error(u, d, L, latency)
    if not err < 1e-8:
        raise ProsamplerError(f"minimax partner is not exact (Nyquist error {err:.1e})")
    return d


def optimise_joint_minimax(pair, *, iterations: int = 8, ripple_db=None, transition_max_db=None,
                           min_gain_db: float = 0.05, n_grid: int = 3000, order: str = "du"):
    """
    Alternating-LP joint minimax refinement of a symmetric centred pair ``(u, d)`` (pipeline step 3, minimax version).

    Each round solves the ``d``-LP for fixed ``u`` and then the ``u``-LP for fixed ``d``.  Both stopband peaks are
    non-increasing; the lengths and the latency are unchanged and the Nyquist condition stays exact.

    Parameters
    ----------
    ripple_db : float, optional
        Allowed passband deviation of each filter from its DC gain (dB), applied to ``u`` and ``d``.  Default:
        twice the ripple of the input pair, but at least 0.001 dB.  Looser values let ``u`` and ``d`` trade passband
        flatness for stopband attenuation (the ripple then partly cancels in ``u * d``); this is rarely wanted.
    transition_max_db : float, optional
        Bound on the gain of each filter between the passband and stopband edges (relative to its DC gain).
    min_gain_db : float
        Stop when a full round improves the worst stopband peak by less than this.
    order : {"du", "ud"}
        Which kernel is optimised first.

    Returns a new :class:`~prosampler.FilterPair`; ``info["joint_minimax"]`` has the history of the two peaks (dB).
    """
    from .pair import FilterPair
    if order not in ("du", "ud"):
        raise ValueError("order must be 'du' or 'ud'")
    L = pair.L
    spec = Spec(L, pair.passband, None)
    u, d = np.array(pair.u, float), np.array(pair.d, float)
    u = 0.5 * (u + u[::-1])
    d = 0.5 * (d + d[::-1])
    latency, n0 = _check_centred(u, len(d), L, pair.latency)
    m0 = pair.metrics()
    if ripple_db is None:
        ripple_db = max(2 * max(m0["u_ripple_db"], m0["d_ripple_db"]), 1e-3)
    gu, gd = float(u.sum()), float(d.sum())

    def peaks(uu, dd):
        m = FilterPair(uu, dd, L, latency, pair.passband).metrics()
        return m["u_stop_db"], m["d_stop_db"]

    hist = [peaks(u, d)]
    kw = dict(transition_max_db=transition_max_db, ripple_db=ripple_db, n_grid=n_grid)
    for rnd in range(iterations):
        for who in order:
            if who == "d":
                d = _solve_kernel(u, len(d), spec, n0, gd, **kw)
            else:
                u = _solve_kernel(d, len(u), spec, n0, gu, **kw)
        hist.append(peaks(u, d))
        if rnd > 0 and max(hist[-2]) - max(hist[-1]) < min_gain_db:
            break
    c = L / u.sum()                                   # restore the DC gain L of u (the product u * d is unchanged)
    return FilterPair(u * c, d / c, L, latency, pair.passband, method=f"{pair.method}+jointminimax",
                      info={**pair.info, "joint_minimax": dict(history=hist, ripple_db=ripple_db,
                                                               transition_max_db=transition_max_db)})
