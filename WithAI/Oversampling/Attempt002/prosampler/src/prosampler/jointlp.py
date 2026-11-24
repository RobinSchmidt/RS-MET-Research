"""
Simultaneous joint minimax of ``(u, d)`` by sequential linear programming (SLP).

Why not alternate?  For a fixed partner the transition band of the other kernel is almost forced by the Nyquist
condition, so alternating ``d``-steps and ``u``-steps cannot move the split of the roll-off between the two filters
(see :func:`prosampler.optimise_joint_minimax`).  Here both kernels move in the same step.

Everything is linear in the kernel parameters ``(a, b)`` (zero-phase amplitudes: stopband peak, passband flatness,
prescribed transition tube) except the Nyquist condition ``conv(u, d)[n0 + L k] = delta[k]``, which is bilinear.  Each
iteration linearises it at the current feasible point,

    J_a da + J_b db = 0,     sum(da . 1_u) = 0   (the DC gain of ``u`` stays ``L``),

adds a trust region ``|da|, |db| <= rho`` and solves the LP

    minimise  t + tube_weight * xi
    s.t.      |A_u| / L <= t,  |A_d| <= t   on the stopband,
              A_u / L, A_d within ``ripple_db`` of 1 in the passband,
              |A_u / L - target_u| <= tol + xi,  |A_d - target_d| <= tol + xi   in the transition band,

(``xi >= 0`` is an elastic slack so that a start outside the passband ripple bound, the tube or the cap is allowed; without a ``shape`` there is no tube;
``transition_max_db`` adds the elastic rows ``A_u / L, A_d <= 10**(x/20) + xi`` in the transition band).
The step is then projected back onto the exact Nyquist set by Gauss-Newton.  A step is accepted if the merit
``peak + tube_weight * tube_violation`` decreases; otherwise the trust region shrinks.  The limit is a local optimum.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ._util import conv_rows, mirror_matrix
from .errors import ProsamplerError
from .minimax import _amp_matrix, _check_centred
from .spec import Spec
from .transition import TransitionShape


def _lp(cost, A_ub, b_ub, A_eq, b_eq, bounds):
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method,
                      options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if res.status == 0 and res.x is not None:
            return res
    return res


DEFAULT_RIPPLE_DB = 0.1      # default allowed peak-to-peak passband ripple of each filter (dB); inaudible and a good trade-off, see docs/theory.md


def optimise_joint_slp(pair, *, shape: TransitionShape | None = None, transition_max_db="ripple", iterations: int = 40, trust: float = 0.05,
                       ripple_db=DEFAULT_RIPPLE_DB, tube_weight: float = 1.0, n_grid: int = 3000, min_improve: float = 1e-4,
                       verbose: bool = False):
    """
    Simultaneous joint minimax refinement of a symmetric centred pair, optionally with a prescribed transition shape.

    Parameters
    ----------
    pair : FilterPair
        Start point (exact Nyquist, symmetric, odd lengths, centred latency), e.g. from :func:`design_pair`.
    shape : TransitionShape, optional
        Prescribed roll-off of both filters in the transition band (see :mod:`prosampler.transition`), as a tube of
        half-width ``shape.tol`` around ``u_target`` / ``d_target``.  ``None``: no transition constraint.
    transition_max_db : float, "ripple" or None
        Default ``"ripple"``.  ``None``: no cap.  Cap on the gain of each filter between the passband and stopband edges, relative to its DC gain (``0`` = no
        overshoot at all).  Elastic like the tube: a start that violates it is allowed and driven into the cap.  ``"ripple"``: the cap
        is half the allowed peak-to-peak passband ripple (``ripple_db / 2``), i.e. the bump may not exceed the gain that the
        passband ripple reaches anyway.
    iterations : int
        Maximum number of SLP iterations (each is one LP of a few seconds for kernels of a few hundred taps).
    trust : float
        Initial trust region as a fraction of the largest coefficient of each kernel.
    ripple_db : float or None
        Allowed peak-to-peak passband ripple of each filter in dB (as reported by ``metrics()``).  Default: 0.1 dB
        (``DEFAULT_RIPPLE_DB``).  ``None``: twice the start's ripple, at least 0.001 dB.  Looser values let the optimiser trade flatness for stopband attenuation.
        The bound is elastic: a start that exceeds it (by more than 5 %) is driven down to it, with a violation weight 100 times that of the tube and the cap.
    tube_weight : float
        Weight of the elastic tube violation against the stopband peak in the LP objective and the acceptance test.
    min_improve : float
        Relative decrease of the merit below which three consecutive accepted steps end the iteration.

    Returns a new :class:`~prosampler.FilterPair` (same lengths and latency, exact Nyquist); ``info["slp"]`` holds the
    history ``(u_stop_db, d_stop_db, tube_violation, trust)`` per accepted step.
    """
    from .pair import FilterPair
    L = pair.L
    spec = Spec(L, pair.passband, None)
    u0 = np.array(pair.u, float)
    d0 = np.array(pair.d, float)
    u0, d0 = 0.5 * (u0 + u0[::-1]), 0.5 * (d0 + d0[::-1])
    latency, n0 = _check_centred(u0, len(d0), L, pair.latency)
    n_u, n_d = len(u0), len(d0)
    n_h = n_u + n_d - 1
    Su, Sd = mirror_matrix(n_u), mirror_matrix(n_d)
    ma, mb = Su.shape[1], Sd.shape[1]
    a = u0[(n_u - 1) // 2 - np.arange(ma)]
    b = d0[(n_d - 1) // 2 - np.arange(mb)]
    idx = np.arange(n0 % L, n_h, L)
    idx = idx[idx <= n0]
    e = (idx == n0).astype(float)
    gu = Su.sum(axis=0)

    def resid(a_, b_):
        return np.convolve(Su @ a_, Sd @ b_)[idx] - e

    def jac(a_, b_):
        return conv_rows(Sd @ b_, n_u, idx) @ Su, conv_rows(Su @ a_, n_d, idx) @ Sd

    def project(a_, b_):
        for _ in range(30):
            R = np.concatenate([resid(a_, b_), [gu @ a_ - L]])
            if np.max(np.abs(R)) < 1e-15:
                break
            Ja, Jb = jac(a_, b_)
            J = np.block([[Ja, Jb], [gu[None, :], np.zeros((1, mb))]])
            step = np.linalg.lstsq(J, R, rcond=None)[0]
            a_, b_ = a_ - step[:ma], b_ - step[ma:]
        return a_, b_

    a, b = project(a, b)

    grid = spec.grid(n_grid)
    w_pass, w_stop = grid.w[grid.passband], grid.w[grid.stopband]
    w_tr = grid.w[grid.transition]
    x_tr = (w_tr - spec.passband_edge) / (spec.stopband_edge - spec.passband_edge)
    w_fine = np.linspace(spec.stopband_edge, np.pi, 8 * n_grid)
    Au_f, Ad_f = _amp_matrix(n_u, w_fine, Su), _amp_matrix(n_d, w_fine, Sd)
    Au_p, Ad_p = _amp_matrix(n_u, w_pass, Su), _amp_matrix(n_d, w_pass, Sd)
    Au_t, Ad_t = _amp_matrix(n_u, w_tr, Su), _amp_matrix(n_d, w_tr, Sd)
    elastic = True                                   # slack xi: ripple, cap and tube are elastic (a start may violate them)
    gmax = None
    if shape is not None:
        tu, td = shape.u_target(x_tr), shape.d_target(x_tr)

    def peaks(a_, b_):
        return float(np.max(np.abs(Au_f @ a_)) / L), float(np.max(np.abs(Ad_f @ b_)))

    def tube_violation(a_, b_):
        ru, rd = Au_p @ a_ / L, Ad_p @ b_
        v = 100 * max(ru.max() - hi, lo - ru.min(), rd.max() - hi, lo - rd.min())          # passband ripple excess (weighted x100)
        if shape is not None:
            dev = max(np.max(np.abs(Au_t @ a_ / L - tu)), np.max(np.abs(Ad_t @ b_ - td)))
            v = max(v, dev - shape.tol)
        if gmax is not None:
            v = max(v, np.max(Au_t @ a_ / L) - gmax, np.max(Ad_t @ b_) - gmax)
        return float(max(v, 0.0))

    def ripple_of(a_, b_):
        ru, rd = Au_p @ a_ / L, Ad_p @ b_
        return max(20 * np.log10(ru.max() / ru.min()), 20 * np.log10(rd.max() / rd.min()))

    if ripple_db is None:
        ripple_db = max(2 * ripple_of(a, b), 1e-3)
    lo, hi = 10 ** (-ripple_db / 40), 10 ** (ripple_db / 40)          # peak-to-peak ripple_db, like metrics()
    if isinstance(transition_max_db, str):
        if transition_max_db != "ripple":
            raise ValueError('transition_max_db must be a number, None or "ripple"')
        transition_max_db = ripple_db / 2          # the bump may not exceed the gain the passband ripple reaches anyway
    gmax = None if transition_max_db is None else 10 ** (transition_max_db / 20)
    _ru, _rd = Au_p @ a / L, Ad_p @ b
    # ripple rows are hard if the start satisfies them, elastic (driven down to the bound) if it does not
    rip_soft = 0.01 if ripple_of(a, b) > 1.05 * ripple_db else 0.0          # (5 % slack: LP grid vs metrics grid)

    def merit(a_, b_):
        return max(peaks(a_, b_)) + tube_weight * tube_violation(a_, b_)

    extra = np.array([])
    cur = merit(a, b)
    hist = []

    def record():
        pu, pd = peaks(a, b)
        hist.append((20 * np.log10(pu), 20 * np.log10(pd), tube_violation(a, b), rho))

    rho = trust
    stall = 0
    record()
    for it in range(iterations):
        pu, pd = peaks(a, b)
        sigma = max(pu, pd)
        # exchange points: local maxima of the stopband magnitudes on the fine grid
        add = []
        for A_f, v, g in ((Au_f, a, L), (Ad_f, b, 1.0)):
            mag = np.abs(A_f @ v) / g
            loc = np.nonzero((mag >= np.roll(mag, 1)) & (mag >= np.roll(mag, -1)) & (mag > 0.5 * sigma))[0]
            add.append(w_fine[loc[(loc > 0) & (loc < len(mag) - 1)]])
        extra = np.unique(np.concatenate([extra, *add]))
        ws = np.unique(np.concatenate([w_stop, extra]))
        Au_s, Ad_s = _amp_matrix(n_u, ws, Su), _amp_matrix(n_d, ws, Sd)

        nv = ma + mb + 1 + (1 if elastic else 0)
        rows, rhs = [], []

        def add_rows(blocks_u, blocks_d, s_coef_u, s_coef_d, xi_coef_u, xi_coef_d, rhs_vec):
            M = np.zeros((len(rhs_vec), nv))
            M[:, :ma] = blocks_u
            M[:, ma:ma + mb] = blocks_d
            M[:, ma + mb] = s_coef_u
            if elastic:
                M[:, -1] = xi_coef_u
            rows.append(M)
            rhs.append(rhs_vec)

        n_s = len(ws)
        Z_u, Z_d = np.zeros((n_s, ma)), np.zeros((n_s, mb))
        for sg in (1.0, -1.0):
            add_rows(sg * Au_s, Z_d, -L * sigma, 0, 0, 0, -sg * (Au_s @ a))            # |A_u| <= L t
            add_rows(Z_u, sg * Ad_s, -sigma, 0, 0, 0, -sg * (Ad_s @ b))                # |A_d| <= t
        n_p = len(w_pass)
        add_rows(Au_p, np.zeros((n_p, mb)), 0, 0, -L * rip_soft, 0, L * hi - Au_p @ a)
        add_rows(-Au_p, np.zeros((n_p, mb)), 0, 0, -L * rip_soft, 0, -L * lo + Au_p @ a)
        add_rows(np.zeros((n_p, ma)), Ad_p, 0, 0, -1.0 * rip_soft, 0, hi - Ad_p @ b)
        add_rows(np.zeros((n_p, ma)), -Ad_p, 0, 0, -1.0 * rip_soft, 0, -lo + Ad_p @ b)
        n_t = len(w_tr)
        Z_ut, Z_dt = np.zeros((n_t, ma)), np.zeros((n_t, mb))
        if gmax is not None:
            add_rows(Au_t, Z_dt, 0, 0, -L, 0, L * gmax - Au_t @ a)
            add_rows(Z_ut, Ad_t, 0, 0, -1.0, 0, gmax - Ad_t @ b)
        if shape is not None:
            add_rows(Au_t, Z_dt, 0, 0, -L, 0, L * (tu + shape.tol) - Au_t @ a)
            add_rows(-Au_t, Z_dt, 0, 0, -L, 0, L * (shape.tol - tu) + Au_t @ a)
            add_rows(Z_ut, Ad_t, 0, 0, -1.0, 0, (td + shape.tol) - Ad_t @ b)
            add_rows(Z_ut, -Ad_t, 0, 0, -1.0, 0, (shape.tol - td) + Ad_t @ b)
        A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)

        Ja, Jb = jac(a, b)
        A_eq = np.zeros((len(idx) + 1, nv))
        A_eq[:len(idx), :ma], A_eq[:len(idx), ma:ma + mb] = Ja, Jb
        A_eq[len(idx), :ma] = gu
        b_eq = np.concatenate([-resid(a, b), [0.0]])
        ra, rb = rho * np.max(np.abs(a)), rho * np.max(np.abs(b))
        bounds = [(-ra, ra)] * ma + [(-rb, rb)] * mb + [(10 ** (-160 / 20) / sigma, None)]
        cost = np.zeros(nv)
        cost[ma + mb] = 1.0
        if elastic:
            bounds.append((0.0, None))
            cost[-1] = tube_weight / sigma
        res = _lp(cost, A_ub, b_ub, A_eq, b_eq, bounds)
        if res.status != 0 or res.x is None:
            rho *= 0.5
            if rho < 1e-6:
                break
            continue
        da, db = res.x[:ma], res.x[ma:ma + mb]
        a_new, b_new = project(a + da, b + db)
        new = merit(a_new, b_new)
        if new < cur * (1 - 1e-9):
            gain = (cur - new) / cur
            touched = max(np.max(np.abs(da)) / max(ra, 1e-300), np.max(np.abs(db)) / max(rb, 1e-300)) > 0.95
            a, b, cur = a_new, b_new, new
            record()
            hist[-1] = hist[-1][:3] + (rho,)
            if touched:
                rho = min(rho * 1.5, 0.3)
            stall = stall + 1 if gain < min_improve else 0
            if verbose:
                print(f"  it {it:2d}: u {hist[-1][0]:.1f} d {hist[-1][1]:.1f} dB, tube viol {hist[-1][2]:.4f}, rho {rho:.3f}")
            if stall >= 3:
                break
        else:
            rho *= 0.5
            if rho < 1e-6:
                break
    u, d = Su @ a, Sd @ b
    out = FilterPair(u, d, L, latency, pair.passband, method=f"{pair.method}+jointslp",
                     info={**pair.info, "slp": dict(history=hist, ripple_db=ripple_db,
                                                    shape=None if shape is None else vars(shape).copy(),
                                                    transition_max_db=transition_max_db)})
    return out
