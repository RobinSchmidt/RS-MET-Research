"""
Joint optimisation of ``(u, d)`` (EXPERIMENTAL), length rebalancing and tying ``u = d``.

Pipeline steps 3-5.  Everything here requires a symmetric, centred, odd-length pair whose ``d`` has zero end taps
(which is what :func:`~prosampler.design_pair` returns) and keeps the Nyquist(L) condition exact at every step.

* :func:`optimise_joint`: optimise ``u`` AND ``d`` together.  The partner design fixes ``u`` and optimises only ``d``;
  here both move, with the balanced objective ``E_stop(u)/L^2 + E_stop(d)`` under the exact (now bilinear)
  Nyquist constraints plus ``sum(u) = L``.  SLSQP with rescaling and perturbed restarts, then Gauss-Newton polish.
* :func:`rebalance`: move taps from ``d`` to ``u`` two at a time (``n_u + n_d`` stays constant), re-optimising after
  each move, until the lengths are equal.  The path is recorded; ``target="best"`` returns the point with the
  best worst-case stopband along it.
* :func:`tie`: make ``u`` and ``d`` equal (up to the gain split) by a penalty continuation on an equal-length pair,
  average, and polish with the exact matched solver, giving a true ``u = d = g`` pair.

Findings from the experiments behind this module: the gain of joint optimisation depends on how far the starting
``u`` is from optimal (about 5 dB worst-case at L=4 / 40 dB from Kaiser-sinc kernels; little at L=4 / 100 dB);
while rebalancing the stopband stays roughly level for a while and may then degrade; tying typically costs 20-35 dB
at equal total length (L=4, 40 dB start: about -48 dB at equal lengths, about -28 dB after tying).  Treat these
routines as research tools.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from ._util import is_symmetric, mirror_matrix
from .analysis import band_energy_matrix
from .errors import ProsamplerError
from .matched import optimise_matched
from .pair import FilterPair
from .spec import Spec


class _Joint:
    """Parametrisation, constraints and objective of the joint problem for fixed lengths."""

    def __init__(self, L: int, Nu: int, Nd: int, passband: float, passband_weight: float = 0.0):
        self.L, self.Nu, self.Nd, self.p = L, Nu, Nd, passband
        self.pw = float(passband_weight)
        if Nu % 2 == 0 or Nd % 2 == 0 or (Nu + Nd - 2) % (2 * L):
            raise ProsamplerError(f"lengths ({Nu}, {Nd}) are not a centred linear-phase pair "
                                  f"(need odd lengths and (n_u + n_d - 2) a multiple of 2L)")
        self.n0 = (Nu + Nd - 2) // 2
        self.K = self.n0 // L
        self.Su, self.Sd = mirror_matrix(Nu), mirror_matrix(Nd, drop_ends=True)
        ws = Spec(L, passband).stopband_rad
        self.Qu = self.Su.T @ band_energy_matrix(Nu, ws, np.pi) @ self.Su / L ** 2
        self.Qd = self.Sd.T @ band_energy_matrix(Nd, ws, np.pi) @ self.Sd
        self.mu, self.md = self.Su.shape[1], self.Sd.shape[1]
        self.idx = self.n0 + L * np.arange(self.K)          # k = K is implied by d's zero end taps
        self.tgt = (np.arange(self.K) == 0).astype(float)
        self.gu = self.Su.sum(axis=0)
        self.lam = 0.0                                       # tie penalty weight (see tie())
        if self.pw:
            self._init_passband()

    def _init_passband(self):
        """Quadratic form of the passband flatness: int_0^{p pi/L} (A(w)/g - 1)^2 dw per filter (g = DC gain target)."""
        x, wgt = np.polynomial.legendre.leggauss(600)
        wp = self.p * np.pi / self.L
        w, wgt = 0.5 * wp * (x + 1), 0.5 * wp * wgt
        def parts(N, S, g):
            A = np.cos(np.outer(w, np.arange(N) - (N - 1) / 2)) @ S / g      # zero-phase amplitude / g
            return A.T @ (wgt[:, None] * A), A.T @ wgt
        self.pc = float(wgt.sum())                          # 0.5 * (constant of u) + 0.5 * (constant of d)
        self.Pu, self.pu = parts(self.Nu, self.Su, self.L)
        self.Pd, self.pd = parts(self.Nd, self.Sd, 1.0)

    # ---- parametrisation
    def to_x(self, u, d):
        return np.concatenate([[u[(self.Nu - 1) // 2 - k] for k in range(self.mu)],
                               [d[(self.Nd - 1) // 2 - k] for k in range(self.md)]])

    def filt(self, x):
        return self.Su @ x[:self.mu], self.Sd @ x[self.mu:]

    # ---- constraints
    def r(self, x):
        u, d = self.filt(x)
        h = np.convolve(u, d)
        return np.concatenate([h[self.idx] - self.tgt, [u.sum() - self.L]])

    def J(self, x):
        u, d = self.filt(x)
        Ju, Jd = np.zeros((self.K, self.Nu)), np.zeros((self.K, self.Nd))
        ju, jd = np.arange(self.Nu), np.arange(self.Nd)
        for k, pp in enumerate(self.idx):
            i = pp - ju
            ok = (i >= 0) & (i < self.Nd)
            Ju[k, ju[ok]] = d[i[ok]]
            i = pp - jd
            ok = (i >= 0) & (i < self.Nu)
            Jd[k, jd[ok]] = u[i[ok]]
        top = np.hstack([Ju @ self.Su, Jd @ self.Sd])
        return np.vstack([top, np.concatenate([self.gu, np.zeros(self.md)])[None, :]])

    # ---- objective (stopband energy plus optional tie penalty)
    def diff(self, x):
        u, d = self.filt(x)
        return u / np.sqrt(self.L) - d * np.sqrt(self.L)

    def f(self, x):
        a, b = x[:self.mu], x[self.mu:]
        val = 0.5 * (a @ self.Qu @ a + b @ self.Qd @ b)
        if self.pw:
            val += self.pw * (0.5 * (a @ self.Pu @ a + b @ self.Pd @ b) - self.pu @ a - self.pd @ b + self.pc)
        if self.lam:
            dd = self.diff(x)
            val += 0.5 * self.lam * dd @ dd
        return val

    def grad(self, x):
        a, b = x[:self.mu], x[self.mu:]
        g = np.concatenate([self.Qu @ a, self.Qd @ b])
        if self.pw:
            g += self.pw * np.concatenate([self.Pu @ a - self.pu, self.Pd @ b - self.pd])
        if self.lam:
            dd = self.diff(x)
            g[:self.mu] += self.lam * (self.Su.T @ dd) / np.sqrt(self.L)
            g[self.mu:] -= self.lam * (self.Sd.T @ dd) * np.sqrt(self.L)
        return g

    def polish(self, x, iters=40):
        for _ in range(iters):
            rr = self.r(x)
            if np.max(np.abs(rr)) < 1e-16:
                break
            x = x - np.linalg.lstsq(self.J(x), rr, rcond=None)[0]
        return x

    def optimise(self, x, restarts=4, maxiter=150):
        rng = np.random.default_rng(0)
        stalls = 0
        x = self.polish(x)
        for _ in range(restarts):
            f0 = self.f(x)
            sc = 1.0 / f0
            start = x if stalls == 0 else self.polish(x + 1e-6 * np.sqrt(f0) * rng.standard_normal(len(x)))
            res = minimize(lambda t: sc * self.f(t), start, jac=lambda t: sc * self.grad(t), method="SLSQP",
                           constraints=[dict(type="eq", fun=self.r, jac=self.J)],
                           options=dict(maxiter=maxiter, ftol=1e-14))
            xn = self.polish(res.x)
            if self.f(xn) < 0.999 * f0:
                x, stalls = xn, 0
            else:
                stalls += 1
                if stalls >= 2:
                    break
        return x


def _prepare(pair: FilterPair, passband_weight: float = 0.0) -> _Joint:
    if pair.passband is None:
        raise ProsamplerError("the pair needs a passband")
    if not (is_symmetric(pair.u) and is_symmetric(pair.d) and pair.n_u % 2 == 1 and pair.n_d % 2 == 1):
        raise ProsamplerError("joint optimisation supports symmetric odd-length u and d only")
    jt = _Joint(pair.L, pair.n_u, pair.n_d, pair.passband, passband_weight)
    if jt.K != pair.latency:
        raise ProsamplerError(f"pair latency {pair.latency} is not the centred latency {jt.K}")
    if abs(pair.d[0]) > 1e-6 * np.max(np.abs(pair.d)):
        raise ProsamplerError("d must have (numerically) zero end taps, as produced by design_pair")
    return jt


def _start_x(jt: _Joint, pair: FilterPair):
    d = 0.5 * (pair.d + pair.d[::-1])
    d[0] = d[-1] = 0.0
    return jt.to_x(0.5 * (pair.u + pair.u[::-1]), d)


def _as_pair(jt: _Joint, x, pair: FilterPair, tag: str, **info) -> FilterPair:
    u, d = jt.filt(x)
    return FilterPair(u, d, jt.L, jt.K, pair.passband, method=f"{pair.method}+{tag}", info={**pair.info, **info})


def _worst_db(fp: FilterPair) -> float:
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


def optimise_joint(pair: FilterPair, *, restarts: int = 4, maxiter: int = 150, passband_weight: float = 1.0,
                   on_worse_peak: str = "warn") -> FilterPair:
    """
    Optimise ``u`` and ``d`` jointly (EXPERIMENTAL; exact Nyquist and ``sum(u) = L`` at every step).

    The objective is the balanced stopband ENERGY plus ``passband_weight`` times the passband flatness energy of
    both filters.  Without the passband term (``passband_weight=0``) the stopband peak can improve by a few dB but
    u and d ripple against each other (up to several dB per filter), because the Nyquist condition only fixes the
    product ``u * d``.  With it, the energy still drops, but the worst stopband PEAK is usually not better than that
    of the plain ``design_pair`` result: do not expect a gain in peak attenuation.

    ``on_worse_peak``: ``"warn"`` returns the result anyway and emits a warning if the worst stopband peak got worse;
    ``"keep"`` returns the input pair unchanged in that case.  Before/after numbers are in ``info["joint"]``.
    The input pair must be symmetric, centred and have ``d`` with zero end taps (as from ``design_pair``).
    """
    import warnings
    if on_worse_peak not in ("warn", "keep"):
        raise ProsamplerError("on_worse_peak must be 'warn' or 'keep'")
    jt = _prepare(pair, passband_weight)
    f0 = jt.f(_start_x(jt, pair))
    x = jt.optimise(_start_x(jt, pair), restarts, maxiter)
    out = _as_pair(jt, x, pair, "joint")
    mb, ma = pair.metrics(), out.metrics()
    out.info["joint"] = dict(
        objective_ratio=float(jt.f(x) / f0) if f0 else float("nan"), passband_weight=float(passband_weight),
        stop_db_before=(mb["u_stop_db"], mb["d_stop_db"]), stop_db_after=(ma["u_stop_db"], ma["d_stop_db"]),
        ripple_db_before=(mb["u_ripple_db"], mb["d_ripple_db"]), ripple_db_after=(ma["u_ripple_db"], ma["d_ripple_db"]))
    if _worst_db(out) > _worst_db(pair) + 1e-9:
        if on_worse_peak == "keep":
            return pair
        warnings.warn(f"joint optimisation made the worst stopband peak worse "
                      f"({_worst_db(pair):.1f} -> {_worst_db(out):.1f} dB)", stacklevel=2)
    return out


def rebalance(pair: FilterPair, *, target: str = "equal", step: int = 2, restarts: int = 4, maxiter: int = 150,
              progress: Optional[Callable[[int, int, float, float], None]] = None) -> FilterPair:
    """
    Shift taps from ``d`` to ``u`` (``step`` at a time, ``n_u + n_d`` constant) with joint re-optimisation, until
    ``n_u = n_d``.

    Parameters
    ----------
    target : {"equal", "best"}
        ``"equal"`` returns the final equal-length pair, ``"best"`` the point along the path with the lowest
        worst-case stopband (``min(max(u_stop, d_stop))``).
    step : int
        Even number of taps moved per iteration.
    progress : callable(n_u, n_d, u_stop_db, d_stop_db), optional
        Called after every optimisation (useful for long paths: each step takes seconds for ~500 taps).

    ``info["path"]`` of the result lists ``(n_u, n_d, u_stop_db, d_stop_db)`` for every visited split.
    """
    if target not in ("equal", "best"):
        raise ValueError("target must be 'equal' or 'best'")
    if step < 2 or step % 2:
        raise ValueError("step must be an even integer >= 2")
    jt = _prepare(pair)
    rng = np.random.default_rng(0)
    u, d = 0.5 * (pair.u + pair.u[::-1]), 0.5 * (pair.d + pair.d[::-1])
    d[0] = d[-1] = 0.0
    path: List = []
    best, best_val = None, np.inf
    while True:
        jt = _Joint(pair.L, len(u), len(d), pair.passband)
        x = jt.optimise(jt.to_x(u, d), restarts, maxiter)
        u, d = jt.filt(x)
        fp = _as_pair(jt, x, pair, "rebalanced")
        m = fp.metrics()
        path.append((len(u), len(d), m["u_stop_db"], m["d_stop_db"]))
        if progress:
            progress(*path[-1])
        worst = max(m["u_stop_db"], m["d_stop_db"])
        if worst < best_val:
            best, best_val = fp, worst
        if len(d) < len(u) + step:
            break
        pad = step // 2
        u = np.pad(u, pad)
        u = u + 1e-7 * rng.standard_normal(len(u)) * (np.abs(u) < 1e-12)       # break zero-padding degeneracy
        u = 0.5 * (u + u[::-1])
        d = d[pad:-pad].copy()
        d[0] = d[-1] = 0.0
    result = fp if target == "equal" else best
    result.info = {**result.info, "path": path, "rebalance_target": target}
    return result


def tie(pair: FilterPair, *, rel_schedule: Sequence[float] = (0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10),
        restarts: int = 4, maxiter: int = 200) -> FilterPair:
    """
    Turn an equal-length pair into a matched pair ``u = d = g`` (``g`` with DC gain ``sqrt(L)``).

    A penalty on ``|u/sqrt(L) - d*sqrt(L)|^2`` (weights ``rel * f0 / base``, ramped up along ``rel_schedule``) pulls
    the pair together while the Nyquist condition stays exact.  The average of the two is then refined by the exact
    matched solver (:func:`~prosampler.optimise_matched`), so the result is an exact ``u = d`` pair.

    ``info`` holds ``tie_path`` (relative difference and stopbands per penalty weight), ``averaged_stop_db``
    and ``refined_stop_db``.  Needs ``n_u = n_d = L*K + 1`` with ``L*K`` even and ``L >= 3``
    (for ``L = 2`` the exact matched problem has no freedom; see :mod:`prosampler.matched`).
    """
    if pair.n_u != pair.n_d:
        raise ProsamplerError("tie needs equal lengths; run rebalance first")
    jt = _prepare(pair)
    L, N, K = pair.L, pair.n_u, jt.K
    if N != L * K + 1:
        raise ProsamplerError("tie needs n_u = n_d = L*K + 1")
    x = _start_x(jt, pair)
    f0 = jt.f(x)
    base = float(np.linalg.norm(jt.diff(x)) ** 2)
    if base == 0:
        base = 1.0
    path = []
    for rel in rel_schedule:
        jt.lam = rel * f0 / base
        x = jt.optimise(x, restarts, maxiter)
        u, d = jt.filt(x)
        dd = jt.diff(x)
        fp = FilterPair(u, d, L, K, pair.passband)
        m = fp.metrics()
        path.append((rel, float(np.linalg.norm(dd) / np.linalg.norm(u / np.sqrt(L))), m["u_stop_db"], m["d_stop_db"]))
    gavg = 0.5 * (u / np.sqrt(L) + d * np.sqrt(L))
    gavg[0] = gavg[-1] = 0.0
    avg = FilterPair(gavg, gavg, L, K, pair.passband)
    g = optimise_matched(L, K, pair.passband, g_init=gavg)
    out = FilterPair(g, g, L, K, pair.passband, method=f"{pair.method}+tied",
                     info={**pair.info, "tie_path": path, "averaged_stop_db": avg.metrics()["u_stop_db"],
                           "averaged_nyquist_error": avg.nyquist_error})
    out.info["refined_stop_db"] = out.metrics()["u_stop_db"]
    return out
