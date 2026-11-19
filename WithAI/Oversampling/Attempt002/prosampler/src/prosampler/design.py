"""High-level design of a perfect-reconstruction ``(u, d)`` pair."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ._util import is_symmetric
from .errors import ProsamplerError, SpecNotReached
from .kernels import design_kaiser_sinc, kaiser_sinc
from .nyquist import partner_length
from .pair import FilterPair
from .partner import design_partner, resolve_latency
from .spec import Spec


def design_pair(L: int, passband: float, atten_db: Optional[float] = None, *, u=None,
                n_u: Optional[int] = None, n_d: Optional[int] = None, beta: Optional[float] = None,
                latency: Optional[int] = None, method: str = "nullspace", objective: str = "grid",
                stop_weight: float = 1.0, transition_weight: float = 0.0, n_grid: int = 3000,
                max_pass_ripple_db: Optional[float] = None, max_transition_gain_db: float = 3.0,
                transition_max_db: Optional[float] = None, passband_flat_db: Optional[float] = None,
                strict: bool = False) -> FilterPair:
    """
    Design ``u`` and its exact partner ``d``  (pipeline steps 1 and 2).

    Two ways to call it:

    * **Spec-driven** (``atten_db`` given, ``n_d`` not given): the lengths are chosen automatically.  Without ``u``
      the shortest Kaiser-sinc meeting the attenuation is used; with an external symmetric ``u`` it is kept.  The
      shortest ``d`` (smallest latency) meeting the spec is then found by bisection (assumes that longer is better).
    * **Length-driven** (``n_d`` given): ``u`` is ``u`` or a Kaiser-sinc of ``n_u`` taps (give ``atten_db`` or
      ``beta`` to set the window); ``d`` has ``n_d`` taps.  ``atten_db`` is then only used to judge ``meets_spec``.

    Parameters
    ----------
    L, passband : see :class:`~prosampler.Spec`.  ``passband`` is a fraction of the original Nyquist frequency.
    atten_db : float, optional
        Required stopband attenuation of BOTH kernels.
    u : array_like, optional
        External upsampling kernel (any window, optimised FIR, ...).  It is rescaled to DC gain ``L``.
    n_u, n_d : int, optional
        Kernel lengths (odd for linear phase).  For a centred linear-phase pair ``n_u + n_d - 2`` must be a
        multiple of ``2 L``.
    beta : float, optional
        Kaiser beta for ``u`` in length-driven mode (alternative to ``atten_db``).
    latency : int, optional
        Roundtrip latency in base-rate samples; default is the centred one.
    method, objective, stop_weight, transition_weight, n_grid :
        Passed to :func:`~prosampler.design_partner`.  ``objective="minimax"`` minimises the stopband peak (LP).
    transition_max_db, passband_flat_db : float, optional
        Only for ``objective="minimax"``: bound on the gain of ``d`` between the passband and stopband edges, and on
        the passband deviation of ``u * d / L`` (both in dB).
    max_pass_ripple_db : float, optional
        Largest accepted passband ripple of each kernel (default ``max(0.01, 4 * 8.686 * 10**(-atten_db/20))`` dB).
    max_transition_gain_db : float
        Largest accepted peak gain of ``d`` between the passband and stopband edges.
    strict : bool
        Raise :class:`~prosampler.SpecNotReached` if the spec is not met (default: warn and set
        ``info["meets_spec"] = False``).

    Returns
    -------
    FilterPair
        ``u`` with DC gain ``L``, ``d`` with DC gain about 1, exact to machine precision.
        ``info`` holds ``meets_spec`` and the design parameters.
    """
    spec = Spec(L, passband, atten_db)
    ripple_limit = None
    if atten_db is not None:
        ripple_limit = max_pass_ripple_db if max_pass_ripple_db is not None else max(0.01, 4 * 8.686 * 10 ** (-atten_db / 20))
    pkw = dict(method=method, objective=objective, stop_weight=stop_weight,
               transition_weight=transition_weight, n_grid=n_grid)
    if objective == "minimax":
        pkw.update(transition_max_db=transition_max_db, passband_flat_db=passband_flat_db)

    # ------------------------------------------------------------------ the upsampling kernel
    external = u is not None
    if external:
        u = np.asarray(u, dtype=float)
        if abs(u.sum()) < 1e-12:
            raise ValueError("u has (almost) zero DC gain")
        u = u * (L / u.sum())
    elif n_u is None:
        if n_d is not None:
            raise ValueError("give n_u (or u) together with n_d")
        if atten_db is None:
            raise ValueError("give atten_db (spec-driven) or n_u / n_d / u (length-driven)")
        u = design_kaiser_sinc(spec)
    else:
        if beta is None and atten_db is None:
            raise ValueError("to design a Kaiser-sinc u of n_u taps give beta or atten_db (or pass u)")
        u = kaiser_sinc(L, n_u, atten_db=None if beta is not None else atten_db, beta=beta)
    n_u = len(u)
    tag = "external-u" if external else "kaiser-sinc"

    def judge(fp: FilterPair):
        m = fp.metrics(n_grid=4000)
        ok = (m["u_stop_db"] <= -atten_db and m["d_stop_db"] <= -atten_db
              and m["u_ripple_db"] <= ripple_limit and m["d_ripple_db"] <= ripple_limit
              and m["d_transition_peak_db"] <= max_transition_gain_db)
        return ok, m

    def make(d, lat):
        return FilterPair(u, d, L, lat, passband, method=f"{tag}+partner[{method},{objective}]",
                          info=dict(objective=objective, partner_method=method, atten_db=atten_db, u_rescaled=external))

    # ------------------------------------------------------------------ length-driven
    if n_d is not None:
        lat = resolve_latency(u, n_d, L, latency)
        fp = make(design_partner(u, spec, n_d, lat, **pkw), lat)
        fp.info["meets_spec"] = judge(fp)[0] if atten_db is not None else None
        if fp.info["meets_spec"] is False:
            _report_failure(strict, f"the pair does not meet {atten_db} dB / ripple limits", fp.metrics())
        return fp

    # ------------------------------------------------------------------ spec-driven
    if atten_db is None:
        raise ValueError("spec-driven design needs atten_db")
    if not (is_symmetric(u) and n_u % 2 == 1):
        raise ValueError("spec-driven design needs a symmetric odd-length u; for other kernels give n_d and latency")
    if latency is not None:
        raise ValueError("latency is chosen by the search in spec-driven mode; give n_d to fix it")

    def attempt(K):
        n = partner_length(n_u, L, K)
        try:
            fp = make(design_partner(u, spec, n, K, **pkw), K)
        except ProsamplerError:
            return None, False
        ok, _ = judge(fp)
        return fp, ok

    # u itself must meet the spec: no partner can repair a bad u
    probe = FilterPair(u, np.array([1.0]), L, 0, passband)
    mu = probe.metrics()
    if mu["u_stop_db"] > -atten_db or mu["u_ripple_db"] > ripple_limit:
        msg = (f"u itself does not meet the spec (stopband {mu['u_stop_db']:.1f} dB, ripple {mu['u_ripple_db']:.4f} dB); "
               "no partner can fix that")
        raise SpecNotReached(msg)

    def bisect(K_lo, K_hi):
        fp, ok = attempt(K_hi)
        if not ok:
            return None, fp
        best = fp
        while K_hi - K_lo > 1:
            mid = (K_lo + K_hi) // 2
            fp, ok = attempt(mid)
            if ok:
                K_hi, best = mid, fp
            else:
                K_lo = mid
        return best, fp

    K_lo = int(np.ceil((n_u - 1) / L))
    K_hi = int(np.ceil(2.6 * n_u / L))
    best, last = None, None
    for _ in range(4):
        best, last = bisect(K_lo, K_hi)
        if best is not None:
            break
        K_lo, K_hi = K_hi, int(K_hi * 1.5)
    if best is None:
        _report_failure(strict, f"no partner up to latency {K_hi} meets {atten_db} dB", last.metrics() if last else {})
        if last is None:
            raise SpecNotReached(f"no partner could be constructed for this u (tried latencies up to {K_hi})")
        best = last
        best.info["meets_spec"] = False
    else:
        best.info["meets_spec"] = True
    return best


def _report_failure(strict: bool, msg: str, metrics: dict) -> None:
    detail = ", ".join(f"{k}={v:.2f}" for k, v in metrics.items() if isinstance(v, float) and k.endswith("_db"))
    full = f"{msg} ({detail})" if detail else msg
    if strict:
        raise SpecNotReached(full)
    warnings.warn(full, stacklevel=3)
