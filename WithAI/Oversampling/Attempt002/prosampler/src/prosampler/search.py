"""
Shortest pair for a specification: search over total length and the split between ``u`` and ``d``.

For every candidate ``(n_u, n_d)`` the start is a Kaiser-sinc ``u`` of ``n_u`` taps with its minimax partner of ``n_d``
taps (:func:`~prosampler.design_pair` with ``objective="minimax"``), refined by :func:`~prosampler.optimise_joint_slp`
with the passband ripple held at a reference value and, optionally, a cap on the transition gain.  The total length is
reduced stepwise from that of the plain pair until no split meets the specification any more.  Local optima: the result
is the shortest length *found*, not a proof of optimality.
"""
from __future__ import annotations

import numpy as np

from .design import design_pair
from .errors import ProsamplerError
from .jointlp import optimise_joint_slp


def _worst(fp) -> float:
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


def search_pair(L: int, passband: float, atten_db: float, *, transition_max_db=None, ripple_factor: float = 1.0,
                step: int = 8, n_u_offsets=(0, 8, 16, 24, 32, 40), iterations: int = 25, max_steps: int = 12,
                verbose: bool = False):
    """
    Shortest (total taps) pair meeting ``atten_db`` with passband ripple at most ``ripple_factor`` times that of the plain
    ``design_pair(L, passband, atten_db)`` and, if given, a transition cap ``transition_max_db`` (dB, both filters).

    Parameters
    ----------
    step : int
        Decrement of the total length per round (must be a multiple of ``2 L`` to keep the centred latency).
    n_u_offsets : sequence of int
        Candidate lengths of ``u`` relative to the plain pair's (even offsets keep ``u`` odd). Larger ``u`` at the same total is
        often much better (the plain pair has a short ``u`` and a long ``d``), hence the wide default range.
    iterations : int
        SLP iterations per candidate.

    Returns the best :class:`~prosampler.FilterPair` found (``info["search"]`` has the table of all candidates tried:
    total, n_u, n_d, worst stopband peak, bump, meets).  The plain pair's lengths are the starting point, so a result
    at that length is returned even when nothing shorter works.
    """
    if step % (2 * L):
        raise ValueError("step must be a multiple of 2 L")
    plain = design_pair(L, passband, atten_db)
    mp = plain.metrics()
    r_ref = ripple_factor * max(mp["u_ripple_db"], mp["d_ripple_db"])
    total0 = plain.n_u + plain.n_d
    tried, best = [], None
    for k in range(max_steps):
        total = total0 - k * step
        found = None
        for off in n_u_offsets:
            n_u = plain.n_u + off
            n_d = total - n_u
            if n_u < 2 * L + 1 or n_d < 2 * L + 1 or n_u % 2 == 0 or n_d % 2 == 0 or (n_u + n_d - 2) % (2 * L):
                continue
            try:
                start = design_pair(L, passband, atten_db, n_u=n_u, n_d=n_d, objective="minimax")
                ms = start.metrics()
                r = max(r_ref, ms["u_ripple_db"], ms["d_ripple_db"])
                fp = optimise_joint_slp(start, ripple_db=r, transition_max_db=transition_max_db, iterations=iterations)
            except ProsamplerError:
                continue
            m = fp.metrics()
            worst = max(m["u_stop_db"], m["d_stop_db"])
            bump = max(m["d_transition_peak_db"], m["u_transition_peak_db"])
            ok = worst <= -atten_db and (transition_max_db is None or bump <= transition_max_db + 0.02) \
                and max(m["u_ripple_db"], m["d_ripple_db"]) <= r * (1 + 1e-3) + 1e-6
            tried.append(dict(total=total, n_u=n_u, n_d=n_d, worst_db=worst, bump_db=bump, meets=bool(ok)))
            if verbose:
                print(f"  total {total} split {n_u}/{n_d}: worst {worst:.1f} dB, bump {bump:+.2f} dB, "
                      f"{'meets' if ok else 'misses'}", flush=True)
            if ok and (found is None or worst < _worst(found)):
                found = fp
        if found is None:
            break
        best = found
    if best is None:
        best = plain
    best.info["search"] = dict(table=tried, ripple_ref_db=r_ref, plain_total=total0,
                               transition_max_db=transition_max_db)
    return best
