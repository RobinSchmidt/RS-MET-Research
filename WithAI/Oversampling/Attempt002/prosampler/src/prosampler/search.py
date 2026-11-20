"""
Shortest pair for a specification: search over total length and the split between ``u`` and ``d``.

For every candidate ``(n_u, n_d)`` the start is a Kaiser-sinc ``u`` of ``n_u`` taps with its minimax partner of ``n_d``
taps (:func:`~prosampler.design_pair` with ``objective="minimax"``), refined by :func:`~prosampler.optimise_joint_slp`
with the passband ripple held at a reference value and, optionally, a cap on the transition gain.  The total length is
reduced stepwise from that of the plain pair until no split meets the specification any more.  Local optima: the result
is the shortest length *found*, not a proof of optimality.

Resolution
----------
Two grids control the search: the **length step** (how much the total shrinks per round; a centred linear-phase pair
needs ``n_u + n_d - 2`` to be a multiple of ``2 L`` and both lengths odd, so it cannot be finer than ``2 L`` taps) and the
**split step** (spacing of the candidate lengths of ``u`` within a round; even, at least 2).  The ``resolution`` presets
are:

========  ===========  ==========  ===============  ===========  ==========================================
preset    length step  split step  ``u`` offsets    SLP iter.    candidates per round (typ.) / use
========  ===========  ==========  ===============  ===========  ==========================================
coarse    4 L          16          0 ... +32        20           3     quick look, can miss the best split
medium    2 L          8           0 ... +40        25           6     default
fine      2 L          4           -8 ... +48       30           15    finest useful split resolution
========  ===========  ==========  ===============  ===========  ==========================================

The number of configurations is (rounds) x (candidates per round); the search stops at the first round in which no
candidate meets the specification, so it is usually 4-8 rounds (typically ~20 % length reduction).  Each configuration is one SLP run: roughly 20-45 s for
kernels of 130-300 taps in total (measured on one core, 20-25 iterations); longer kernels take longer.  So a ``"medium"``
search takes 10-20 minutes and a ``"fine"`` one 30-90 minutes on one core; ``n_jobs`` divides that by the number of
cores used.  ``progress=True`` prints one line per finished configuration with an upper bound of the remaining time.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .design import design_pair
from .errors import ProsamplerError
from .jointlp import optimise_joint_slp

# (length step in units of 2 L, split step, offsets of n_u relative to the plain pair's, SLP iterations, window)
# window: after the first round only candidates within +-window taps of the previous round's best n_u are tried
RESOLUTIONS = {
    "coarse": (2, 16, (0, 32), 20, None),
    "medium": (1, 8, (0, 40), 25, None),
    "fine": (1, 4, (-8, 48), 30, 12),
}


def _worst(fp) -> float:
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


def _fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def _run_candidate(args):
    """One configuration (module level so that it can run in a worker process)."""
    L, passband, atten_db, n_u, n_d, r_ref, cap, iterations = args
    try:
        start = design_pair(L, passband, atten_db, n_u=n_u, n_d=n_d, objective="minimax")
        fp = optimise_joint_slp(start, ripple_db=r_ref, transition_max_db=cap, iterations=iterations)
    except ProsamplerError:
        return None
    m = fp.metrics()
    worst = max(m["u_stop_db"], m["d_stop_db"])
    bump = max(m["d_transition_peak_db"], m["u_transition_peak_db"])
    ok = worst <= -atten_db and (cap is None or bump <= cap + 0.02) \
        and max(m["u_ripple_db"], m["d_ripple_db"]) <= r_ref * 1.02 + 1e-4      # (LP grid vs metrics grid)
    return fp, worst, bump, bool(ok), max(m["u_ripple_db"], m["d_ripple_db"])


def _adaptive(plain, first, total0, length_step, split_step, span, win, max_rounds, atten_db, L, run_round):
    """Secant search on the length index k (total = first - k * length_step); returns the best passing pair or the plain."""
    margin = 0.3                                          # aim this many dB below the requirement
    w = 8 if win is None else win
    lo_k, hi_k = 0, None                                  # longest known passing index (0 assumed), shortest known failing
    cache, best, ratio = {}, None, None
    kmax = max(1, int((first - 0.6 * total0) / length_step))

    def cands_for(total):
        out, grid = [], range(span[0], span[1] + 1, split_step)
        if ratio is not None:
            centre = ratio * total
            grid = [o for o in range(span[0] - 4 * split_step, span[1] + 4 * split_step + 1, split_step)
                    if abs(plain.n_u + o - centre) <= w]
        for off in grid:
            n_u, n_d = plain.n_u + off, total - plain.n_u - off
            if n_u >= 2 * L + 1 and n_d >= 2 * L + 1 and n_u % 2 == 1 and n_d % 2 == 1:
                out.append((n_u, n_d))
        return out

    def probe(k, rnd):
        nonlocal best, ratio
        total = first - k * length_step
        found, w_best, c_best = run_round(rnd, total, cands_for(total))
        if w_best is None:
            return None
        cache[k] = (w_best, found is not None)
        if found is not None:
            if best is None or total < best.n_u + best.n_d:
                best = found
            ratio = found.n_u / total
        if found is None and ratio is None:                # nothing passed yet: centre on the least bad split's ratio
            ratio = c_best[0] / total
        return cache[k]

    k = max(1, min(kmax, int(round(0.12 * total0 / length_step))))
    for rnd in range(max_rounds):
        r = probe(k, rnd)
        if r is None:
            hi_k = k if hi_k is None else min(hi_k, k)
        elif r[1]:
            lo_k = max(lo_k, k)
        else:
            hi_k = k if hi_k is None else min(hi_k, k)
        if hi_k is not None and hi_k - lo_k <= 1:
            break
        if hi_k is None and lo_k >= kmax:
            break
        # next index: secant through the two most informative points, else a step of 2
        pts = sorted((kk, v[0]) for kk, v in cache.items())
        target = -atten_db - margin
        if hi_k is not None and lo_k in cache and hi_k in cache:
            pair = ((lo_k, cache[lo_k][0]), (hi_k, cache[hi_k][0]))        # bracket ends
        else:
            pair = (pts[-2], pts[-1]) if len(pts) >= 2 else None             # the two longest-reduction probes
        nk = None
        if pair is not None:
            (k1, g1), (k2, g2) = pair
            if k2 != k1 and g2 > g1:
                nk = int(round(k1 + (target - g1) * (k2 - k1) / (g2 - g1)))
        if nk is None:
            nk = lo_k + 2 if hi_k is None else (lo_k + hi_k) // 2
        lo_lim = lo_k + 1
        hi_lim = (hi_k - 1) if hi_k is not None else kmax
        nk = max(lo_lim, min(hi_lim, nk))
        if hi_k is None:
            nk = min(nk, lo_k + 6)
        if nk in cache:
            break
        k = nk
    return best if best is not None else plain


def search_pair(L: int, passband: float, atten_db: float, *, transition_max_db=None, ripple_factor: float = 1.0,
                resolution: str = "medium", length_step=None, split_step=None, split_span=None, iterations=None,
                window="preset", max_rounds=None, start_total=None, strategy: str = "scan", n_jobs: int = 1, progress=True):
    """
    Shortest (total taps) pair meeting ``atten_db`` with passband ripple at most ``ripple_factor`` times (+2 % for the
    difference between the LP's grid and the metrics grid) that of the plain
    ``design_pair(L, passband, atten_db)`` and, if given, a transition cap ``transition_max_db`` (dB, both filters).

    Parameters
    ----------
    resolution : {"coarse", "medium", "fine"}
        Preset for the search grids (see the module docstring for steps, number of candidates and expected duration).
    length_step : int, optional
        Decrement of the total length per round, a multiple of ``2 L`` (overrides the preset).
    split_step : int, optional
        Spacing of the candidate lengths of ``u`` in a round, even (overrides the preset).
    split_span : (int, int), optional
        Range of offsets of ``u``'s length relative to the plain pair's, e.g. ``(-8, 48)`` (overrides the preset).  Larger
        ``u`` at the same total is often much better, hence the wide default ranges.
    iterations : int, optional
        SLP iterations per candidate (overrides the preset).
    window : int or None, optional
        After the first (full) round only candidates with ``|n_u - previous best n_u| <= window`` are tried (the best
        split moves slowly with the total length).  ``None``: the full span every round.  Default: the preset's value
        (``"fine"``: 12, others ``None``).  A window cuts the cost of a fine grid by 2-3x.
    max_rounds : int, optional
        Maximum number of rounds (default: until the total would drop 40 % below the plain pair's).
    start_total : int, optional
        First total length to try (default: the plain pair's).  Use it to skip rounds already known to be feasible, e.g.
        after an earlier coarser search; must differ from the plain total by a multiple of ``length_step``.
    strategy : {"scan", "adaptive"}
        ``"scan"`` (default): descend from the plain length in steps of ``length_step`` until a round fails.
        ``"adaptive"``: treat the best stopband peak at a total length as a nearly linear function of the length, probe a
        first length (full split scan), then jump by secant steps to where the line reaches the specification, keep a
        bracket (longest passing / shortest failing length) and finish when they are one ``length_step`` apart.  After
        the first probe only candidates within ``window`` taps (default 8) of the previous best split *ratio* are tried.
        Typically 2-3x fewer configurations than ``"scan"`` (the final answer is verified by the failing neighbour).
        Because outcomes scatter by a few dB between neighbours (local optima) the result can differ from ``"scan"`` by
        one length step.  ``max_rounds`` limits the number of probed lengths.
    n_jobs : int
        Number of worker processes for the candidates of a round (``-1``: all cores).
    progress : bool or callable
        ``True`` prints one line per finished configuration, e.g.
        ``[ 14 of at most 36 | 4:12 elapsed | typically ~3:10, at most 6:30 left] total 266 split 115/151: worst -61.2 dB, bump +0.50 dB, meets``
        (the search ends at the first round without a feasible candidate, typically after ~20 % length reduction, so
        "at most" is a bound and "typically" a guess from the average time per configuration so far).  A callable
        receives a dict with ``done, upper_bound, typical, elapsed, eta, eta_typical, round, total, n_u, n_d, worst_db,
        bump_db, meets``.  ``False``: silent.

    Returns the best :class:`~prosampler.FilterPair` found (``info["search"]`` has the table of all candidates tried:
    total, n_u, n_d, worst stopband peak, bump, ripple, meets).  The plain pair's lengths are the starting point, so the plain pair
    is returned when nothing shorter works.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {tuple(RESOLUTIONS)}")
    k_len, s_split, span, n_iter, win = RESOLUTIONS[resolution]
    length_step = length_step if length_step is not None else k_len * 2 * L
    split_step = split_step if split_step is not None else s_split
    span = tuple(split_span) if split_span is not None else span
    iterations = iterations if iterations is not None else n_iter
    win = win if window == "preset" else window
    if length_step % (2 * L) or length_step <= 0:
        raise ValueError("length_step must be a positive multiple of 2 L")
    if split_step % 2 or split_step <= 0:
        raise ValueError("split_step must be a positive even number")
    if strategy not in ("scan", "adaptive"):
        raise ValueError('strategy must be "scan" or "adaptive"')
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    plain = design_pair(L, passband, atten_db)
    mp = plain.metrics()
    r_ref = ripple_factor * max(mp["u_ripple_db"], mp["d_ripple_db"])
    total0 = plain.n_u + plain.n_d
    first = total0 if start_total is None else int(start_total)
    if (total0 - first) % length_step or first > total0:
        raise ValueError("start_total must be the plain total minus a multiple of length_step")
    if max_rounds is None:
        max_rounds = max(1, min(30, int((first - 0.6 * total0) / length_step)))

    def candidates(total):
        out = []
        for off in range(span[0], span[1] + 1, split_step):
            n_u, n_d = plain.n_u + off, total - plain.n_u - off
            if n_u >= 2 * L + 1 and n_d >= 2 * L + 1 and n_u % 2 == 1 and n_d % 2 == 1 and (n_u + n_d - 2) % (2 * L) == 0:
                out.append((n_u, n_d))
        return out

    rounds = [(first - k * length_step, candidates(first - k * length_step)) for k in range(max_rounds)]
    rounds = [(t, c) for t, c in rounds if c]
    n_win = len(range(-win, win + 1, split_step)) if win is not None else None     # candidates per later round (bound)
    sizes = [len(c) if (k == 0 or n_win is None) else min(len(c), n_win) for k, (_, c) in enumerate(rounds)]
    upper = sum(sizes)
    if strategy == "adaptive":
        w_ad = 8 if win is None else win
        n_w = len(range(-w_ad, w_ad + 1, split_step))
        n_first = len(candidates(first))
        upper = n_first + max(0, min(max_rounds, 8) - 1) * n_w
        sizes = [n_first] + [n_w] * 7
    n_typ = max(1, int(max(first - 0.8 * total0, 0) / length_step) + 1) if strategy == "scan" else 4          # typical: ~20 % shorter, then one failing round
    typical = sum(sizes[:n_typ])
    done, t0, tried, best = 0, time.time(), [], None
    pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None

    def emit(rnd, total, n_u, n_d, worst, bump, ok, rip=None):
        nonlocal done
        done += 1
        elapsed = time.time() - t0
        per = elapsed / done
        eta, eta_typ = per * max(upper - done, 0), per * max(typical - done, 0)
        info = dict(done=done, upper_bound=upper, typical=typical, elapsed=elapsed, eta=eta, eta_typical=eta_typ, round=rnd, total=total, n_u=n_u, n_d=n_d,
                    worst_db=worst, bump_db=bump, ripple_db=rip, meets=ok)
        if callable(progress):
            progress(info)
        elif progress:
            msg = "no result" if worst is None else \
                f"worst {worst:.1f} dB, bump {bump:+.2f} dB, ripple {rip:.4f} dB, {'meets' if ok else 'misses'}"
            print(f"[{done:3d} of at most {upper} | {_fmt_time(elapsed)} elapsed | typically ~{_fmt_time(eta_typ)}, "
                  f"at most {_fmt_time(eta)} left] total {total} split {n_u}/{n_d}: {msg}", flush=True)

    def run_round(rnd, total, cands):
        """Run the candidates of one total length; return the best passing pair (or None) and the best worst-peak."""
        args = [(L, passband, atten_db, n_u, n_d, r_ref, transition_max_db, iterations) for n_u, n_d in cands]
        if pool is None:
            results = ((c, _run_candidate(a)) for c, a in zip(cands, args))
        else:
            futs = {pool.submit(_run_candidate, a): c for c, a in zip(cands, args)}
            results = ((futs[f], f.result()) for f in as_completed(futs))
        found, best_w, best_c = None, None, None
        for (n_u, n_d), res in results:
            if res is None:
                emit(rnd, total, n_u, n_d, None, None, False)
                continue
            fp, worst, bump, ok, rip = res
            tried.append(dict(total=total, n_u=n_u, n_d=n_d, worst_db=worst, bump_db=bump, ripple_db=rip, meets=ok))
            emit(rnd, total, n_u, n_d, worst, bump, ok, rip)
            if best_w is None or worst < best_w:
                best_w, best_c = worst, (n_u, n_d)
            if ok and (found is None or worst < _worst(found)):
                found = fp
        return found, best_w, best_c

    try:
        if strategy == "scan":
            for rnd, (total, cands) in enumerate(rounds):
                if win is not None and best is not None:
                    cands = [(n_u, n_d) for n_u, n_d in cands if abs(n_u - best.n_u) <= win]
                found, _, _ = run_round(rnd, total, cands)
                if found is None:
                    break
                best = found
        else:
            best = _adaptive(plain, first, total0, length_step, split_step, span, win, max_rounds, atten_db, L, run_round)
    finally:
        if pool is not None:
            pool.shutdown()
    if best is None:
        best = plain
    best.info["search"] = dict(table=tried, ripple_ref_db=r_ref, plain_total=total0, transition_max_db=transition_max_db,
                               resolution=resolution, strategy=strategy, length_step=length_step, split_step=split_step, split_span=span,
                               iterations=iterations, window=win, start_total=first, elapsed_s=time.time() - t0)
    return best
