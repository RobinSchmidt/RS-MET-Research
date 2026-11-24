"""Tests of the matched minimax SLP and of the shortest-length search for a matched kernel."""
import warnings

import numpy as np
import pytest

import prosampler as ps

L, P = 4, 0.8


@pytest.fixture(scope="module")
def start():
    K = 16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ps.optimise_matched(L, K, P, 4.0)
    return ps.FilterPair(g, g, L, K, P, method="matched[from-scratch]", info={})


def test_matched_slp_exact_and_better(start):
    m0 = start.metrics()
    fp = ps.optimise_matched_slp(start, iterations=25)
    m = fp.metrics()
    assert fp.is_matched and m["nyquist_error"] < 1e-13
    assert m["u_stop_db"] < m0["u_stop_db"] - 3                       # the minimax step improves the energy-optimal start
    assert m["u_ripple_db"] <= 0.1 * 1.02 + 1e-4                     # ripple bound (default 0.1 dB)
    assert m["u_transition_peak_db"] <= 0.05 + 0.02                   # bump cap (default ripple / 2)
    assert fp.info["slp"]["transition_max_db"] == pytest.approx(0.05)


def test_matched_slp_rejects_pairs_and_bad_cap(start):
    pair = ps.design_pair(L, P, 40)
    with pytest.raises(ps.ProsamplerError):
        ps.optimise_matched_slp(pair)
    with pytest.raises(ValueError):
        ps.optimise_matched_slp(start, transition_max_db="bump")


def test_search_matched_finds_shortest_found_length():
    ev = []
    fp = ps.search_matched(L, P, 36, progress=ev.append, iterations=20)
    m = fp.metrics()
    assert fp.is_matched and m["u_stop_db"] <= -36 and m["nyquist_error"] < 1e-13
    s = fp.info["search_matched"]
    assert len(s["table"]) == len(ev) and s["ripple_db"] == pytest.approx(0.1)
    ks = {r["K"]: r["meets"] for r in s["table"]}
    assert ks[fp.latency] and not ks.get(fp.latency - 1, False)       # the next shorter length (if probed) misses


def test_search_matched_argument_checks():
    with pytest.raises(ValueError):
        ps.search_matched(L, P, 36, ripple_db=None, progress=False)
    with pytest.raises(ValueError):
        ps.search_matched(L, P, 36, transition_max_db="bump", progress=False)
