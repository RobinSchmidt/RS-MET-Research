"""Transition-shape families and the simultaneous joint minimax (sequential LP)."""
import numpy as np
import pytest

import prosampler as ps

L, P, ATT = 4, 0.8, 40.0


@pytest.fixture(scope="module")
def base():
    return ps.design_pair(L, P, ATT)                      # 47 / 115 taps


@pytest.fixture(scope="module")
def r0(base):
    m = base.metrics()
    return max(m["u_ripple_db"], m["d_ripple_db"])


def _worst(fp):
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


@pytest.mark.parametrize("kind", ps.transition.KINDS)
def test_product_shape_is_point_symmetric(kind):
    sh = ps.TransitionShape(kind)
    x = np.linspace(0, 1, 201)
    r = sh.product(x)
    assert abs(r[0] - 1) < 1e-9 and abs(r[-1]) < 1e-9
    assert np.max(np.abs(r + r[::-1] - 1)) < 1e-3          # r(x) + r(1 - x) = 1  (Nyquist symmetry)
    assert np.all(np.diff(r) <= 1e-12)                      # monotone


def test_split_exponents_multiply_to_the_product():
    sh = ps.TransitionShape("cosine", alpha=0.3)
    x = np.linspace(0, 1, 50)
    assert np.allclose(sh.u_target(x) * sh.d_target(x), sh.product(x))
    with pytest.raises(ValueError):
        ps.TransitionShape("cosine", alpha=1.5)
    with pytest.raises(ValueError):
        ps.TransitionShape("bogus")


def test_slp_improves_peak_at_equal_ripple_and_stays_exact(base, r0):
    j = ps.optimise_joint_slp(base, iterations=12, ripple_db=r0)
    m = j.metrics()
    assert (j.n_u, j.n_d, j.latency) == (base.n_u, base.n_d, base.latency)
    assert m["nyquist_error"] < 1e-13
    assert abs(j.u.sum() - L) < 1e-9
    assert max(m["u_ripple_db"], m["d_ripple_db"]) <= r0 * (1 + 1e-3) + 1e-6          # ripple not traded
    assert _worst(j) < _worst(base) - 2.0                                              # real gain, same lengths
    h = j.info["slp"]["history"]
    assert all(max(b[:2]) <= max(a[:2]) + 1e-6 for a, b in zip(h, h[1:]))               # accepted steps never worsen
    j.verify()


def test_slp_transition_cap_removes_most_of_the_bump(base, r0):
    j = ps.optimise_joint_slp(base, iterations=25, ripple_db=r0, transition_max_db=0.5)
    m = j.metrics()
    assert m["d_transition_peak_db"] <= 0.5 + 0.05 and m["u_transition_peak_db"] <= 0.5 + 0.05
    assert _worst(j) < _worst(base) + 0.5                                               # cap costs at most ~ nothing here
    assert m["nyquist_error"] < 1e-13


def test_search_pair_returns_pair_meeting_spec_and_table():
    plain = ps.design_pair(L, P, ATT)
    events = []
    best = ps.search_pair(L, P, ATT, transition_max_db=0.5, split_span=(0, 16), split_step=16, iterations=8, max_rounds=2,
                          progress=events.append)
    assert events and events[-1]["done"] == len(events) and events[0]["upper_bound"] >= len(events)
    assert all(e["eta"] >= 0 and e["eta_typical"] >= 0 for e in events)
    s = best.info["search"]
    assert best.n_u + best.n_d <= plain.n_u + plain.n_d
    assert s["table"] and all({"total", "n_u", "n_d", "worst_db", "bump_db", "meets"} <= set(r) for r in s["table"])
    m = best.metrics()
    assert max(m["u_stop_db"], m["d_stop_db"]) <= -ATT
    assert m["nyquist_error"] < 1e-13
    assert max(m["u_ripple_db"], m["d_ripple_db"]) <= s["ripple_ref_db"] * 1.02 + 1e-4
    best.verify()


def test_search_pair_argument_checks():
    with pytest.raises(ValueError):
        ps.search_pair(L, P, ATT, resolution="ultra", progress=False)
    with pytest.raises(ValueError):
        ps.search_pair(L, P, ATT, length_step=4, progress=False)          # not a multiple of 2 L
    with pytest.raises(ValueError):
        ps.search_pair(L, P, ATT, split_step=3, progress=False)


def test_slp_ripple_bound_is_elastic(base, r0):
    j = ps.optimise_joint_slp(base, iterations=20, ripple_db=0.6 * r0)   # start violates the bound: driven down
    m = j.metrics()
    assert max(m["u_ripple_db"], m["d_ripple_db"]) < r0 * 0.8
    assert m["nyquist_error"] < 1e-13


def test_search_pair_adaptive_runs_and_meets():
    ev = []
    fp = ps.search_pair(L, P, ATT, strategy="adaptive", resolution="coarse", progress=ev.append)
    m = fp.metrics()
    assert max(m["u_stop_db"], m["d_stop_db"]) <= -ATT + 1e-6
    assert m["nyquist_error"] < 1e-13
    s = fp.info["search"]
    assert s["strategy"] == "adaptive" and ev and len(s["table"]) <= len(ev)


def test_search_pair_rejects_bad_strategy():
    with pytest.raises(ValueError):
        ps.search_pair(L, P, ATT, strategy="bisect", progress=False)


def test_search_pair_rescue_starts_are_tried_and_recorded():
    fp = ps.search_pair(L, P, ATT, resolution="coarse", rescue=(-6.0, 6.0), rescue_top=2, progress=False)
    s = fp.info["search"]
    assert s["rescue"] == (-6.0, 6.0) and s["rescue_top"] == 2
    assert all("start_db" in r for r in s["table"])
    assert any(r["start_db"] != 0.0 for r in s["table"])               # the scan ends in a failing round, which is rescued
    m = fp.metrics()
    assert max(m["u_stop_db"], m["d_stop_db"]) <= -ATT + 1e-6


def test_transition_cap_ripple_option(base, r0):
    j = ps.optimise_joint_slp(base, iterations=10, transition_max_db="ripple", ripple_db=2 * r0)
    assert j.info["slp"]["transition_max_db"] == pytest.approx(r0)             # cap = ripple_db / 2
    with pytest.raises(ValueError):
        ps.optimise_joint_slp(base, iterations=1, transition_max_db="bump")
    with pytest.raises(ValueError):
        ps.search_pair(L, P, ATT, transition_max_db="bump", progress=False)


def test_best_pair_for_length():
    ev = []
    fp = ps.best_pair_for_length(L, P, 74, ripple_db=0.05, atten_db=ATT, transition_max_db="ripple", split_step=8,
                                 iterations=10, progress=ev.append)
    b = fp.info["best_for_length"]
    assert fp.n_u + fp.n_d == 74 and ev and len(b["table"]) == len(ev)
    assert b["transition_max_db"] == pytest.approx(0.025)
    m = fp.metrics()
    assert m["nyquist_error"] < 1e-13
    assert max(m["u_stop_db"], m["d_stop_db"]) == pytest.approx(min(r["worst_db"] for r in b["table"]))
    with pytest.raises(ValueError):
        ps.best_pair_for_length(L, P, 75, progress=False)
    with pytest.raises(ValueError):
        ps.best_pair_for_length(L, P, 74, split_step=3, progress=False)
