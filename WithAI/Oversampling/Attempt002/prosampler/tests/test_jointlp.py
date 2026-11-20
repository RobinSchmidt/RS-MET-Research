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
    best = ps.search_pair(L, P, ATT, transition_max_db=0.5, n_u_offsets=(0, 16), iterations=8, max_steps=2)
    s = best.info["search"]
    assert best.n_u + best.n_d <= plain.n_u + plain.n_d
    assert s["table"] and all({"total", "n_u", "n_d", "worst_db", "bump_db", "meets"} <= set(r) for r in s["table"])
    m = best.metrics()
    assert max(m["u_stop_db"], m["d_stop_db"]) <= -ATT
    assert m["nyquist_error"] < 1e-13
    assert max(m["u_ripple_db"], m["d_ripple_db"]) <= s["ripple_ref_db"] * (1 + 1e-3) + 1e-6
    best.verify()
