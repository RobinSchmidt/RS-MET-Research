"""Minimax (LP) partner design and the alternating joint minimax refinement."""
import numpy as np
import pytest

import prosampler as ps

L, P, ATT = 4, 0.8, 40.0


@pytest.fixture(scope="module")
def base():
    return ps.design_pair(L, P, ATT)                      # 47 / 115 taps, grid least-squares partner


def _worst(fp):
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


def test_minimax_partner_is_exact_and_better_than_grid(base):
    spec = ps.Spec(L, P, ATT)
    d = ps.design_partner(base.u, spec, base.n_d, base.latency, objective="minimax")
    fp = ps.FilterPair(base.u, d, L, base.latency, P)
    assert fp.nyquist_error < 1e-13
    assert np.allclose(d, d[::-1])
    assert fp.metrics()["d_stop_db"] <= base.metrics()["d_stop_db"] + 0.01      # global optimum of the peak
    fp.verify()


def test_minimax_partner_is_equiripple(base):
    spec = ps.Spec(L, P, ATT)
    d = ps.design_partner(base.u, spec, base.n_d, base.latency, objective="minimax")
    g = spec.grid(8000)
    mag = np.abs(ps.freqz(d, g.w[g.stopband]))
    peaks = mag[1:-1][(mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:])]
    top = peaks[peaks > 0.98 * mag.max()]
    assert len(top) >= 5                                                       # many ripples touch the maximum


def test_minimax_transition_bound(base):
    spec = ps.Spec(L, P, ATT)
    d = ps.design_partner(base.u, spec, base.n_d, base.latency, objective="minimax", transition_max_db=1.0)
    m = ps.FilterPair(base.u, d, L, base.latency, P).metrics()
    assert m["d_transition_peak_db"] <= 1.0 + 0.01
    with pytest.raises(ps.ProsamplerError):
        ps.design_partner(base.u, spec, base.n_d, base.latency, objective="minimax", transition_max_db=-0.5)


def test_minimax_needs_symmetric_centred(base):
    spec = ps.Spec(L, P, ATT)
    with pytest.raises(ps.ProsamplerError):
        ps.design_partner(base.u, spec, base.n_d, base.latency + 1, objective="minimax")


def test_design_pair_minimax_meets_spec_with_shorter_d(base):
    fp = ps.design_pair(L, P, ATT, objective="minimax")
    assert fp.info["meets_spec"] is True
    assert fp.n_u == base.n_u and fp.n_d <= base.n_d
    assert fp.nyquist_error < 1e-13


def test_joint_minimax_monotone_exact_and_flat(base):
    j = ps.optimise_joint_minimax(base)
    h = j.info["joint_minimax"]["history"]
    assert all(max(b) <= max(a) + 1e-6 for a, b in zip(h, h[1:]))               # worst peak never increases
    assert all(b[0] <= a[0] + 1e-6 and b[1] <= a[1] + 1e-6 for a, b in zip(h, h[1:]))
    assert (j.n_u, j.n_d, j.latency) == (base.n_u, base.n_d, base.latency)
    assert j.nyquist_error < 1e-13
    assert abs(j.u.sum() - L) < 1e-9
    assert max(j.metrics()["u_ripple_db"], j.metrics()["d_ripple_db"]) <= j.info["joint_minimax"]["ripple_db"] + 1e-3
    assert j.metrics()["d_stop_db"] < base.metrics()["d_stop_db"] - 5            # d profits a lot at 47/115
    j.verify()
