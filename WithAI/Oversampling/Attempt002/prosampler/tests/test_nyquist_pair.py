import numpy as np
import pytest

from prosampler import (FilterPair, Spec, VerificationError, nyquist_error, partner_length, reference_roundtrip,
                        symmetric_latency, verify_roundtrip, windowed_sinc)

L = 4


def zoh_pair():
    """Zero-order hold: u = [1]*L (gain L), d = [1]; an exact PR pair with latency 0."""
    return np.ones(L), np.array([1.0])


def linear_interp_pair():
    """Linear interpolation kernel (triangle, gain L) padded to centre L, d = [1]; latency 1."""
    n = np.arange(-(L - 1), L)
    tri = 1.0 - np.abs(n) / L
    return np.concatenate([[0.0], tri]), np.array([1.0])


def sinc_scalar_pair(m=5):
    """Windowed sinc u (centre at L*m) with scalar partner d = 1/u[centre]; exact because sinc has Nyquist zeros."""
    N = 2 * L * m + 1
    u = windowed_sinc(L, N, "hann")
    return u, np.array([1.0 / u[(N - 1) // 2]]), m


def test_known_pairs_are_exact():
    for (u, d), lat in [(zoh_pair(), 0), (linear_interp_pair(), 1)]:
        assert nyquist_error(u, d, L, lat) == 0.0
        assert verify_roundtrip(u, d, L, lat, tol=1e-12) < 1e-12
    u, d, m = sinc_scalar_pair()
    assert nyquist_error(u, d, L, m) < 1e-15
    assert verify_roundtrip(u, d, L, m, tol=1e-12) < 1e-12


def test_wrong_latency_is_detected():
    u, d = linear_interp_pair()
    assert nyquist_error(u, d, L, 0) > 0.1
    assert nyquist_error(u, d, L, 99) == 1.0           # centre outside of u*d


def test_non_pr_pair_fails_verification():
    u, d = np.ones(L), np.array([0.5])
    assert nyquist_error(u, d, L, 0) == pytest.approx(0.5)
    with pytest.raises(VerificationError):
        verify_roundtrip(u, d, L, 0)
    assert verify_roundtrip(u, d, L, 0, raise_on_fail=False) > 0.1


def test_symmetric_latency_and_partner_length():
    assert symmetric_latency(5, 9, 2) == 3
    with pytest.raises(ValueError, match="nearest valid n_d"):
        symmetric_latency(5, 7, 2)
    with pytest.raises(ValueError):
        symmetric_latency(4, 9, 2)
    assert partner_length(5, 2, 3) == 9
    assert partner_length(261, 4, 100) == 541           # the 261/541 pair designed earlier in this project


def test_reference_roundtrip_applies_shaper_between_filters():
    u, d = zoh_pair()
    x = np.array([0.5, -1.0, 2.0, 0.0, 1.0])
    y = reference_roundtrip(x, u, d, L, 0, shaper=lambda v: 2.0 * v)
    assert np.allclose(y, 2.0 * x)
    with pytest.raises(ValueError):
        reference_roundtrip(x, u, d, L, 10)


def test_filterpair_basics_and_storage():
    u, d = zoh_pair()
    fp = FilterPair(u, d, L, 0)
    assert fp.nyquist_error == 0.0
    assert fp.n_u == 4 and fp.n_d == 1
    assert fp.stored_coeffs == 2 + 1                     # symmetric u -> 2, d -> 1
    assert fp.macs_per_sample == 5
    fp.verify()
    padded = FilterPair(np.array([0, 1, 2, 3, 2, 1, 0.0]), np.array([1.0, 2.0, 5.0]), L, 1)
    assert padded.stored_coeffs == 3 + 3                 # zero end taps dropped, symmetry halves the rest
    assert FilterPair(np.array([1.0, 2.0, 4.0]), np.array([1.0]), L, 0).stored_coeffs == 3 + 1   # not symmetric


def test_swapped_keeps_product_and_gain_convention():
    u, d = zoh_pair()
    fp = FilterPair(u, d, L, 0)
    sw = fp.swapped()
    assert sw.u.sum() == pytest.approx(L) and sw.d.sum() == pytest.approx(1.0)
    assert np.allclose(np.convolve(sw.u, sw.d), np.convolve(fp.u, fp.d))
    assert sw.nyquist_error < 1e-15
    sw.verify()
    back = sw.swapped()
    assert np.allclose(back.u, fp.u) and np.allclose(back.d, fp.d)
    with pytest.raises(ValueError):
        FilterPair(u, np.array([1.0, -1.0]), L, 0).swapped()


def test_swapped_differs_under_nonlinearity():
    u, d, m = sinc_scalar_pair()
    fp = FilterPair(u, d, L, m)
    sw = fp.swapped()
    x = np.sin(0.05 * np.arange(300)) * 2.0
    y1 = reference_roundtrip(x, fp.u, fp.d, L, m, shaper=np.tanh)
    y2 = reference_roundtrip(x, sw.u, sw.d, L, m, shaper=np.tanh)
    assert np.max(np.abs(y1 - y2)) > 1e-6                # the order matters once there is a nonlinearity


def test_metrics_needs_passband_and_reports_keys():
    u, d, m = sinc_scalar_pair()
    with pytest.raises(ValueError):
        FilterPair(u, d, L, m).metrics()
    mt = FilterPair(u, d, L, m, passband=0.7).metrics()
    assert {"u_stop_db", "d_stop_db", "u_ripple_db", "nyquist_error"} <= set(mt)
    assert mt["nyquist_error"] < 1e-15


def test_filterpair_validation():
    with pytest.raises(ValueError):
        FilterPair(np.ones((2, 2)), np.ones(3), L, 0)
    with pytest.raises(ValueError):
        FilterPair(np.ones(3), np.ones(3), 1, 0)
    with pytest.raises(ValueError):
        FilterPair(np.ones(3), np.ones(3), L, -1)


def test_stopband_measurement_independent_of_grid_size():
    import prosampler as ps
    fp = ps.design_pair(4, 0.8, 40)
    vals = [fp.metrics(n_grid=n)["u_stop_db"] for n in (500, 1000, 4000, 9000)]
    assert max(vals) - min(vals) < 0.05          # the exact band edges are always on the grid
    g = ps.Spec(4, 0.8).grid(777)
    assert np.any(g.w == ps.Spec(4, 0.8).stopband_edge) and np.any(g.w == ps.Spec(4, 0.8).passband_edge)


def test_level_compensation_makes_pair_and_matched_equivalent_for_nonlinearity():
    import prosampler as ps
    m = ps.design_matched(4, 0.8, K=20)
    p = ps.design_pair(4, 0.8, 40)
    assert abs(p.shaper_level - 1) < 1e-9 and abs(m.shaper_level - 0.5) < 1e-5
    x = 0.5 * np.sin(2 * np.pi * 0.01 * np.arange(400))
    shaper = lambda v: np.tanh(3 * v)
    y_id = ps.reference_roundtrip(x, m.u, m.d, 4, m.latency, compensate_levels=True)
    assert np.max(np.abs(y_id - x)) < 1e-12
    y_m = ps.reference_roundtrip(x, m.u, m.d, 4, m.latency, shaper=shaper, compensate_levels=True)
    y_p = ps.reference_roundtrip(x, p.u, p.d, 4, p.latency, shaper=shaper, compensate_levels=True)
    assert np.max(np.abs(y_m - y_p)) < 0.02       # same nonlinearity, same level -> nearly the same result
    y_raw = ps.reference_roundtrip(x, m.u, m.d, 4, m.latency, shaper=shaper)
    assert np.max(np.abs(y_raw - y_p)) > 0.1      # without compensation the matched chain differs clearly
