import warnings
import numpy as np
import pytest
import prosampler as ps


@pytest.fixture(scope="module")
def ref():
    return ps.kaiser_sinc(4, 261, atten_db=100), ps.Spec(4, 0.9, 100)


@pytest.mark.parametrize("method", ["nullspace", "kkt"])
@pytest.mark.parametrize("objective", ["grid", "energy"])
def test_exact_all_combinations(ref, method, objective):
    u, spec = ref
    d = ps.design_partner(u, spec, 541, 100, method=method, objective=objective)
    assert ps.nyquist_error(u, d, 4, 100) < 1e-12
    assert np.allclose(d, d[::-1])
    ps.verify_roundtrip(u, d, 4, 100)


def test_regression_reference_case(ref):
    """L=4, passband 0.9, 261-tap Kaiser-sinc u (100 dB), 541-tap d, latency 100.

    Measured with the exact stopband edge on the grid.  (Before the edge was added to the grid the energy design
    looked 2 dB better, -100.4 dB, because the grid missed the edge peak.)"""
    u, spec = ref
    fp = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100, objective="energy", method="kkt"), 4, 100, 0.9)
    m = fp.metrics()
    assert m["u_stop_db"] < -99.5 and -99.5 < m["d_stop_db"] < -97.5
    fp = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100), 4, 100, 0.9)
    m = fp.metrics()
    assert m["d_stop_db"] < -103.5 and m["d_transition_peak_db"] < 1.5


def test_transition_weight_is_not_a_bump_remedy(ref):
    """Documented finding: transition_weight > 0 makes the bump of d larger and the stopband much worse."""
    u, spec = ref
    base = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100), 4, 100, 0.9).metrics()
    tw = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100, transition_weight=0.1), 4, 100, 0.9).metrics()
    assert tw["d_stop_db"] > base["d_stop_db"] + 20
    assert tw["d_transition_peak_db"] >= base["d_transition_peak_db"]


def test_methods_agree(ref):
    u, spec = ref
    a = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100, objective="energy"), 4, 100, 0.9).metrics()
    b = ps.FilterPair(u, ps.design_partner(u, spec, 541, 100, objective="energy", method="kkt"), 4, 100, 0.9).metrics()
    assert abs(a["d_stop_db"] - b["d_stop_db"]) < 1.0


def test_nonsymmetric_u_explicit_latency():
    rng = np.random.default_rng(1)
    u = ps.kaiser_sinc(3, 31, atten_db=60) * (1 + 0.0 * rng.standard_normal(31))
    u = np.roll(u, 1); u[0] = 0.0; u[1:] = u[1:]            # break symmetry
    u = u * (3 / u.sum()); u[2] += 0.01; u = u * (3 / u.sum())
    d = ps.design_partner(u, ps.Spec(3, 0.8), 90, latency=22)
    assert ps.nyquist_error(u, d, 3, 22) < 1e-10


def test_latency_required_for_nonsymmetric():
    u = np.array([1.0, 2.0, 0.5, 0.3, 0.1])
    with pytest.raises(ValueError):
        ps.design_partner(u, ps.Spec(2, 0.5), 20)


def test_symmetric_true_requires_centred(ref):
    u, spec = ref
    with pytest.raises(ValueError):
        ps.design_partner(u, spec, 541, 101, symmetric=True)


def test_short_partner_is_exact_but_misses_spec(ref):
    # A windowed-sinc u is (nearly) Nyquist(L) itself, so d ~ delta is an exact partner: exactness alone proves
    # nothing about quality; the spec check is what catches a too-short d.
    u, spec = ref
    with pytest.warns(UserWarning):
        d = ps.design_partner(u, spec, 61)
    assert ps.nyquist_error(u, d, 4, 40) < 1e-10
    assert ps.FilterPair(u, d, 4, 40, 0.9).metrics()["d_stop_db"] > -20
    with pytest.raises(ps.SpecNotReached):
        ps.design_pair(4, 0.9, 100, u=u, n_d=61, strict=True)


def test_bad_options(ref):
    u, spec = ref
    with pytest.raises(ValueError):
        ps.design_partner(u, spec, 541, 100, method="foo")
    with pytest.raises(ValueError):
        ps.design_partner(u, spec, 541, 100, objective="foo")


def test_design_pair_spec_driven():
    fp = ps.design_pair(4, 0.85, 60)
    assert fp.info["meets_spec"]
    assert 1.8 <= fp.n_d / fp.n_u <= 2.5
    assert fp.nyquist_error < 1e-12
    fp.verify()


def test_design_pair_length_driven():
    fp = ps.design_pair(4, 0.85, 60, n_u=99, n_d=215)
    assert fp.latency == 39 and fp.nyquist_error < 1e-12


def test_design_pair_external_u_rescaled():
    u = 7.0 * ps.kaiser_sinc(4, 99, atten_db=60)
    fp = ps.design_pair(4, 0.85, 55, u=u, n_d=215)
    assert abs(fp.u.sum() - 4) < 1e-12 and fp.info["u_rescaled"]
    fp.verify()


def test_design_pair_strict_vs_warn():
    with pytest.raises(ps.SpecNotReached):
        ps.design_pair(4, 0.85, 120, n_u=99, n_d=215, strict=True)
    with pytest.warns(UserWarning):
        fp = ps.design_pair(4, 0.85, 120, n_u=99, n_d=215)
    assert fp.info["meets_spec"] is False


def test_design_pair_bad_u_spec_driven():
    with pytest.raises(ps.SpecNotReached):
        ps.design_pair(4, 0.85, 100, u=ps.kaiser_sinc(4, 41, atten_db=40))


def test_design_pair_argument_errors():
    with pytest.raises(ValueError):
        ps.design_pair(4, 0.9)
    with pytest.raises(ValueError):
        ps.design_pair(4, 0.9, 60, n_d=215)
