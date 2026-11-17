import numpy as np
import pytest
import prosampler as ps


def test_spec_driven_small():
    fp = ps.design_matched(4, 0.8, 40)
    assert fp.info["meets_spec"] and fp.u is not fp.d and np.array_equal(fp.u, fp.d)
    assert fp.n_u == 4 * fp.latency + 1
    assert fp.nyquist_error < 1e-14
    assert abs(fp.u.sum() - 2.0) < 1e-5                    # ~sqrt(L); not exact, Nyquist only fixes every L-th sample of g*g
    assert fp.u[0] == pytest.approx(0, abs=1e-12) and np.allclose(fp.u, fp.u[::-1])
    fp.verify()


def test_length_driven_and_warm_start():
    a = ps.design_matched(4, 0.8, K=20)
    b = ps.design_matched(4, 0.8, K=24, g_init=a.u)
    ea = ps.FilterPair(a.u, a.d, 4, 20, 0.8).metrics()["u_stop_db"]
    eb = b.metrics()["u_stop_db"]
    assert eb <= ea + 1e-6                                  # warm start can only improve
    assert b.nyquist_error < 1e-14


def test_errors_and_warnings():
    with pytest.raises(ValueError):
        ps.design_matched(3, 0.8, K=5)                      # L*K odd
    with pytest.raises(ValueError):
        ps.design_matched(4, 0.8)                           # neither K nor atten_db
    with pytest.warns(UserWarning):
        ps.design_matched(2, 0.8, K=20)                     # L=2 has no freedom
    with pytest.raises(ps.SpecNotReached):
        ps.design_matched(4, 0.8, 90, K=10, strict=True)


def test_betas_multistart_not_worse_than_single():
    one = ps.design_matched(4, 0.8, K=24, betas=(4.0,))
    many = ps.design_matched(4, 0.8, K=24, betas=(2.0, 4.0, 6.0))
    assert many.metrics()["u_stop_db"] <= one.metrics()["u_stop_db"] + 1e-9
    assert many.nyquist_error < 1e-14
