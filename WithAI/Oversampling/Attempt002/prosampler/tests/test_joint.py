import numpy as np
import pytest
import prosampler as ps


@pytest.fixture(scope="module")
def pair():
    return ps.design_pair(4, 0.8, 40)


def worst(fp):
    m = fp.metrics()
    return max(m["u_stop_db"], m["d_stop_db"])


def test_optimise_joint_exact_and_lower_objective(pair):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # the peak may legitimately get worse (documented)
        j = ps.optimise_joint(pair)
    assert (j.n_u, j.n_d, j.latency) == (pair.n_u, pair.n_d, pair.latency)
    assert j.nyquist_error < 1e-13
    assert abs(j.u.sum() - 4) < 1e-12
    info = j.info["joint"]
    assert info["objective_ratio"] < 1.0                 # the optimised (energy) objective drops
    assert max(info["ripple_db_after"]) < 0.5            # passband term keeps ripple small (about 1-6 dB without it)
    j.verify()


def test_optimise_joint_warns_or_keeps_when_peak_worse(pair):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        j = ps.optimise_joint(pair)
    worse = worst(j) > worst(pair) + 1e-9
    assert worse == any("worse" in str(x.message) for x in w)
    k = ps.optimise_joint(pair, on_worse_peak="keep")
    assert worst(k) <= worst(pair) + 1e-9


def test_rebalance_equal_and_best(pair):
    r = ps.rebalance(pair)
    assert r.n_u == r.n_d == (pair.n_u + pair.n_d) // 2
    assert r.nyquist_error < 1e-13
    path = r.info["path"]
    assert path[0][:2] == (pair.n_u, pair.n_d) and path[-1][:2] == (r.n_u, r.n_d)
    b = ps.rebalance(pair, target="best")
    assert worst(b) <= worst(r) + 1e-9 and b.nyquist_error < 1e-13
    r.verify()


def test_tie_gives_exact_matched_pair(pair):
    t = ps.tie(ps.rebalance(pair))
    assert np.array_equal(t.u, t.d) and t.nyquist_error < 1e-14
    assert t.info["refined_stop_db"] < -20
    t.verify()


def test_input_validation(pair):
    with pytest.raises(ps.ProsamplerError):
        ps.tie(pair)                                        # unequal lengths
    nonsym = ps.FilterPair(pair.u, np.roll(pair.d, 1), 4, pair.latency, 0.8)
    with pytest.raises(ps.ProsamplerError):
        ps.optimise_joint(nonsym)
    with pytest.raises(ps.ProsamplerError):
        ps.optimise_joint(ps.FilterPair(pair.u, pair.d, 4, pair.latency))   # no passband
    with pytest.raises(ValueError):
        ps.rebalance(pair, target="x")
    with pytest.raises(ValueError):
        ps.rebalance(pair, step=3)
