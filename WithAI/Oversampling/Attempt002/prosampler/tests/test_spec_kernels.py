import numpy as np
import pytest

from prosampler import (Spec, SpecNotReached, design_kaiser_sinc, kaiser_length_estimate, kaiser_sinc,
                        passband_from_hz, stopband_db, windowed_sinc)


def test_spec_edges():
    s = Spec(4, 0.9, 80)
    assert s.stopband == pytest.approx(1.1)                      # fraction of the original Nyquist frequency
    assert s.passband_rad == pytest.approx(0.9 * np.pi / 4)
    assert s.stopband_rad == pytest.approx(1.1 * np.pi / 4)
    assert s.transition_width == pytest.approx(0.2 * np.pi / 4)


def test_grid_masks_partition():
    g = Spec(3, 0.8).grid(1000)
    total = g.passband.astype(int) + g.stopband.astype(int) + g.transition.astype(int)
    assert np.all(total == 1)
    assert g.w[0] == 0 and g.w[-1] == pytest.approx(np.pi)


@pytest.mark.parametrize("kwargs", [dict(L=1, passband=0.9), dict(L=2.5, passband=0.9), dict(L=4, passband=1.0),
                                    dict(L=4, passband=0.0), dict(L=4, passband=0.9, atten_db=-3)])
def test_spec_validation(kwargs):
    with pytest.raises(ValueError):
        Spec(**kwargs)


def test_passband_from_hz():
    assert passband_from_hz(19800, 44100) == pytest.approx(19800 / 22050)
    with pytest.raises(ValueError):
        passband_from_hz(23000, 44100)


@pytest.mark.parametrize("window", ["hann", ("kaiser", 8.0), lambda n: np.hamming(n)])
def test_windowed_sinc_symmetric_with_gain_L(window):
    u = windowed_sinc(4, 41, window)
    assert np.allclose(u, u[::-1])
    assert u.sum() == pytest.approx(4.0)


def test_windowed_sinc_custom_gain_and_array_window():
    u = windowed_sinc(3, 31, np.blackman(31), gain=1.0)
    assert u.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        windowed_sinc(3, 31, np.blackman(30))


def test_windowed_sinc_has_nyquist_zeros():
    u = windowed_sinc(4, 41, "hann")          # centre index 20 = multiple of L
    k = np.arange(-5, 6)
    k = k[k != 0]
    assert np.max(np.abs(u[20 + 4 * k])) < 1e-15


def test_kaiser_sinc_argument_check():
    with pytest.raises(ValueError):
        kaiser_sinc(4, 41)
    with pytest.raises(ValueError):
        kaiser_sinc(4, 41, atten_db=80, beta=8)
    assert kaiser_sinc(4, 41, atten_db=80).sum() == pytest.approx(4.0)


def test_design_kaiser_sinc_meets_spec_and_is_shortest():
    spec = Spec(4, 0.9, 80)
    u = design_kaiser_sinc(spec)
    assert len(u) % 2 == 1 and np.allclose(u, u[::-1])
    assert u.sum() == pytest.approx(4.0)
    assert stopband_db(u, spec, gain=4.0) <= -80
    est = kaiser_length_estimate(spec)
    assert len(u) >= 0.75 * est
    shorter = kaiser_sinc(4, len(u) - 2, atten_db=80)
    assert stopband_db(shorter, spec, gain=4.0) > -80


def test_design_kaiser_sinc_failure_and_missing_spec():
    with pytest.raises(SpecNotReached):
        design_kaiser_sinc(Spec(4, 0.95, 100), max_factor=0.71)
    with pytest.raises(ValueError):
        design_kaiser_sinc(Spec(4, 0.9))
