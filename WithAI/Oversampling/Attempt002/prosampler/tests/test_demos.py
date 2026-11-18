"""Smoke tests for the scripts in demos/ (small, fast parameters; plots rendered with the Agg backend)."""
import importlib.util
import pathlib

import matplotlib
matplotlib.use("Agg")
import pytest

DEMO = pathlib.Path(__file__).resolve().parents[1] / "demos" / "design_and_plot.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("design_and_plot", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", ["pair", "matched"])
def test_demo_runs_and_saves_figures(demo, tmp_path, kind, capsys):
    prefix = str(tmp_path / "fig")
    fp = demo.main(["--no-show", "--save", prefix, "--design", kind, "--atten-db", "40", "--passband-hz", "16000"])
    assert fp.nyquist_error < 1e-12
    assert (tmp_path / "fig.png").stat().st_size > 10_000
    assert (tmp_path / "fig_detail.png").stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "identity roundtrip" in out and "Nyquist error" in out
    assert (kind == "matched") == fp.is_matched


def test_demo_design_function_accepts_overrides(demo):
    fp = demo.design(44100.0, 3, 15000.0, 40.0, "pair", method="kkt", objective="energy")
    assert fp.L == 3 and fp.nyquist_error < 1e-12
