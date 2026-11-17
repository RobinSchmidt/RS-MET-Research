import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import prosampler as ps
from prosampler import plots


@pytest.fixture(scope="module")
def designs():
    pair = ps.design_pair(4, 0.8, 40)
    matched = ps.design_matched(4, 0.8, 40)
    return pair, matched


def test_stored_coeffs_counts_shared_matched_once(designs):
    pair, matched = designs
    assert matched.is_matched and not pair.is_matched
    assert matched.stored_coeffs == (len(ps._util.trim_zero_ends(matched.u)) + 1) // 2
    assert matched.macs_per_sample == 2 * len(ps._util.trim_zero_ends(matched.u))


def test_compare_table_and_best(designs):
    pair, matched = designs
    c = ps.compare({"pair": pair, "matched": matched})
    assert [r["name"] for r in c.rows] == ["pair", "matched"]
    txt = str(c)
    assert "pair" in txt and "matched" in txt and "stored" in txt and "MACs" in txt
    assert "|" in c.table(markdown=True)
    assert c.best()["name"] in ("pair", "matched")
    assert c.rows[1]["stored"] < c.rows[0]["stored"]            # matched stores fewer coefficients
    assert len(ps.compare([pair, matched]).rows) == 2           # sequence form, named by method


def test_plots_run(designs):
    pair, _ = designs
    ax = plots.plot_responses(pair)
    assert ax.get_xlim()[1] == 4
    axes = plots.plot_kernels(pair)
    assert len(axes) == 2
    with pytest.raises(ValueError):
        plots.plot_path(pair)
    plots.plot_path(ps.rebalance(pair))
