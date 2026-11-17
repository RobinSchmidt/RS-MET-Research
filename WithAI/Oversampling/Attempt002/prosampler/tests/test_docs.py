"""Executes every python code block of docs/usage.md in one shared namespace."""
import pathlib
import re

import matplotlib
matplotlib.use("Agg")

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "usage.md"


def test_usage_examples_run():
    text = DOC.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) >= 10
    ns = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"usage.md block {i}", "exec"), ns)
        except Exception as e:                      # pragma: no cover
            raise AssertionError(f"block {i} failed: {e!r}\n{code}") from e


def test_theory_frequency_domain_identity():
    """docs/theory.md: sum_j H(w - 2 pi j / L) = L exp(-i w n0) for an exact pair."""
    import numpy as np
    import prosampler as ps
    fp = ps.design_pair(4, 0.8, 40)
    h = np.convolve(fp.u, fp.d)
    L, n0 = 4, 4 * fp.latency
    w = np.linspace(0.1, 3.0, 7)
    H = lambda ww: np.exp(-1j * np.outer(np.atleast_1d(ww), np.arange(len(h)))) @ h
    lhs = sum(H(w - 2 * np.pi * j / L) for j in range(L))
    assert np.max(np.abs(lhs - L * np.exp(-1j * w * n0))) < 1e-9
