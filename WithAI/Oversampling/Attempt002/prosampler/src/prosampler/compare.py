"""Side-by-side comparison of designs: quality versus memory and multiplies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Union

from .pair import FilterPair

_COLUMNS = [
    ("name", "name", "{}"),
    ("L", "L", "{:d}"),
    ("n_u", "n_u", "{:d}"),
    ("n_d", "n_d", "{:d}"),
    ("latency", "lat", "{:d}"),
    ("stored", "stored", "{:d}"),
    ("macs", "MACs", "{:d}"),
    ("u_stop_db", "u stop", "{:.1f}"),
    ("d_stop_db", "d stop", "{:.1f}"),
    ("ripple_db", "ripple", "{:.4f}"),
    ("nyquist_error", "nyq err", "{:.0e}"),
]


@dataclass
class Comparison:
    """Rows of figures for several :class:`FilterPair` objects; ``str(c)`` / :meth:`table` renders them as text."""

    rows: List[Dict]

    def table(self, markdown: bool = False) -> str:
        cells = [[hdr for _, hdr, _ in _COLUMNS]]
        for r in self.rows:
            cells.append([fmt.format(r[key]) for key, _, fmt in _COLUMNS])
        widths = [max(len(c[i]) for c in cells) for i in range(len(_COLUMNS))]
        def line(c):
            body = " | ".join(x.rjust(w) if i else x.ljust(w) for i, (x, w) in enumerate(zip(c, widths)))
            return f"| {body} |" if markdown else body
        out = [line(cells[0])]
        out.append(("|" + "|".join("-" * (w + 2) for w in widths) + "|") if markdown else "-+-".join("-" * w for w in widths))
        out += [line(c) for c in cells[1:]]
        return "\n".join(out)

    def __str__(self) -> str:
        return self.table()

    def best(self, key: str = "worst_stop_db") -> Dict:
        """Row with the smallest value of ``key`` (default: the worse of the two stopbands)."""
        return min(self.rows, key=lambda r: r[key])


def compare(pairs: Union[Mapping[str, FilterPair], Sequence[FilterPair]], n_grid: int = 4000) -> Comparison:
    """
    Compare designs side by side.

    Parameters
    ----------
    pairs : mapping name -> FilterPair, or a sequence (named by their ``method``).
    n_grid : frequency grid size for the measurements.

    Columns: lengths, latency, ``stored`` (coefficients to store: symmetry exploited, zero end taps dropped, a
    matched ``u = d`` counted once), ``MACs`` (multiplications per base-rate sample, polyphase, symmetry NOT
    exploited, zero end taps dropped), stopband attenuation of ``u`` and ``d`` in dB (negative numbers, more
    negative is better), the larger passband ripple in dB and the Nyquist error.  Each row also has
    ``worst_stop_db``.  All pairs need a ``passband``.
    """
    items = list(pairs.items()) if isinstance(pairs, Mapping) else [(p.method, p) for p in pairs]
    rows = []
    for name, fp in items:
        m = fp.metrics(n_grid=n_grid)
        rows.append(dict(
            name=str(name), L=fp.L, n_u=fp.n_u, n_d=fp.n_d, latency=fp.latency,
            stored=fp.stored_coeffs, macs=fp.macs_per_sample,
            u_stop_db=m["u_stop_db"], d_stop_db=m["d_stop_db"],
            worst_stop_db=max(m["u_stop_db"], m["d_stop_db"]),
            ripple_db=max(m["u_ripple_db"], m["d_ripple_db"]), nyquist_error=m["nyquist_error"]))
    return Comparison(rows)
