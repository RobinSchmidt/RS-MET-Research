

```
```


## Status

Early development (`0.1.0.dev0`); everything below is implemented and tested.

| Area | API |
|---|---|
| specification | `Spec`, `passband_from_hz` |
| two-kernel design | `design_pair` (spec- or length-driven, own `u` allowed), `design_partner` (null-space / KKT, grid / energy objective) |
| matched kernel `u = d = g` | `design_matched` (from scratch, multi-start; `L = 2` not recommended) |
| experimental | `optimise_joint`, `rebalance`, `tie` |
| analysis | `FilterPair.metrics()`, `compare()`, `prosampler.plots` (matplotlib optional) |
| checks | `nyquist_error`, `verify_roundtrip`, `reference_roundtrip` |

Important: a windowed-sinc `u` is itself (nearly) Nyquist(L), so `d = delta` is an exact partner. Exactness of the
roundtrip says nothing about filter quality; check `FilterPair.metrics()` / `info["meets_spec"]`.

Documentation: [usage guide](docs/usage.md), [theory and design notes](docs/theory.md).
Demo: [`demos/design_and_plot.py`](demos/design_and_plot.py) designs a pair from a few parameters (in Hz) and plots impulse,
step and magnitude responses; see [demos/README.md](demos/README.md).

Note: with a nonlinearity between the filters, a matched kernel changes the signal level at the nonlinearity
(`FilterPair.shaper_level`, `1/sqrt(L)`); compensate with `reference_roundtrip(..., compensate_levels=True)` or scalar
gains around the waveshaper (usage guide, section 3).

```python
import prosampler as ps
fp = ps.design_pair(L=4, passband=0.85, atten_db=60)
fp.verify()
print(ps.compare({"pair": fp, "matched": ps.design_matched(4, 0.85, 60)}))
```

## Conventions

* Filters run at the high rate; frequencies are in rad per high-rate sample.
* `passband` is a fraction of the original Nyquist frequency (0.9 -> 19.8 kHz at 44.1 kHz).
* The stopband starts at `(2 - passband)`, the mirror image around the original Nyquist frequency
  (forced by the Nyquist condition).
* `u` has DC gain `L`, `d` has DC gain 1.

## Install and test

```
pip install -e .[dev]
pytest
```
