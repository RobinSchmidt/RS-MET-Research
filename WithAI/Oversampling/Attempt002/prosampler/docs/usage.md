# prosampler - usage guide

All code blocks on this page are executed by the test suite (`tests/test_docs.py`), so they stay correct.

## 1. The idea in one picture

```
x -> zero-stuff by L -> u -> [waveshaper] -> d -> keep every L-th sample -> y
```

`u` is the anti-imaging (upsampling) filter, `d` the anti-aliasing (downsampling) filter. With the waveshaper
removed, `y[m] = x[m - latency]` exactly (up to float roundoff) because `u * d` is a Nyquist(L) filter. See
[theory.md](theory.md).

Conventions: `passband` is a **fraction of the original Nyquist frequency** (0.9 means 19.8 kHz at 44.1 kHz).
`u` has DC gain `L`, `d` has DC gain 1. `latency` counts **base-rate** samples.

## 2. Quick start: a pair for a given specification

```python
import numpy as np
import prosampler as ps

fp = ps.design_pair(L=4, passband=0.85, atten_db=60)
print(fp)                                  # lengths, latency, Nyquist error
print(fp.info["meets_spec"])               # True: both kernels reach 60 dB and the ripple limits
m = fp.metrics()
print(round(m["u_stop_db"], 1), round(m["d_stop_db"], 1))
fp.verify()                                # identity roundtrip on white noise; raises VerificationError otherwise
```

`design_pair` searches the shortest linear-phase pair that meets the specification. Always look at
`info["meets_spec"]` (or call with `strict=True` to get an exception): an exact roundtrip alone proves nothing about
quality, since a windowed-sinc `u` is itself nearly Nyquist(L) and `d = delta` is an exact partner.

Passband from Hz:

```python
passband = ps.passband_from_hz(19000.0, 44100.0)       # 0.8617...
assert 0.86 < passband < 0.87
```

## 3. Using the filters in your own code

`fp.u` and `fp.d` are plain NumPy arrays. The reference chain `ps.reference_roundtrip(x, u, d, L, latency, shaper)`
is slow (full convolutions) and meant for testing. In a real implementation use polyphase filters: the upsampler
computes `y[L m + p] = sum_j x[m - j] u[L j + p]` for each phase `p`; the downsampler only computes every L-th
output. A block-processing version to check your own implementation against:

```python
def poly_up(x, u, L):
    y = np.zeros(len(x) * L + len(u) - 1)
    for p in range(L):
        c = np.convolve(x, u[p::L])
        idx = L * np.arange(len(c)) + p
        k = idx < len(y)
        y[idx[k]] = c[k]
    return y

def poly_down(v, d, L, latency, n):
    out = np.zeros(n)
    for r in range(L):
        dr = d[r::L]
        idx = L * np.arange(n + latency + len(dr)) - r
        ok = (idx >= 0) & (idx < len(v))
        vr = np.where(ok, v[np.clip(idx, 0, len(v) - 1)], 0.0)
        out += np.convolve(vr, dr)[latency:latency + n]
    return out

x = np.random.default_rng(0).standard_normal(300)
y = poly_down(np.tanh(poly_up(x, fp.u, fp.L)), fp.d, fp.L, fp.latency, len(x))
ref = ps.reference_roundtrip(x, fp.u, fp.d, fp.L, fp.latency, shaper=np.tanh)
assert np.max(np.abs(y - ref)) < 1e-12
```

**Levels with a nonlinearity.** The level at the waveshaper is `fp.shaper_level = sum(u)/L` times the input level:
1 for a pair (`u` has DC gain `L`), but `1/sqrt(L)` for a matched kernel (`u = d = g`, gain `sqrt(L)` each). The
identity roundtrip does not care, a nonlinearity does. So multiply by `1/fp.shaper_level` before the waveshaper and by
`fp.shaper_level` after it (for a matched kernel this is two scalar gains, no extra kernel). The reference chain
does this with `compensate_levels=True`.

The output is delayed by `fp.latency` base-rate samples. Linear-phase kernels have a constant delay, so this is a
pure delay. Zero end taps (forced by the Nyquist condition) can be dropped; `fp.macs_per_sample` and
`fp.stored_coeffs` already account for that and for symmetry.

## 4. Variations of the two-kernel design

Fixed lengths instead of a specification (`n_d` given: length-driven; then `atten_db` is only used to judge the result):

```python
fp2 = ps.design_pair(4, 0.85, 55, n_u=99, n_d=215)
assert fp2.latency == 39
```

For a centred linear-phase pair `n_u + n_d - 2` must be a multiple of `2 L`; `ps.partner_length(n_u, L, latency)`
and `ps.symmetric_latency(n_u, n_d, L)` help choosing lengths.

Your own `u` (any window, any optimised FIR); it is rescaled to DC gain `L`:

```python
u = ps.kaiser_sinc(4, 99, atten_db=60)
fp3 = ps.design_pair(4, 0.85, 55, u=u, n_d=215)
assert fp3.info["u_rescaled"] and fp3.nyquist_error < 1e-12
```

Solver and objective of the partner design (all four combinations are exact; they differ in the stopband of `d`):

```python
spec = ps.Spec(4, 0.85)
d_ns  = ps.design_partner(u, spec, 215, latency=39)                                  # null-space, grid objective (default)
d_kkt = ps.design_partner(u, spec, 215, latency=39, method="kkt", objective="energy")  # as in the old KKT script
assert ps.nyquist_error(u, d_ns, 4, 39) < 1e-12 and ps.nyquist_error(u, d_kkt, 4, 39) < 1e-12
```

`method="nullspace"` is the robust default; `"kkt"` solves the KKT system directly and is more fragile for long
kernels. `objective="grid"` fits a flat passband and a zero stopband on a frequency grid, `"energy"` minimises the
closed-form stopband energy. `transition_weight` (default 0) adds a least-squares penalty on the transition band. It looks like a way to flatten the
small bump of `d` at the passband edge (about +1.1 dB), but it does not: measured on `L=4`, passband 0.9, 261/541 taps,
the bump grows to +1.5...+1.6 dB while the stopband of `d` degrades from -104.6 dB (weight 0) to -87.9 dB (0.003),
-68 dB (0.01) and -53 dB (0.3); a spec-driven 80 dB design then needs a `d` that is 25-50 % longer. Leave it at 0.

## 5. A single matched kernel (u = d = g)

```python
g = ps.design_matched(4, 0.85, 60)         # from scratch; spec-driven (or K=... for a fixed length)
assert g.is_matched and g.nyquist_error < 1e-12
print(g.n_u, g.latency)                    # g has L*K + 1 taps, latency K
```

Matched kernels need about a quarter fewer stored coefficients but about half again as many multiplications as the
pair (section 7). The optimisation has local minima, so the result depends on the start and is not monotone in `K`; above 60 dB
`design_matched` automatically tries several Kaiser starts (`betas=(2, 4, 6)`; set `betas=` yourself to change this),
which makes high attenuations (80 dB and more) reachable but slower (minutes). `info["meets_spec"]` says whether the
target was reached. `L = 2` is not recommended (warns): the problem has no degrees of freedom.

## 6. Experimental: joint optimisation, rebalancing, tying

These work on symmetric centred pairs such as the output of `design_pair`:

```python
small = ps.design_pair(4, 0.8, 40)
joint = ps.optimise_joint(small)                       # optimise u and d together
equal = ps.rebalance(joint, target="equal")            # shift taps from d to u until the lengths are equal
tied  = ps.tie(equal)                                  # exact u = d = g, polished by the matched solver
assert equal.n_u == equal.n_d and tied.is_matched
```

`rebalance` records the whole path in `equal.info["path"]` (use `plots.plot_path`); `target="best"` returns the
point on the path with the best worst-case stopband. Tying usually costs a lot of stopband attenuation at equal
total length.

### Minimax objective and joint minimax

```python
fp = ps.design_pair(4, 0.85, 60, objective="minimax")      # linear program: minimal stopband PEAK of d, shorter d
ref = ps.optimise_joint_minimax(fp)                        # alternating LPs for d and u; each filter stays flat
print(ref.info["joint_minimax"]["history"])                # (u, d) stopband peaks after every round
```

`objective="minimax"` is a global optimum (convex), typically 7-14 % fewer taps in `d` than the grid objective for
the same specification, at the cost of seconds to about a minute. `transition_max_db=` and `passband_flat_db=` add
linear bounds (a tight transition bound costs a lot of stopband because `d` must compensate `u`'s roll-off).
`optimise_joint_minimax` keeps the lengths; its gain at the usual 1 : 2 split is mostly in `d`, the worst peak (that of
`u`) improves by only 1-1.4 dB. To use the freedom, shift taps from `d` to `u` (give `n_u`, `n_d` to `design_pair` with a
stronger `u` start and refine). Both routines need symmetric odd-length kernels and the centred latency.

### Simultaneous joint minimax and the transition bump

```python
small = ps.design_pair(4, 0.8, 40)
m = small.metrics()
r0 = max(m["u_ripple_db"], m["d_ripple_db"])             # hold the passband ripple at the start's value
sl = ps.optimise_joint_slp(small, ripple_db=r0, transition_max_db=0.5, iterations=15)
print(max(sl.metrics()["u_stop_db"], sl.metrics()["d_stop_db"]), sl.metrics()["d_transition_peak_db"])
```

`optimise_joint_slp` moves `u` and `d` in the same step (sequential LPs, seconds per iteration for kernels of a few hundred
taps, minutes in total at 100+/200+ taps). At the same lengths and ripple it gains 7-9 dB without a transition cap; the
cap `transition_max_db` limits the gain of both filters between the band edges (`0` = no bump) at a modest cost: caps of
0.5 / 0.2 / 0 dB give +3 / +1 / -1.5 dB relative to the plain pair in the 0.85 / 60 dB example. **Compare at equal ripple:**
the allowed ripple (`ripple_db`, peak-to-peak like `metrics()`) is a free resource and every doubling buys about 3 dB.
The result is a local optimum; `shape=ps.TransitionShape(...)` additionally prescribes a roll-off tube (research option,
see `docs/theory.md` section 5).

### Shortest pair for a specification

```py
best = ps.search_pair(4, 0.8, 40, transition_max_db=0.5, resolution="medium", n_jobs=2)   # minutes to hours
print(best.n_u, best.n_d, best.stored_coeffs)
print(best.info["search"]["table"][:3])                          # every candidate tried
```

`search_pair` reduces the total length round by round and tries several `u` / `d` splits per length (start: Kaiser-sinc `u`
+ minimax partner, refined by `optimise_joint_slp` with the passband ripple held at the plain pair's value, +2 % for grid
differences, and an optional transition cap). It returns the shortest pair found that meets the specification; the plain pair
is returned if nothing shorter works. The result is the shortest length *found* by a local search, not a proof of optimality;
neighbouring candidates scatter by a few dB.

**Resolution.** Two grids control the search; `resolution` selects a preset and each grid can be overridden:

| `resolution` | `length_step` | `split_step` / span of `u` offsets | window | SLP iterations | candidates per round | use |
|---|---|---|---|---|---|---|
| `"coarse"` | 4 L (16 taps for L = 4) | 16 / 0...+32 | none | 20 | 3 | quick look; can miss the best split |
| `"medium"` (default) | 2 L (8) | 8 / 0...+40 | none | 25 | 6 | good default |
| `"fine"` | 2 L (8) | 4 / -8...+48 | +-12 after round 1 | 30 | 15, then 7 | finest useful split resolution |

* **`length_step`** (total taps removed per round) cannot be finer than `2 L`: a centred linear-phase pair needs `n_u + n_d - 2` to be a
  multiple of `2 L` with both lengths odd. The result is therefore only accurate to `length_step` taps.
* **`split_step`** (even, >= 2) is the spacing of the candidate lengths of `u` within a round; **`split_span`** is their range relative to the
  plain pair's `u`. Longer `u` is often much better, hence ranges that extend upward. **`window`** restricts later rounds to
  splits near the previous round's best `u` (the best split moves slowly), which cuts the cost of a fine grid by 2-3x.
* **Cost:** one configuration is one SLP run. Measured (one run per core, 20-30 iterations): about 10-40 s for totals of 130-160 taps
  and 30-60 s for about 300 taps; it grows faster than linearly with the length. The number of configurations is
  (rounds) x (candidates per round); the search ends at the first round with no feasible candidate, typically after a ~20 %
  length reduction, i.e. 5-9 rounds. Typical durations on one core: `"coarse"` 5-15 min, `"medium"` 10-40 min, `"fine"` 30-120 min,
  longer for long kernels (high attenuation, passband near Nyquist). **`n_jobs`** (`-1` = all cores) runs the candidates of a
  round in parallel and divides the time by about the number of cores.

**Best pair for a given length.** `search_pair` finds the shortest length for a required attenuation. The opposite question - the best
stopband for a *given* total length - is answered by `best_pair_for_length(L, passband, total, ripple_db=0.05, transition_max_db="ripple", n_jobs=2)`.
It scans the split between `u` and `d` (default: `n_u` from 30 % to 46 % of `total` in steps of `split_step=4`, then the two
neighbours of the best at half the step), optimises every candidate with the SLP at the given ripple and bump cap, and returns the best pair;
`info["best_for_length"]["table"]` lists all candidates. `total - 2` must be a multiple of `2 L`. `atten_db` (default 60) only shapes the starting
designs and the reference ripple if `ripple_db` is omitted. Each candidate takes about a minute at ~300 taps. Neighbouring splits scatter by 1-2 dB (local optima), so
the result is the best *found*. Example: 0.85 passband, L = 4, 290 taps, 0.05 dB ripple, cap ripple/2: -66.2 dB at 109/181 (14 candidates, 11 min on 2 cores).

**Bump cap.** `transition_max_db` bounds the gain of both filters between the passband and stopband edges, relative to their DC gain (`0` = no overshoot). `transition_max_db="ripple"` sets it to half the allowed peak-to-peak passband ripple (the reference ripple in `search_pair`, `ripple_db` in `optimise_joint_slp`): the bump may then not exceed the gain that the passband ripple reaches anyway.

**Strategy.** `strategy="scan"` (default) descends from the plain length round by round. `strategy="adaptive"` treats the best
stopband peak at a total length as a nearly linear function of the length: it probes one length (about 12 % shorter than the plain
pair, full split scan), then jumps by secant steps to where the line reaches the specification (0.3 dB margin), keeps a bracket
(longest passing / shortest failing length) and stops when the two are one `length_step` apart. After the first probe only splits within
`window` taps (default 8) of the best split *ratio* are tried. It tests about half as many configurations as the scan (see below),
and the answer is confirmed by its failing neighbour. Measured for 0.85 / 60 dB, `resolution="fine"`, cap +0.5 dB, two cores: the scan needed 62 configurations (50 min), the adaptive search 27 (18 min) and both ended at 274 taps (103/171). Because neighbours scatter by a few dB, it can differ from the scan by one
`length_step`. `max_rounds` limits the number of probed lengths; the "at most" figure in the progress line is then a rough bound.

**Extra starts (`rescue`).** If no candidate of a length meets the specification, the `rescue_top` (3) best candidates are rerun with the
Kaiser window of the starting `u` designed for `atten_db + delta` dB, for each `delta` in `rescue` (default `(-6, 6)`), to see whether a
different start lands in a better local optimum. It costs `rescue_top * len(rescue)` extra configurations per failing length; `rescue=()`
switches it off. In the one case tested (0.85 / 60 dB, cap +0.2 dB, 266 and 258 taps) it changed nothing: the best result stayed at -58.6 dB
(266) and -56.6 dB (258), so the SLP result depends little on the start and the scatter between neighbours is mostly the split itself.
The run took 39 instead of 27 configurations (29 instead of 18 min).

**Progress.** With `progress=True` (default) one line is printed per finished configuration, for example

```text
[ 14 of at most 36 | 4:12 elapsed | typically ~3:10, at most 6:30 left] total 266 split 115/151: worst -61.2 dB, bump +0.50 dB, ripple 0.0160 dB, meets
```

"At most" assumes that every round is run to the limit of `max_rounds`; "typically" assumes ~20 % length reduction and uses the average
time per configuration so far (so it is pessimistic for the first few lines). Pass a callable to receive the same information as a
dict (`done, upper_bound, typical, elapsed, eta, eta_typical, round, total, n_u, n_d, worst_db, bump_db, meets`), or `progress=False`.
These examples are not part of the executed tests because of the runtime.

## 7. Comparing designs

```python
print(ps.compare({"pair": fp, "matched": g}))
```

```
name    | L | n_u | n_d | lat | stored | MACs | u stop | d stop | ripple | nyq err
```

`stored` is the number of coefficients to keep (symmetry exploited, zero end taps dropped, a matched kernel counted
once), `MACs` the multiplications per base-rate sample (polyphase, symmetry not exploited). Measured for
`L=4, passband=0.85, 60 dB`: pair 99 + 215 taps, 157 stored, 312 MACs, -60.3 / -60.9 dB; matched 233 taps, 116
stored, 462 MACs, -60.2 dB.

For a complete, runnable scenario (design from Hz, report, roundtrip check, impulse/step/magnitude plots) see
[`demos/design_and_plot.py`](../demos/design_and_plot.py).

## 8. Plots (optional, needs matplotlib)

```python
import matplotlib
matplotlib.use("Agg")
from prosampler import plots

ax = plots.plot_responses(fp)              # magnitude responses with shaded passband/stopband
plots.plot_kernels(fp)
plots.plot_path(equal)                     # stopbands along a rebalance path
```

## 9. Troubleshooting

| Symptom | Cause / remedy |
|---|---|
| `info["meets_spec"]` is `False` / `SpecNotReached` | spec too hard for the chosen lengths: give longer `n_d` (or `n_u`), or use spec-driven mode |
| `d` has a small bump (about +1 dB) between passband and stopband | normal for these designs; `transition_weight` does not remove it (see section 4); `info`/`metrics()` report it as `d_transition_peak_db` |
| `ProsamplerError: ... no exact partner` | `n_d` too short for the Nyquist constraints, or wrong latency |
| `ValueError: latency must be given` | `u` is not symmetric/odd or `n_d` is even: pass `latency=` explicitly |
| matched design misses the target | local minimum: try other `betas=`, a slightly different `K=`, or use the pair |
| `rebalance` is slow | each step costs seconds at several hundred taps; use `progress=` to watch |

## 10. API summary

| Name | Purpose |
|---|---|
| `Spec(L, passband, atten_db)`, `passband_from_hz` | specification |
| `kaiser_sinc`, `windowed_sinc`, `design_kaiser_sinc` | starting kernels |
| `design_pair`, `design_partner` | pair design (steps 1-2) |
| `design_matched`, `optimise_matched` | matched kernel from scratch |
| `optimise_joint`, `rebalance`, `tie` | experimental steps 3-5 |
| `design_partner_minimax`, `optimise_joint_minimax` | LP-based minimax partner design and alternating joint refinement |
| `search_pair` | shortest pair found for a specification (length and split search with the SLP) |
| `optimise_joint_slp`, `TransitionShape` | simultaneous joint minimax by sequential LP, with an optional transition cap or prescribed roll-off |
| `FilterPair` | result: `u, d, L, latency, metrics(), verify(), swapped(), stored_coeffs, macs_per_sample` |
| `nyquist_error`, `verify_roundtrip`, `reference_roundtrip` | checking |
| `compare`, `prosampler.plots` | analysis |
