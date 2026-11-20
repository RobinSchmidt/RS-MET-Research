# Exactly transparent FIR oversampling filter pairs for waveshaping — a research summary

*Written up 2026-11-17, corrected 2026-11-18 (see the note at the end of section 3 about a measurement fix). Companion code: the `prosampler` Python package (`prosampler.zip`, docs in `docs/usage.md` and `docs/theory.md`). Numbers below were measured in the session unless marked otherwise.*

---

## 1. Problem statement

Nonlinear audio processing (waveshaping, distortion) is usually run at an oversampled rate to limit aliasing:

```
x -> zero-stuff by L -> u -> [nonlinearity f] -> d -> keep every L-th sample -> y
```

`u` is the upsampling (anti-imaging) filter, `d` the downsampling (anti-aliasing) filter, both FIR. Normally `u` and `d` are designed independently as good lowpass filters. Then, even for `f` = identity, the roundtrip is only *approximately* transparent.

**Goal.** Design `u` and `d` such that the identity roundtrip is exact (to float roundoff) while both are good lowpass filters (high stopband attenuation, flat passband up to a chosen fraction of the original Nyquist frequency). Variants wanted:

1. a pair `(u, d)` with `u != d` (lengths free);
2. a single matched kernel `g` with `u = d = g` (mainly to save memory in resource-constrained environments);
3. user-adjustable parameters: oversampling factor `L`, stopband attenuation, passband width.

Conventions: passband is a **fraction of the original Nyquist frequency**; `u` has DC gain `L`, `d` has DC gain 1; latency is counted in base-rate samples.

## 2. Theory in brief

### 2.1 The Nyquist(L) condition

Let `h = u * d` and `n0 = L·K` (`K` = latency in base-rate samples). With `f` = identity the chain reproduces the input iff

```
h[n0 + L·j] = δ[j]     for all integers j.                              (N)
```

I.e. `h` is an L-th-band (Nyquist(L)) filter. In the frequency domain: `Σ_{j=0}^{L-1} H(ω − 2πj/L) = L·e^{−iω·n0}` (verified numerically). Imaging terms of the interpolator and aliasing terms of the decimator cancel exactly, independent of the signal.

### 2.2 Consequences

* **Mirrored band edges.** The stopband cannot start before `(2 − passband)·π/L` (rad per high-rate sample): passband edge and stopband edge are mirror images about the original Nyquist frequency.
* **Exactness proves nothing about quality.** A windowed-sinc `u` is itself nearly Nyquist(L), so `d = δ` is an exact partner. Always check stopband/ripple, not just the roundtrip error.
* **Zero end taps.** `h[0] = u[0]·d[0] = 0` forces the outer taps of `d` (and of a matched `g`) to be ≈ 0.
* **Linear phase.** For symmetric odd-length kernels the centre of `h` is at `(n_u + n_d − 2)/2`, which must be a multiple of `L`; this fixes the latency `K`.

### 2.3 Two-kernel design (linear problem)

For a fixed `u`, (N) is **linear** in `d`: `C d = e`. The design of `d` is then an equality-constrained least-squares problem, solved by

* the **null-space method** (SVD of `C`; robust; default), or
* the **KKT system** `[H Cᵀ; C 0]` (what the first attempt's script did; the matrix is ill-conditioned for long kernels, condition ~1e12).

Objectives: a frequency-grid fit (flat passband, zero stopband, optional transition weight) or the closed-form stopband energy `dᵀQd`. Both solvers reach the same optimum (agreement within 1 dB in the tests).

*Degrees of freedom:* `C` has about `(n_u + n_d)/L` rows, so `d` needs to be about **twice as long as `u`** (observed `n_d/n_u` ≈ 1.8–2.5) for a comparable stopband.

### 2.4 Matched design (nonlinear problem)

For `u = d = g`, (N) becomes **quadratic**: `(g*g)[n0 + L·j] = δ[j]`. Nonconvex NLP: minimise the stopband energy of `g` subject to it.

* Degrees of freedom for symmetric `g` with `N = L·K + 1` taps: `(L − 2)·K/2`.
* **L = 2 has zero degrees of freedom**: solutions are isolated points of a square system, ill-conditioned (Jacobian condition ≈ 3e12). Good solutions can exist (one at about −69 dB was found by continuation from a pair) but are hard to find and to project to exact `u = d` (≈ −52 dB at 5e-10 error). Prefer the pair for L = 2.
* `g` has DC gain only **approximately** `√L` (off by ≈ 6e-7): (N) fixes every L-th sample of `g*g`, not the total sum.

## 3. Findings

### 3.1 Two-kernel pair (`design_pair`, `design_partner`)

* Exact to ≈ 1e-16 for all four combinations of {null-space, KKT} × {grid, energy}.
* Reference case L = 4, passband 0.9, `u` = 261-tap Kaiser-sinc (100 dB, measures −100.2 dB), 541-tap `d`, latency 100 (stopband of `d`, exact edge on the grid):
  * energy objective, null-space or KKT (the first attempt's script used KKT + energy): **−98.4 dB**;
  * grid objective (null-space or KKT): **−104.6 dB**. The grid objective is the better default.
  * (Earlier in the session the energy design was reported as −100.4 dB and said to "reproduce" the first attempt's number. That measurement missed the stopband-edge peak by about 2 dB, see the correction note below.)
* Spec-driven search (L = 4, passband 0.85, 60 dB): 99-tap `u`, 215-tap `d`, latency 39, both ≈ −60 / −61 dB.
* Demo scenario (L = 4, 20 kHz passband at 44.1 kHz, 80 dB): `u` 217 taps, `d` 457 taps, latency 84 samples (1.9 ms). Lengths drop strongly with a lower passband: 18 kHz → 121/241 taps, 16 kHz → 81/161 (80 dB).
* **Low-latency (non-linear-phase) variants turned out useless** in the early experiments; the library is linear-phase only.
* `d` has a small bump of ≈ +1.1 dB at the passband edge. **`transition_weight > 0` does not remove it**: measured at 261/541 taps the bump grows to +1.5…+1.6 dB and the stopband of `d` degrades from −104.6 dB (weight 0) to −87.9 (0.003), −68 (0.01) and −53 dB (0.3); spec-driven 80 dB designs then need a 25–50 % longer `d`. Keep it at 0. (It was introduced for an early low-latency `d` with a +117 dB bump, a different regime; hence also the 3 dB limit in the spec check.)

### 3.2 Matched kernel (`design_matched`)

* At equal total length a matched kernel is **20–35 dB worse** than the pair (far fewer degrees of freedom), so for equal attenuation it needs more taps.
* **Memory vs multiplies** (L = 4, passband 0.85, 60 dB):

  | Design | taps | stored coefficients (symmetry, shared) | MACs per base-rate sample | stopband |
  |---|---|---|---|---|
  | pair `u`, `d` | 99 + 215 | 157 | 312 | −60.3 / −60.9 dB |
  | matched `g` | 233 | 116 | 462 | −60.2 dB |

  Matching saves ≈ 26 % coefficient memory but costs ≈ 48 % more multiplies; worthwhile only if memory, not arithmetic, is the constraint.
* **Local minima.** The result depends on the starting kernel and is **not monotone in `K`**. L = 4, passband 0.9, default Kaiser start (`beta = 4`):

  | K | 100 | 110 | 120 | 130 | 140 |
  |---|---|---|---|---|---|
  | stopband (dB) | −68.0 | −72.4 | −76.5 | −82.7 | −74.5 |

  At K = 140 a start with `beta = 2` gives −85.6 dB (but only −79.8 dB at K = 130). A spec-driven search that only warm-started from the previous *failed* design got stuck near −75 dB all the way to K = 596 (505 s, spec missed). The documented "plateau at −74 dB" was this warm-start trap, not a solver limit.
* **Fix:** multi-start (several Kaiser betas plus the warm start, keep the best; automatic above 60 dB). With it (L = 4, passband 0.9): 70 dB at K = 104 (N = 417, −71.4 dB, 3 min); 80 dB at K = 124 (N = 497, **−87.3 dB**, 4 min); 85 dB at K = 138 (N = 553, −85.4 dB, 4 min). K = 124 beats K = 130 (−82.7 dB) by 4 dB: a shorter kernel can be better, so the bisection finds *a* length meeting the spec, not necessarily the shortest.

### 3.3 Joint optimisation, rebalancing, tying (experimental)

* **Joint optimisation** of `(u, d)` (balanced stopband ENERGY objective, optional passband flatness term; exact Nyquist; `sum(u) = L`): the energy always drops (to 30–60 %), but the worst stopband **peak** does not reliably improve. Without a passband term the peak gains up to ≈ 6 dB (L = 4, passband 0.8, 40 dB: −41.9 → −47.7 dB) while per-filter ripple explodes (0.12 → 0.96 dB there, up to 5.9 dB elsewhere), because (N) only fixes the product `u * d`. With `passband_weight = 1` the peak got **worse** in every case tried (p 0.8/40: −41.9 → −38.3; 0.8/60: −60.4 → −55.4; 0.8/80: −80.2 → −79.0; 0.9/40: −40.1 → −34.3; 0.9/60: −60.3 → −53.7; 0.9/80: −80.0 → −75.7; unchanged at 100 dB). No weight beat the plain pair in both peak and ripple. *(Earlier claims of a ≈ 6 dB benefit ignored passband ripple and were wrong.)* A minimax objective is needed for a real gain.
* **Rebalancing** (move taps from `d` to `u` two at a time, total length constant, re-optimise): 47/115 → 81/81 gives −47.4 / −49.6 dB.
* **Tying** (penalty continuation, average, exact matched polish): −28.9 dB for the same example — tying costs 20–35 dB at equal total length.
* **Continuation at larger size** (L = 4, passband 0.9, N = 521, K = 130; start 361/681 taps, 100 dB; start pair −103.3 / −112.5 dB, ripple 0.0001 dB), three runs:

  | Route | stopband | notes |
  |---|---|---|
  | from scratch, best of 5 Kaiser starts, K = 130 | −82.7 dB | 33 s |
  | run 1: old grid, no passband term; rebalance, tie | −81.7 dB | kept as originally measured |
  | run 2: corrected grid, no passband term; rebalance, tie | −88.2 dB | ripple unchecked |
  | run 3: corrected grid, passband term on; path ends 521/521 at −93.9 / −86.2 dB (ripple 0.0002) | averaged −83.8 dB; refined tied `g` **−82.9 dB** (ripple 0) | path 935 s, total 1140 s |

  The path stays near −103 dB for ~3 steps, then degrades to ≈ −86 dB. **With the passband term, tie ≈ from scratch (−82.9 vs −82.7 dB): no reliable advantage**, at ~30× the cost. Run 2's −88.2 dB was probably helped by unconstrained ripple. (Earlier versions of this summary drew the opposite conclusions from run 1 alone, then from runs 1–2; both are superseded.)

* **Correction note (2026-11-18).** The stopband peak of a good design sits at the stopband edge, but the evaluation grid did not contain the exact edge frequency, so reported attenuations were grid-size dependent and up to ~2 dB too optimistic (a 60 dB matched design printed −59.6 dB in one report and passed `meets_spec` in another). The grid now always contains both band edges. The tables in 3.1–3.3 were re-measured with the fix (N = 521 run 1 is kept as originally measured and marked as such). Not re-measured, and therefore only approximate: the "20–35 dB worse at equal total length" figure and the L = 2 matched-design numbers (−69 dB, −52 dB) from earlier experiments.

### 3.4 Swapping roles

Exchanging `u` and `d` (with a gain fix: new `u = d·L/sum(d)`, new `d = u·sum(d)/L`) leaves `h` and the identity roundtrip unchanged. With a nonlinearity in between the two assignments are **not** equivalent (the first filter determines what `f` sees). For matched kernels swapping is meaningless.

### 3.5 Minimax (peak) objective (`objective="minimax"`, `optimise_joint_minimax`)

* **Why:** energy objectives can lower the far stopband while raising the peak at the edge (which is what the joint-energy experiments showed). For symmetric centred kernels the stopband bound `|A(w)| <= t` and the Nyquist conditions are linear, so the partner design is an LP: global optimum, equiripple, no local minima. The same LP designs `u` given `d`.
* **Partner design (u fixed):** L = 4, passband 0.9, 261-tap Kaiser `u` (100 dB), 541-tap `d`: −104.6 dB (grid LS) → **−114.5 dB**. Spec-driven, `d` gets 7–14 % shorter: 115 → 99 taps (0.8 / 40 dB), 215 → 191 (0.85 / 60 dB), 433 → 401 (0.9 / 80 dB). LP time seconds to ~70 s.
* **Transition bump:** a bound on `d` between the band edges is respected exactly but costs stopband steeply (passband 0.9, 541 taps: +1.0 dB → −99.9 dB, +0.5 → −87.1, +0.2 → −68.6, ≤ 0 infeasible). With `u` fixed, `d` must compensate `u`'s roll-off (`d(edge) ≈ L / u(edge) ≈ 0 dB`), so the bump is a matter of how the transition is split between `u` and `d`.
* **Joint (alternating LPs, each filter's passband ripple bounded):** both peaks are non-increasing and the ripple stays unchanged, but at the usual 1 : 2 split `u` gains only 1–1.4 dB while `d` gains 12–17 dB. Real gains need a different split at constant total length: 0.85 / 60 dB, 314 taps: 117/197 gives −63.7 / −63.8 dB against −60.3 dB for the plain 99/215 (start dependent: another start gave −63.0 dB at 109/205).
* **Shortest total length meeting the spec:** 0.8 / 40 dB: 162 (plain) → 146 (minimax partner) → 138 (joint, 45/93). 0.85 / 60 dB: 314 → 290 (minimax partner; joint gives the same length with more margin). So most of the saving comes from the minimax partner.
* **Not done:** a minimax version of the matched design (`u = d`, nonconvex); a joint LP over lengths and split; checking how much better the best local optimum of the alternation is (several starts).

### 3.6 Simultaneous joint minimax and the transition bump (`optimise_joint_slp`)

* **Alternating LPs cannot move the roll-off split** (with a bound on `d`, `u` stayed identical and `d` paid in stopband). Moving both kernels at once needs the bilinear Nyquist coupling: sequential LP (linearise, trust region, Gauss-Newton projection). One LP is seconds; 99/215 taps takes 30–170 s in total. Local optimum.
* **Ripple is a free resource**, so compare at equal ripple (held at the start's value): each doubling of the allowed ripple buys ≈ 3 dB and enlarges the bump (0.8 / 40 dB: −48.5 / −51.5 / −55.3 dB at 1x / 2x / 4x ripple, bumps +1.7 / +2.5 / +4.0 dB). An early run of mine at 4x ripple overstated the gain.
* **Equal lengths and equal ripple:**

  | cap on transition gain | 0.8 / 40 dB, 47/115 | 0.85 / 60 dB, 99/215 |
  |---|---|---|
  | plain pair | −41.9 dB (bump +1.25) | −60.3 dB (+1.17) |
  | none | −48.5 (+1.69) | −69.1 (+1.72) |
  | +1.0 dB | −45.3 | −65.7 |
  | +0.5 dB | −42.7 | −63.2 |
  | +0.2 dB | −40.9 | −61.5 |
  | 0 dB | −37.8 | −58.8 |

  Gain 7–9 dB without a cap; 1–3 dB with a bump ≤ +0.5 dB; an essentially bump-free pair costs ≈ 1 dB or gains ≈ 1 dB. This answers the earlier open question: yes, the optimiser can move the bump into `u` once `u` and `d` move together, and it shifts the roll-off split toward the matched case (at the original Nyquist frequency `u` / `d`: −5.9 / 0.0 dB plain → −3.9 / −2.0 dB at cap 0, matched −3 / −3 dB).
* **Prescribed shapes:** the product of the transition amplitudes must satisfy `r(x) + r(1 − x) = 1` (cosine², linear, Kaiser-window integral; `1 − x^p` is not admissible), with split `u = r^alpha`, `d = r^(1 − alpha)`. As a linear tube this was too restrictive for a short `u` and long `d` (alpha → 1 forces `d` ≈ 0 dB up to the stopband edge; cosine tubes with tol 0.05 stalled at −22.7 / −31.8 dB); the simple gain cap is the practical control. `TransitionShape` stays as a research option.
* **Not done:** lengths and split as LP variables (best split by scanning only), several starts to gauge local optima, the same simultaneous move for the matched design, and combining the SLP with the shorter-`d` result of the minimax partner for a minimum-length search.

## 4. Pitfalls (things that went wrong or are easy to get wrong)

1. **Exact ≠ good.** `d = δ` is an exact partner of a windowed-sinc `u` (a 61-tap `d` came out exact with ≈ 0 dB stopband). Check `metrics()` / `info["meets_spec"]`.
2. **Length constraint.** `(n_u + n_d − 2)` must be a multiple of `2L` for a centred linear-phase pair; `L·K` must be even for matched kernels (`N = L·K + 1`).
3. **Gain conventions.** `u` gain `L`, `d` gain 1; matched `g` ≈ `√L` (not exact). Swapping needs the gain fix.
4. **KKT conditioning.** Direct KKT solves are fragile for long kernels (≈ 1e12); null-space is the default.
5. **Matched L = 2:** zero degrees of freedom. An earlier claim that "L = 2 has no good matched solution" was wrong (zero DOF ≠ no solution) but practical solutions are poor and ill-conditioned.
6. **Warm starts can trap the optimiser.** Zero-padding a shorter solution keeps feasibility so the result should never get worse — but it gives no escape from a bad basin (and numerical noise cost 0.2 dB in one case: −84.0 → −83.8 dB at K = 150). Use multiple cold starts.
7. **Non-monotone in K.** Longer is not automatically better for the matched solver; scan K.
8. **Zero end taps** of `d`/`g` are forced; degenerate constraints there broke early SLSQP attempts (fixed by dropping the end taps from the parametrisation).
9. **Objective scaling.** Stopband energies of 1e-10 and below are under SLSQP's absolute tolerances; rescale to O(1) and restart.
10. **Overstated early claims** (joint-optimisation benefit, the "−74 dB plateau", "L = 2 impossible") — all corrected; treat anything not in the tables above as unverified.
11. **Reading helper:** reference roundtrips use full convolutions (slow); use polyphase in production (a tested block implementation is in `docs/usage.md`).
12. **Evaluation grid must contain the band edges.** The stopband peak is at the edge; a grid that misses it under-reports it (and makes `meets_spec` grid dependent). Fixed in `Spec.grid`; keep it in mind for any own measurement code.
13. **Signal level at the nonlinearity.** The shaper sees `sum(u)/L` times the input level: 1 for a pair, but `1/√L` for a matched kernel (gain `√L` each). The identity roundtrip hides this; a nonlinearity does not (a matched chain driven through `tanh(4x)` gave an output peak of 1.9 instead of ≤ 1). Scale by `1/shaper_level` before and `shaper_level` after the waveshaper (`FilterPair.shaper_level`, `reference_roundtrip(..., compensate_levels=True)`).
14. **`transition_weight` is not a bump remedy** (see 3.1): it enlarges the bump and wrecks the stopband.
15. **Single runs of the continuation/tying route are not reproducible in outcome** (6.5 dB spread, see 3.3); never judge it from one run.

## 5. Related work (not exhaustive; references partly unverified)

* The condition (N) is the **L-th-band (Nth-band) filter** property. Verified: F. Mintzer, "On half-band, third-band, and Nth-band FIR filters and their design", IEEE Trans. ASSP, 1982 (DOI 10.1109/TASSP.1982.1163950; volume/pages not checked); F. Mintzer, "Filters for distortion-free two-band multirate filter banks", IEEE Trans. ASSP, vol. 33, no. 3, pp. 626–630, 1985 (DOI 10.1109/TASSP.1985.1164587). From memory, not re-verified: Vaidyanathan, *Multirate Systems and Filter Banks* (Prentice Hall, 1993); Crochiere & Rabiner, *Multirate Digital Signal Processing* (Prentice Hall, 1983).
* **Perfect-reconstruction filter banks** use the same product-filter math but a different setting: aliasing from decimating each subband is cancelled by the *other* subbands. Our chain has a single channel and nothing to cancel against; only the product-filter condition remains.
* **Digital communications pulse shaping** is the structurally closest relative: oversampled transmit filter, receive filter, decimation, ISI-free requirement = Nyquist criterion. The matched case `u = d = g` is the **root-Nyquist** (square-root raised cosine type) filter. FIR root-Nyquist designs with stopband optimisation exist in that literature; I have **not** reviewed it and cannot say how much of the matched design it already covers.
* Suggested search terms: "root Nyquist filter design", "square-root Nyquist FIR", "Nth-band filter", "Nyquist pulse shaping filter design", "oversampled nonlinear processing alias suppression".
* I do not claim novelty; the combination with a nonlinearity in between, a two-kernel partner `d`, and the memory-vs-multiplies comparison are merely things I did not find in a non-exhaustive search.

## 6. Open questions and ideas for further research

1. **Measure what matters.** We never measured actual alias suppression with a real waveshaper (tanh, folding, …) — compare exact-PR designs against conventional independently designed filters of equal length at equal passband/stopband, on tones and noise. The exactness property only concerns the linear path.
2. **Minimax objective.** The pair's partner design is *convex*; a minimax (Chebyshev) stopband objective could be solved as an LP/SOCP and would give equiripple stopbands (better peak attenuation than the least-squares objectives used here).
3. **Root-Nyquist route for the matched case.** Design a Nyquist `h ≥ 0` with an optimised stopband and factor `h = g*g`. For linear-phase `g` this needs double zeros on the unit circle in the stopband, which turns the problem into a constrained (convex-ish) design of `h` rather than the nonconvex NLP in `g`. Speculative; compare with the root-Nyquist literature first.
4. **Global optimisation for the matched problem.** More starts, basin hopping, or continuation in `L`/passband; map the landscape over `K` and `beta`; understand why `beta = 2` often wins.
5. **Make continuation + tying reliable (or know when it wins).** Repeat the N = 521 experiment with many starting pairs to get a distribution; use the tied result as a warm start for `design_matched`; study the geometry of the feasible set `{g*g Nyquist}`; try a trust-region/Riemannian method instead of penalty + projection, or tie early in the path.
6. **Non-symmetric / low-latency filters.** The `partner` design supports non-symmetric `u` with explicit latency; a systematic study of minimum-phase or low-latency pairs (with nonlinear-phase side effects on the waveshaper) is unexplored — the early low-latency attempt was dropped as useless, but that was a narrow experiment.
7. **Transition band shaping.** `transition_weight` (least-squares penalty) does not help (3.1). A constrained formulation (e.g. as a bound on |D(ω)| in the transition band within a minimax/SOCP design) may; the +1.1 dB bump at the passband edge of `d` is the thing to understand: is it inherent to the Nyquist constraint with a Kaiser `u`?
8. **Larger `L` and other passbands.** Most experiments used L = 4 (some L = 3, 8). Check scaling of the memory/MAC trade-off and the matched solver for L = 8, 16, and for passbands near 1.
9. **Efficient implementation.** Polyphase + symmetric-folding implementations, a C++ port (the first attempt's repo already has a C++ version for the KKT pair), cost models including symmetry in the multiply count (currently not exploited in `macs_per_sample`).
10. **Tied-from-pair at equal memory.** At fixed *stored coefficients* rather than fixed total length, is the pair or the matched kernel better? (The table in 3.2 compares at roughly equal attenuation, not at equal memory.)
11. **Perceptual weighting** of stopband regions (e.g. stricter attenuation near the audible band edge, relaxed far above).
12. **Literature check** as described in section 5 before publishing any claims.

## 7. Library layout (for reference)

```
src/prosampler/
  spec.py       Spec, Grid, passband_from_hz
  kernels.py    kaiser_sinc, windowed_sinc, design_kaiser_sinc
  nyquist.py    nyquist_error, verify_roundtrip, symmetric_latency, partner_length
  partner.py    design_partner(u, spec, n_taps, latency, method=, objective=)
  design.py     design_pair(L, passband, atten_db, ...)          # steps 1-2
  matched.py    design_matched, optimise_matched (multi-start)   # independent route
  joint.py      optimise_joint, rebalance, tie                   # experimental steps 3-5
  pair.py       FilterPair (u, d, latency, metrics, verify, swapped, stored_coeffs, macs_per_sample)
  compare.py    compare()      plots.py   plot_responses / plot_kernels / plot_path
demos/design_and_plot.py   runnable scenario: design from Hz, report, roundtrip check, impulse/step/magnitude plots
docs/usage.md, docs/theory.md; tests: 62 passing (docs examples are executed by the test suite)
```

Minimal usage:

```python
import prosampler as ps
fp = ps.design_pair(L=4, passband=0.85, atten_db=60)    # u (99 taps), d (215 taps)
fp.verify()
g = ps.design_matched(4, 0.85, 60)                       # single kernel, 233 taps
print(ps.compare({"pair": fp, "matched": g}))
```

## 8. History of the work (brief)

1. First attempt (GitHub Copilot chat, later deleted by GitHub): a Python KKT script, C++ version and ReadMe in the repo `RobinSchmidt/RS-MET-Research`, `WithAI/Oversampling/Attempt001`. Reconstructed and checked here; its approach was reproduced (−100.4 dB on the reference case).
2. This session: general design routines for `(u, d)` with `L`, attenuation and passband as parameters; a from-scratch matched design; the continuation/joint-optimisation experiments; memory-vs-multiplies comparison; then condensation into the `prosampler` library (6 build steps) with tests and docs; finally the N = 521 experiment that exposed the multi-start issue.
3. Planned home of the library: `WithAI/Oversampling/Attempt002` in the same repo (location may change).
