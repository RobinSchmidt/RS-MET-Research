# Exactly transparent FIR oversampling filter pairs for waveshaping — a research summary

*Written up 2026-11-17 from a working session. Companion code: the `prosampler` Python package (`prosampler.zip`, docs in `docs/usage.md` and `docs/theory.md`). Numbers below were measured in the session unless marked otherwise.*

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
* Reference case L = 4, `u` = 261-tap Kaiser-sinc (100 dB), 541-tap `d`, latency 100:
  * energy objective, KKT (as in the first attempt's script): **−100.4 dB** — the first attempt's result was reproduced;
  * grid objective: −102.0 dB (−102.8 dB in an earlier routine that also used a transition weight).
* Spec-driven search (L = 4, passband 0.85, 60 dB): 99-tap `u`, 215–223-tap `d`, both ≈ −60 / −64 dB.
* **Low-latency (non-linear-phase) variants turned out useless** in the early experiments; the library is linear-phase only.
* `d` may bump by ≈ 1 dB in the transition band; `transition_weight` (≈ 0.3) suppresses it. (An early low-latency `d` had a +117 dB bump — hence the limit of 3 dB in the spec check.)

### 3.2 Matched kernel (`design_matched`)

* At equal total length a matched kernel is **20–35 dB worse** than the pair (far fewer degrees of freedom), so for equal attenuation it needs more taps.
* **Memory vs multiplies** (L = 4, passband 0.85, 60 dB):

  | Design | taps | stored coefficients (symmetry, shared) | MACs per base-rate sample | stopband |
  |---|---|---|---|---|
  | pair `u`, `d` | 99 + 223 | 161 | 320 | −60.3 / −63.6 dB |
  | matched `g` | 237 | 118 | 470 | −61.6 dB |

  Matching saves ≈ 27 % coefficient memory but costs ≈ 47 % more multiplies; worthwhile only if memory, not arithmetic, is the constraint.
* **Local minima.** The result depends on the starting kernel and is **not monotone in `K`**. L = 4, passband 0.9, default Kaiser start (`beta = 4`):

  | K | 100 | 110 | 120 | 130 | 140 |
  |---|---|---|---|---|---|
  | stopband (dB) | −68.9 | −73.3 | −77.6 | −84.0 | −75.7 |

  At K = 140 a start with `beta = 2` gives −86.7 dB. A spec-driven search that only warm-started from the previous *failed* design got stuck at ≈ −74.6 dB all the way to K = 596 (505 s, spec missed). The documented "plateau at −74 dB" was this warm-start trap, not a solver limit.
* **Fix:** multi-start (several Kaiser betas plus the warm start, keep the best; automatic above 60 dB). With it: 70 dB at K = 103 (N = 413, −70.8 dB, 40 s); 80 dB at K = 123 (N = 493, −80.5 dB); 85 dB at K = 133 (N = 533, −86.8 dB); the latter two take ≈ 4.5 min each.

### 3.3 Joint optimisation, rebalancing, tying (experimental)

* **Joint optimisation** of `(u, d)` (balanced stopband objective; exact Nyquist; `sum(u) = L`): the gain depends on how suboptimal the starting `u` is. L = 4, passband 0.8, 40 dB, Kaiser start (47/115 taps): worst stopband −41.9 → −47.2 dB (≈ 5 dB). At 100 dB starting from a good `u` the gain is small. *(An earlier claim of a larger benefit was measured against an untuned start and was overstated.)*
* **Rebalancing** (move taps from `d` to `u` two at a time, total length constant, re-optimise): 47/115 → 81/81 gives −48.0 / −49.9 dB.
* **Tying** (penalty continuation, average, exact matched polish): −28 dB for the same example — tying costs 20–35 dB at equal total length.
* **Continuation at larger size** (L = 4, passband 0.9, N = 521, start 361/681 taps, 100 dB):

  | Route | stopband | time |
  |---|---|---|
  | rebalance path, equal lengths | −91.0 / −95.8 dB (u / d) | 14 min |
  | after tying + exact refinement (`u = d`) | −81.7 dB | — |
  | from scratch, default start | −84.0 dB | 7 s |
  | from scratch, `beta = 2` (at K = 140) | −86.7 dB | 12 s |

  The path itself looks attractive (≈ −103 dB for the first ~10 steps, ≈ −90 dB mid-path), but the final projection onto an exact `u = d` solution loses most of the gain. **At this size continuation is not better than a good from-scratch start**: the exact-matching constraint, not the starting point, is what costs attenuation.

### 3.4 Swapping roles

Exchanging `u` and `d` (with a gain fix: new `u = d·L/sum(d)`, new `d = u·sum(d)/L`) leaves `h` and the identity roundtrip unchanged. With a nonlinearity in between the two assignments are **not** equivalent (the first filter determines what `f` sees). For matched kernels swapping is meaningless.

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
5. **Why does continuation + tying lose the gain?** Study the geometry of the feasible set `{g*g Nyquist}`; try tying with a trust-region/Riemannian method instead of penalty + projection, or tie early in the path.
6. **Non-symmetric / low-latency filters.** The `partner` design supports non-symmetric `u` with explicit latency; a systematic study of minimum-phase or low-latency pairs (with nonlinear-phase side effects on the waveshaper) is unexplored — the early low-latency attempt was dropped as useless, but that was a narrow experiment.
7. **Transition band shaping.** Trade-offs between `d`'s transition-band bump, `transition_weight`, and attenuation; a principled specification of the transition-band behaviour.
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
docs/usage.md, docs/theory.md; tests: 56 passing (docs examples are executed by the test suite)
```

Minimal usage:

```python
import prosampler as ps
fp = ps.design_pair(L=4, passband=0.85, atten_db=60)    # u (99 taps), d (~223 taps)
fp.verify()
g = ps.design_matched(4, 0.85, 60)                       # single kernel, 237 taps
print(ps.compare({"pair": fp, "matched": g}))
```

## 8. History of the work (brief)

1. First attempt (GitHub Copilot chat, later deleted by GitHub): a Python KKT script, C++ version and ReadMe in the repo `RobinSchmidt/RS-MET-Research`, `WithAI/Oversampling/Attempt001`. Reconstructed and checked here; its approach was reproduced (−100.4 dB on the reference case).
2. This session: general design routines for `(u, d)` with `L`, attenuation and passband as parameters; a from-scratch matched design; the continuation/joint-optimisation experiments; memory-vs-multiplies comparison; then condensation into the `prosampler` library (6 build steps) with tests and docs; finally the N = 521 experiment that exposed the multi-start issue.
3. Planned home of the library: `WithAI/Oversampling/Attempt002` in the same repo (location may change).
