# prosampler - theory and design notes

## 1. The problem

Oversampled waveshaping runs `x -> zero-stuff by L -> u -> f(.) -> d -> keep every L-th -> y`. The nonlinearity `f`
creates harmonics above the original Nyquist frequency; `d` removes them before decimation, `u` removes the images of
the zero-stuffing. Usual designs choose `u` and `d` separately as good lowpass filters. Even then, with `f` the
identity the roundtrip is only approximately transparent: passband ripple, and the images/aliases leaking through the
finite stopbands, change the signal. The question here: can we make the identity roundtrip *exactly* transparent
(to float roundoff) without giving up filter quality?

## 2. The Nyquist condition

Let `h = u * d` (convolution) and `n0 = L * K` (`K` = latency in base-rate samples). With `f` the identity, the
chain output at high-rate position `L (m + K)` is `sum_k x[k] h[L (m + K - k)]`, which equals `x[m]` for every input
iff

    h[n0 + L j] = delta[j]   for all integers j.                       (N)

(The output stream is thus `x` delayed by `K` samples.) This says `h` is an L-th-band (Nyquist(L)) filter: it is 1 at
the centre and 0 at every L-th sample around it; all other samples are free. In the frequency domain (N) reads

    sum_{j=0}^{L-1} H(w - 2 pi j / L) = L exp(-i w n0),

i.e. the aliasing terms of the decimator cancel exactly the images left by the interpolator, independent of the
signal. This L-th-band (Nth-band) property is known from interpolation filters and perfect-reconstruction filter banks
(references below).

Gain convention: `u` has DC gain `L` (it compensates the zero-stuffing), `d` has DC gain 1; `h` then has gain `L` and
(N) holds with a unit centre tap.

### Consequences

* The stopband of `u` and `d` cannot start before `(2 - passband) * pi / L` (rad per high-rate sample): the passband
  edge and the stopband edge are mirror images around the original Nyquist frequency. A wider passband needs a
  narrower transition band and therefore longer filters.
* A windowed-sinc `u` is itself (nearly) Nyquist(L), so `d = delta` is an exact partner: exactness proves nothing
  about quality. The task is to find *good lowpass* kernels `u`, `d` that satisfy (N).
* Linear phase: for symmetric odd-length `u`, `d` the centre of `h` is at `(n_u + n_d - 2) / 2`, which must be a
  multiple of `L`. This sets the latency `K`.
* Condition (N) forces the two end taps of `h` to zero (`h[0] = u[0] d[0] = 0`), so in practice the outer taps of `d`
  (and for matched kernels of `g`) vanish.

## 3. Two-kernel design (`design_pair`, `design_partner`)

For a fixed `u`, (N) is **linear** in `d`: `C d = e`, where row `m` of `C` is a shifted, reversed copy of `u` and
`e` selects the centre. Hence the design of `d` is an equality-constrained least-squares problem

    minimise  || M d - r ||^2   subject to   C d = e

with two possible objectives: a frequency-grid fit (flat passband, zero stopband, optional transition weight) or the
closed-form stopband energy `d^T Q d`. Two solvers: the **null-space method** (SVD of `C`, solution
`d = d0 + Z y`, then unconstrained least squares in `y`) and the **KKT system** `[H C^T; C 0]`. Both reach the same
optimum; the null-space method is more robust because the KKT matrix of long kernels is badly conditioned. A final
minimum-norm correction restores (N) to machine precision. For symmetric `u` and the centred latency `d` is
parametrised as symmetric (half the unknowns, exact linear phase).

**Degrees of freedom.** `C` has about `(n_u + n_d) / L` rows, so `d` keeps roughly `n_d - (n_u + n_d)/L` free
parameters. A decent stopband needs `d` about twice as long as `u`. `design_pair` therefore picks a Kaiser-sinc `u`
meeting the specification and searches the shortest latency for which the partner `d` meets it too.

### 3.1 Minimax objective: a linear program (`objective="minimax"`)

The least-squares objectives minimise energy, but the quoted figure is the worst-case stopband magnitude. For symmetric
centred kernels the zero-phase amplitude `A(w) = sum_n d[n] cos(w (n - mid))` is **linear** in `d`, and so is (N), so

    minimise  t   subject to   C d = e,   -t <= A(w) <= t   on the stopband

is a linear program: convex, one global optimum, no local minima, equiripple solution (`prosampler.minimax`). The
equalities are eliminated through the null space, so (N) holds exactly; HiGHS solves the inequalities on a frequency
grid that is refined Remez-style (the true peaks between grid points are added and the LP re-solved). Extra linear
constraints are available: `transition_max_db` (bound on `|A|` between the passband and stopband edges) and
`passband_flat_db` (the product `A_d A_u / L` stays near 1).

Measured (L = 4):

* Reference case (passband 0.9, 261-tap Kaiser-sinc `u`, 100 dB, `d` with 541 taps): `d` stopband -104.6 dB (grid least
  squares) -> **-114.5 dB** (minimax), 0.5 s -> 15 s; transition bump +1.12 -> +1.15 dB.
* Shorter `d` for the same specification (`design_pair`, `u` unchanged): 115 -> 99 taps (passband 0.8, 40 dB), 215 ->
  191 (0.85, 60 dB), 433 -> 401 (0.9, 80 dB), i.e. about 7-14 % fewer taps in `d`. The gain is limited because the
  Kaiser `u` already sits at the specification and the pair is only as good as its worse member.
* **The bump of `d` at the passband edge cannot be removed by a bound alone.** With `u` fixed, `d` has to compensate the
  roll-off of `u` near the edge (`d(edge) ~ L / u(edge)`, which is ~0 dB), so a bound `transition_max_db` costs
  stopband steeply (passband 0.9, 541 taps, 100 dB start): bound +1.0 dB -> -99.9 dB, +0.5 dB -> -87.1 dB, +0.2 dB ->
  -68.6 dB, <= 0 dB infeasible (rejected immediately). The bump is a property of the split of the transition between `u` and
  `d`, not of the partner objective; shifting it needs `u` to change (see joint minimax in section 5).
* Numerics: the LP level is scaled to the achieved peak and floored at -160 dB (`floor_db`); below that HiGHS's absolute
  tolerances are too coarse and longer kernels could not be exploited anyway.

## 4. Matched design: one kernel for both filters (`design_matched`)

Requiring `u = d = g` makes (N) **quadratic** in `g`: `(g * g)[n0 + L k] = delta[k]`. This is a nonconvex NLP
(minimise the stopband energy of `g`); we solve it with SLSQP from a Kaiser-sinc start, a Gauss-Newton projection
onto the constraint set (so the result is exactly Nyquist) and perturbed restarts. Facts:

* Degrees of freedom: `(L - 2) K / 2` for symmetric `g` with `N = L K + 1` taps. For `L = 2` this is zero: the
  solutions are isolated points, badly conditioned (Jacobian condition about 1e12), and the from-scratch solver finds
  poor ones. For `L = 2` prefer the pair.
* The optimiser finds local minima: the result depends on the starting kernel and is not monotone in `K`.
  `L = 4`, passband 0.9, default Kaiser start (`beta = 4`): -68.0, -72.4, -76.5, -82.7 and -74.5 dB for
  K = 100, 110, 120, 130 and 140; a start with `beta = 2` gives -85.6 dB at K = 140 but only -79.8 dB at K = 130.
  `design_matched` therefore uses several cold starts (`betas`; automatic above 60 dB) plus the warm start, and keeps
  the best. A spec-driven search that only warm-started from failed designs got stuck near -75 dB all the way up to
  K = 596. With multi-start (passband 0.9, L = 4): 70 dB is met at K = 104 (N = 417, -71.4 dB, 3 min), 80 dB at
  K = 124 (N = 497, **-87.3 dB**, 4 min) and 85 dB at K = 138 (N = 553, -85.4 dB, 4 min). Note that K = 124 beats the
  K = 130 result above by 4 dB: a shorter kernel can be better, so the search assumption "longer is better" (bisection)
  only finds *a* length that meets the spec, not necessarily the shortest. The result is always exactly Nyquist; only
  the stopband may be worse than attainable.
* `g` has DC gain only approximately `sqrt(L)`: (N) fixes every L-th sample of `g * g`, not the total sum.

### Cost of matching

At equal total length a matched kernel is 20-35 dB worse than the two-kernel pair (it has far fewer degrees of
freedom), so to reach the same attenuation it needs more taps. Measured for `L = 4`, passband 0.85, 60 dB:

| Design | taps | stored coefficients (symmetry, shared) | multiplies per base-rate sample | stopband |
|---|---|---|---|---|
| pair `u`, `d` | 99 + 215 | 157 | 312 | -60.3 / -60.9 dB |
| matched `g` | 233 | 116 | 462 | -60.2 dB |

So matching saves about 26 % of the coefficient memory but costs about 48 % more multiplications. It pays off only
when memory, not arithmetic, is the constraint. (Symmetry is exploited for the stored count and not for the multiplies.)

## 5. Experimental: joint optimisation, rebalancing, tying

* `optimise_joint`: let `u` and `d` both move, objective `E_stop(u) / L^2 + E_stop(d)` (+ `passband_weight` times the
  passband flatness energy), constraints (N) and `sum(u) = L`. It lowers the stopband ENERGY (to 30-60 % of the start),
  but it does **not** reliably improve the worst stopband PEAK. Because (N) only fixes the product `u * d`, `u` and `d`
  can ripple against each other. Without the passband term the peak improves by up to ~6 dB (L = 4, passband 0.8, 40 dB:
  -41.9 to -47.7 dB) but the per-filter ripple explodes (0.12 to 0.96 dB there; up to 5.9 dB in other cases). With
  `passband_weight = 1` the peak got WORSE in every case tried (passband 0.8 / 0.9 at 40, 60, 80 dB: e.g. -41.9 to
  -38.3, -60.3 to -53.7, -80.0 to -75.7 dB; unchanged at 100 dB), since the far stopband drops (~15 dB) while the
  peak next to the edge rises. No weight beat the plain pair in both peak and ripple. A real gain would need a PEAK
  (minimax) objective that includes passband flatness.
* `rebalance`: move taps from `d` to `u` two at a time (total length constant), re-optimising and tracking the
  worst-case stopband, until `n_u = n_d`. This is a continuation path from the pair towards the matched regime.
* `tie`: penalise `|u / sqrt(L) - d sqrt(L)|^2` with increasing weight on an equal-length pair, average, and polish with
  the exact matched solver. Typical cost: tens of dB (L = 4, passband 0.8, 40 dB start: -47.4 / -49.6 dB at equal lengths 81/81, -28.9 dB
  after tying). The result is exactly `u = d`.

Test at larger size (`L = 4`, passband 0.9, N = 521, i.e. K = 130; start: 361-tap Kaiser-sinc `u` with 100 dB and its
681-tap partner, pair stopband -103.3 / -112.5 dB, ripple 0.0001 dB), run three times with different settings:

| Route | stopband | time |
|---|---|---|
| from scratch, best of 5 Kaiser starts, K = 130 | -82.7 dB | 33 s |
| run 1: old grid, no passband term; rebalance to 521/521, tie | -81.7 dB | 11-14 min + 2-3 min |
| run 2: corrected grid, no passband term; rebalance, tie | -88.2 dB (ripple not checked) | same |
| run 3: corrected grid, passband term on; rebalance path ends 521/521 at -93.9 / -86.2 dB (ripple 0.0002 dB) | averaged: -83.8 dB; exact refinement of the tied `g`: **-82.9 dB** (ripple 0) | path 935 s, total 1140 s |

(In run 1 the from-scratch solver was measured at -84.0 dB on the old grid; runs 2 and 3 use the corrected grid.)
The path stays near -103 dB for about three steps and then degrades to about -86 dB. With the passband term on, tying
gives -82.9 dB versus -82.7 dB from scratch: **no reliable advantage**, at 30 times the cost. Run 2's -88.2 dB was
probably helped by unconstrained passband ripple, which is why it is not reproduced with the passband term.

**Joint minimax (`optimise_joint_minimax`).** The LP is symmetric in the two kernels: for a fixed `d` the same program
designs `u`. Alternating `d`-steps and `u`-steps keeps both stopband PEAKS non-increasing (each step starts from a
feasible point) and the Nyquist condition exact; each filter's passband ripple is bounded (`ripple_db`, default twice the
input's, at least 0.001 dB), which prevents `u` and `d` from rippling against each other as in the energy-based joint
optimisation above. The joint problem is bilinear, so the result is a local optimum that depends on the start. Measured
(Kaiser `u` + grid partner as start, lengths unchanged, L = 4):

| passband / spec | lengths | start `u` / `d` stopband | after joint minimax |
|---|---|---|---|
| 0.8 / 40 dB | 47/115 | -41.9 / -45.0 dB | -42.4 / -57.4 dB |
| 0.85 / 60 dB | 99/215 | -60.3 / -60.9 dB | -61.7 / -76.7 dB |
| 0.9 / 40 dB | 91/215 | -40.1 / -41.8 dB | -41.1 / -54.8 dB |
| 0.9 / 60 dB | 147/319 | -60.3 / -60.9 dB | -61.5 / -75.0 dB |

Ripple is unchanged (flatness bound), `u` gains only 1-1.4 dB, and `d` gains 12-17 dB: at the usual 1 : 2 split `d` is
simply too long, and the worst peak stays that of `u`. The gain comes from **moving taps from `d` to `u`** at constant
total length (the minimax analogue of `rebalance`): at 0.85 / 60 dB with 314 taps the split 117/197 reaches
**-63.7 / -63.8 dB** against -60.3 dB for the plain 99/215 pair (that run started from Kaiser `u` kernels designed for 75 dB; starting from 60 dB designs gave
-63.0 dB at 109/205: the optimum is start dependent). Shortest total length meeting the
specification (best of several splits, `design_pair(..., objective="minimax")` + `optimise_joint_minimax`):

| passband / spec | plain pair | minimax partner only | joint minimax, best split |
|---|---|---|---|
| 0.8 / 40 dB | 47/115 = 162 taps | 47/99 = 146 | **45/93 = 138** (-40.4 dB) |
| 0.85 / 60 dB | 99/215 = 314 taps | 99/191 = 290 | 99/191 = 290 (-62.4 dB; 282 misses at -59.5) |

So most of the saving (5-10 % of the total taps, 7-14 % of `d`) comes from the minimax partner; joint minimax adds margin
(1-3 dB at equal length) and, at 40 dB, a few taps more. Each joint search costs minutes at 60 dB.

**Simultaneous joint minimax (`optimise_joint_slp`).** The alternation above cannot move the split of the transition
roll-off between `u` and `d`: for a fixed partner the Nyquist rows nearly force the other kernel's transition band (with the
transition cap of the previous experiment `u` stayed identical and `d` just paid in stopband). Moving both kernels in the
same step needs the bilinear coupling, handled by sequential linear programming (`prosampler.jointlp`): linearise
`conv(u, d)[n0 + L k] = delta[k]` at the current feasible point, add a trust region, solve one LP over `(da, db, t)` with
everything else (stopband peak, passband flatness, transition constraints) exactly linear, restore exactness by
Gauss-Newton, accept if the merit decreases. The limit is a local optimum; one iteration is an LP of a few seconds.

Measured, equal lengths and the **passband ripple held at the start pair's value** (the ripple is a free resource: every
doubling of the allowed ripple buys about 3 dB of stopband and enlarges the bump, e.g. 0.8 / 40 dB: -48.5 / -51.5 / -55.3 dB
at 1x / 2x / 4x ripple with bumps +1.7 / +2.5 / +4.0 dB, which is why results are only comparable at equal ripple):

| `transition_max_db` (cap on `u` and `d` between the band edges) | 0.8 / 40 dB, 47/115 taps, ripple 0.124 dB | 0.85 / 60 dB, 99/215 taps, ripple 0.016 dB |
|---|---|---|
| plain pair (start) | -41.9 dB, bump +1.25 | -60.3 dB, bump +1.17 |
| no cap | -48.5 dB, bump +1.69 | -69.1 dB, bump +1.72 |
| +1.0 dB | -45.3 dB | -65.7 dB |
| +0.5 dB | -42.7 dB | -63.2 dB |
| +0.2 dB | -40.9 dB | -61.5 dB |
| 0 dB (no bump at all) | -37.8 dB | -58.8 dB |

So at equal lengths and equal ripple the simultaneous step gains 7-9 dB with the bump left alone, still **gains 1-3 dB with a
bump of at most +0.5 dB**, and an essentially bump-free pair (+0.2 dB) costs about 1 dB (0.8 / 40) or gains 1 dB
(0.85 / 60) against the plain pair. The roll-off split moves toward the matched case: at the original Nyquist frequency
(`x = 1/2`, the product is -6 dB) `u` / `d` were -5.9 / 0.0 dB for the plain 0.85 / 60 pair, -5.1 / -0.8 dB with cap +0.5 dB
and -3.9 / -2.0 dB with cap 0 dB (matched kernels: -3 / -3 dB). Times: 10-40 s (47/115 taps) and 30-170 s (99/215 taps).

**Shortest pair for a specification (`search_pair`).** Combining the minimax partner, the simultaneous move and a transition
cap: for each total length (reduced stepwise from the plain pair's) and several splits `n_u / n_d`, start from a Kaiser-sinc `u`
of `n_u` taps with its minimax partner and refine with `optimise_joint_slp`, passband ripple held at the plain pair's value.
Results (L = 4; local search, shortest length *found*; MACs per base-rate sample are about the total number of taps):

| passband / spec | plain pair | minimax partner only | joint SLP, cap +0.5 dB | joint SLP, cap +0.2 dB |
|---|---|---|---|---|
| 0.8 / 40 dB (ripple 0.124 dB) | 47/115 = 162 taps (81 stored) | 47/99 = 146 | **47/83 = 130** (65 stored, -40.7 dB) | **55/83 = 138** (69 stored, -41.8 dB) |
| 0.85 / 60 dB (ripple 0.016 dB) | 99/215 = 314 taps (157 stored) | 99/191 = 290 | 99/183 = 282 (141 stored, -62.7 dB) | 99/183 = 282 (-60.7 dB); fine search: **103/171 = 274** (137 stored, -62.5 dB, bump +0.50 dB) |

That is 15-20 % fewer taps (0.8 / 40) and about 10 % (0.85 / 60) than the plain pair, with a bump of at most the cap (plain: +1.2 dB).
The coarse 60 dB search (steps of 16 taps) gave 282. The fine search (`resolution="fine"`, steps of 8 taps, split step 4, window) found 274 taps (103/171) at the reference ripple of 0.016 dB; at 266 taps the best of 7 splits reaches only -59.6 dB, so the minimum is probably 266-274 (a split step of 2 around 95-111 at 266 taps gave at best -59.6 dB; the splits 95-105 are all within 0.3 dB). An earlier fine run reported 266 (95/171), but with a passband ripple of 0.027 dB, i.e. not at equal ripple; the acceptance test now always compares against the reference ripple. The strict run took 16 min on two cores (about 1 min per configuration at ~300 taps). `strategy="adaptive"` (secant steps on the length with a bracket, narrow split window) reached the same 274 taps (103/171) with 27 instead of 62 configurations, 18 instead of 50 minutes. With a tighter bump cap of +0.2 dB the adaptive fine search also ends at 274 taps (103/171, -60.8 dB; 99/175 also meets at -60.6 dB), i.e. the tighter cap costs 1.7 dB of margin but no length step here; at 266 the best split reaches -58.6 dB and at 258 -56.6 dB. Rerunning the best three splits of each failing length from differently windowed starts (`rescue`, Kaiser design for the attenuation ±6 dB) did not improve these values, so the 266-tap miss is not a start-point artefact of the SLP. With the bump removed entirely (`transition_max_db=0`) the adaptive fine search ends at 290 taps (107/183, -60.8 dB, bump +0.00 dB); 282 misses at -59.2 dB and 274 at -57.5 dB. So the bump costs length in steps: plain pair (bump +1.2 dB) 314, cap +0.5 dB 274, cap +0.2 dB 274, cap 0 dB 290 taps.

**Prescribed transition shapes (`TransitionShape`).** The product of the two transition amplitudes must be point-symmetric
(`r(x) + r(1 - x) = 1`, see `prosampler.transition`); `1 - x^p` is not admissible for `p != 1`, the raised cosine `cos^2`, the
linear ramp and the integral of a Kaiser-window main lobe are. A split exponent `alpha` gives `A_u / L = r^alpha`,
`A_d = r^(1 - alpha)` (alpha = 1/2 is the matched root-Nyquist case). As a tube of half-width `tol` this is linear and
available in `optimise_joint_slp(shape=...)`, but it turned out too restrictive for the usual short `u` / long `d`: alpha -> 1
asks `d` to stay near 0 dB until the stopband edge (a 115-tap `d` cannot), and a cosine tube with tolerance 0.05 stalled at
-22.7 dB (alpha 0.75) and -31.8 dB (alpha 0.5) against -41.9 dB for the plain pair (0.8 / 40 dB, 47/115 taps; run with a
4x ripple allowance). The measured product of the plain pair is also steeper than `cos^2`. The simple gain cap above gives
the practical control, so shapes remain a research option.

These routines are research tools; they work on symmetric centred pairs only.

## 6. Swapping the roles

Exchanging `u` and `d` (with a gain fix: new `u = d * L / sum(d)`, new `d = u * sum(d) / L`) leaves `h` and therefore
the identity roundtrip unchanged. With a nonlinearity in the middle the two assignments are **not** equivalent: the
first filter determines what the nonlinearity sees. `FilterPair.swapped()` provides the swapped pair.

## 7. Practical notes

* **Signal level at the nonlinearity.** The level at the waveshaper is `sum(u)/L` times the input level, i.e. 1 for the
  pair convention but `1/sqrt(L)` for a matched kernel. Compensate with scalar gains around the waveshaper
  (`FilterPair.shaper_level`); otherwise the same distortion curve acts on a different signal level than with a pair.
* **Measuring the stopband.** The stopband peak of an optimised design sits at the stopband edge. All measurements
  include the exact edge frequency on the grid; without it, the reported attenuation depended on the grid size
  (up to a few dB).

* Aliasing from `f` is suppressed only by the stopband attenuation of `d`; the exact-identity property concerns the
  linear signal path, not the nonlinear products.
* Bandwidth: with passband fraction `p`, content above `p * Nyquist` is attenuated by `d` even without nonlinearity;
  choose `p` accordingly.
* Latency is `K` base-rate samples (`L * K` high-rate samples) and a pure delay for linear-phase kernels.

## 8. Literature

Closest classical relatives (the search was not exhaustive; I am not aware of a reference that treats exactly this
up-/down-pair with a nonlinearity in between):

* F. Mintzer, "On half-band, third-band, and Nth-band FIR filters and their design", IEEE Trans. Acoustics, Speech,
  and Signal Processing, 1982, DOI 10.1109/TASSP.1982.1163950. Nth-band filters, i.e. condition (N) for one filter.
  (Author, year and DOI verified against the publisher-linked record; volume/pages not checked.)
* F. Mintzer, "Filters for distortion-free two-band multirate filter banks", IEEE Trans. Acoustics, Speech, and Signal
  Processing, vol. 33, no. 3, pp. 626-630, June 1985, DOI 10.1109/TASSP.1985.1164587. Perfect-reconstruction filter
  banks; (N) with `L = 2` is the product-filter condition. (Verified against the publisher-linked record.)
* P. P. Vaidyanathan, *Multirate Systems and Filter Banks*, Prentice Hall, 1993: Nth-band filters and perfect
  reconstruction in M-channel banks. (From memory, not re-verified.)
* R. E. Crochiere, L. R. Rabiner, *Multirate Digital Signal Processing*, Prentice Hall, 1983: interpolation and
  decimation filter design. (From memory, not re-verified.)
