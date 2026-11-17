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

## 4. Matched design: one kernel for both filters (`design_matched`)

Requiring `u = d = g` makes (N) **quadratic** in `g`: `(g * g)[n0 + L k] = delta[k]`. This is a nonconvex NLP
(minimise the stopband energy of `g`); we solve it with SLSQP from a Kaiser-sinc start, a Gauss-Newton projection
onto the constraint set (so the result is exactly Nyquist) and perturbed restarts. Facts:

* Degrees of freedom: `(L - 2) K / 2` for symmetric `g` with `N = L K + 1` taps. For `L = 2` this is zero: the
  solutions are isolated points, badly conditioned (Jacobian condition about 1e12), and the from-scratch solver finds
  poor ones. For `L = 2` prefer the pair.
* The solver is reliable up to about 60 dB. Beyond that it may stall (e.g. `L = 4`, passband 0.9 near -74 dB although
  -83 dB is achievable). The result is always exactly Nyquist; only its stopband may be worse than attainable.
* `g` has DC gain only approximately `sqrt(L)`: (N) fixes every L-th sample of `g * g`, not the total sum.

### Cost of matching

At equal total length a matched kernel is 20-35 dB worse than the two-kernel pair (it has far fewer degrees of
freedom), so to reach the same attenuation it needs more taps. Measured for `L = 4`, passband 0.85, 60 dB:

| Design | taps | stored coefficients (symmetry, shared) | multiplies per base-rate sample | stopband |
|---|---|---|---|---|
| pair `u`, `d` | 99 + 223 | 161 | 320 | -60.3 / -63.6 dB |
| matched `g` | 237 | 118 | 470 | -61.6 dB |

So matching saves about 27 % of the coefficient memory but costs about 47 % more multiplications. It pays off only
when memory, not arithmetic, is the constraint. (Symmetry is exploited for the stored count and not for the multiplies.)

## 5. Experimental: joint optimisation, rebalancing, tying

* `optimise_joint`: let `u` and `d` both move, objective `E_stop(u) / L^2 + E_stop(d)`, constraints (N) and
  `sum(u) = L`. The gain over the partner design depends on how far `u` is from optimal (about 5 dB worst-case from
  Kaiser-sinc kernels at `L = 4`, 40 dB; small at 100 dB).
* `rebalance`: move taps from `d` to `u` two at a time (total length constant), re-optimising and tracking the
  worst-case stopband, until `n_u = n_d`. This is a continuation path from the pair towards the matched regime.
* `tie`: penalise `|u / sqrt(L) - d sqrt(L)|^2` with increasing weight on an equal-length pair, average, and polish with
  the exact matched solver. Typical cost: tens of dB (L = 4, 40 dB start: about -48 dB at equal lengths, about -28 dB
  after tying). The result is exactly `u = d`.

These routines are research tools; they work on symmetric centred pairs only.

## 6. Swapping the roles

Exchanging `u` and `d` (with a gain fix: new `u = d * L / sum(d)`, new `d = u * sum(d) / L`) leaves `h` and therefore
the identity roundtrip unchanged. With a nonlinearity in the middle the two assignments are **not** equivalent: the
first filter determines what the nonlinearity sees. `FilterPair.swapped()` provides the swapped pair.

## 7. Practical notes

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
