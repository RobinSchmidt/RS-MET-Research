# Oversampling filter design for roundtrip

> Exported from **claude.ai** on 2026-11-16 03:36:14 · 8 messages · ~2,995 words · ~15 min read · [source](https://claude.ai/chat/b98c9ff2-aaed-467d-a295-d17944ea98bf)

---

## You

I write audio signal processing algorithms for musical purposes. In this context, it is often necessary to use oversampling, i.e. to first upsample an incoming signal to a higher sample rate, then do the actual processing at that higher rate, then downsample it to the original sample rate again. Both, up- and downsampling steps involve lowpass filtering the signal. In the case of upsampling, the lowpass is called anti-imaging lowpass, in the case of downsampling, it's called anti-aliasing lowpass. What I want to create is a pair of FIR upsampling and downsampling filter kernels (let's denote them by u[n] and d[n] respectively), such that an upsampling-downsampling roundtrip is an exact identity operations (up to floating point roundoff errors). For concreteness, let's assume I want to implement a waveshaping distortion effect. So, the processing would be:

  input -> upsample -> waveshaper -> downsample -> output

The goal is that my oversampled distortion algorithm is completely transparent as long as the waveshaper that sits in between the upsampler and the downsampler is itself an identity function (i.e. does nothing). Is there any known literature for this particular type of filter design problem?

---

## Claude

### Claude responded: Yes, this is well-studied, though usually under different names.

Yes, this is well-studied, though usually under different names. The condition you want is that the cascade (upsample by L, then downsample by L) is the identity, and that is a **perfect-reconstruction (PR) condition from multirate filter bank theory**.

### The math

With upsampling by L (zero-stuffing, then filtering with u[n]) and downsampling by L (filtering with d[n], then decimating), the roundtrip is the identity iff the composite filter h = u * d (convolution) satisfies

h[Ln] = δ[n − n₀]

for some delay n₀. In words, h is a **Nyquist(L) filter**, i.e. an L-th band filter, or “zero-ISI” filter in communications terminology. Every L-th sample of h is zero except the one at the center (or at a delay). In the frequency domain this means that the L aliased copies of H(e^{jω}) sum to a constant.

Note that this is a condition on the *product* H(z) = U(z)D(z), not on each filter separately. So the design problem splits into two steps:

1. Design a Nyquist(L) filter H(z).
2. Factor it into U(z)·D(z).

### Where to look in the literature

* **Vaidyanathan, *Multirate Systems and Filter Banks* (1993)**: the standard reference. Look at the chapters on L-th band (Nyquist) filters, the polyphase representation, and perfect-reconstruction filter banks. The PR condition for a polyphase decomposition is exactly what you describe.
* **Nyquist / zero-ISI pulse shaping** from digital communications: raised cosine and root-raised-cosine filters are the classic example. A root-raised-cosine u with d = u (matched filter) gives h = raised cosine, which is a Nyquist filter. This is the closest textbook analogue to your problem, since the “matched pair whose cascade is Nyquist” is the central idea of matched-filter pulse shaping.
* **Mintzer, “Filters for distortion-free two-band multirate filter banks” (IEEE Trans. ASSP, 1985)** and related work on L-th band filters: early papers on designing half-band and L-th band filters.
* **Nguyen & Vaidyanathan, “Two-channel perfect-reconstruction FIR QMF structures which yield linear-phase analysis and synthesis filters”**, and Nguyen’s work on **cosine-modulated and L-th band filter design**.
* **Lagrange / sinc interpolation viewpoint**: a windowed-sinc kernel with zeros at multiples of L is an interpolating (Nyquist) filter by construction. Any “interpolating” kernel (h[Ln] = δ[n]) has this property.
* **Polyphase / Smith, J.O.**, *Digital Audio Resampling Home Page* (on his CCRMA site): practical audio-oriented treatment of bandlimited interpolation and polyphase implementation.
* **Parks–McClellan / least-squares designs of Nyquist filters**: constrained-optimization approaches (e.g. linear programming or constrained least squares with the zero-crossing constraints) are covered in the filter-design literature for L-th band filters.

Useful search terms: *L-th band filter*, *Nyquist filter design*, *interpolated FIR with zero crossings*, *perfect reconstruction multirate*, *matched interpolation/decimation filter pair*, *root-Nyquist filter*.

### Practical design routes

**1. Root-Nyquist approach (symmetric choice).** Design a Nyquist(L) lowpass H with a non-negative frequency response (e.g. a raised-cosine-like or windowed design with a positive spectrum), then take its spectral factor: U = D = H^{1/2} (in the magnitude sense, via cepstral or Fejér–Riesz-style factorization). Because the response is non-negative, a real spectral square root exists. This gives a matched pair, and a linear-phase result if you take the square root of the magnitude with zero-phase (then h = u * d is linear-phase and u = d is the symmetric “root” filter).

**2. Direct factorization of a designed Nyquist H.** Design H (L-th band, linear phase, good stopband). Then split its zeros between U and D, for example by assigning zeros inside the unit circle to one and outside to the other (min/max-phase split), or by assigning conjugate pairs. This gives more freedom (e.g. a short, low-latency u and a different d) but loses linear phase in the individual filters unless you keep zeros in symmetric groups.

**3. Joint optimization.** Parametrize u and d directly and minimize stopband energy of both, subject to the *bilinear* constraint (u*d)[Ln] = δ[n − n₀]. This is nonconvex but small in practice; alternating optimization (fix u, solve the linear problem for d, then swap) works well.

### One caveat worth knowing

Your identity holds only if the roundtrip passes the signal without modification at the *original* rate. The Nyquist condition guarantees no distortion of the passband content up to the L-fold aliasing structure, but a *good* anti-aliasing filter wants H to be small above the original Nyquist frequency, and an exact Nyquist(L) filter must have H(e^{jω}) + (aliased copies) = const. So the transition band is constrained: it must be **anti-symmetric around π/L** (like a raised-cosine rolloff). The cost is that the passband edge has to sit below π/L by the transition width. You cannot get both a brick-wall response and exact identity. Higher oversampling headroom (the usable audio band being well below the original Nyquist) is what buys you stopband attenuation, and the audio-bandwidth margin is the real design parameter.

If you tell me L, your target passband edge, the stopband attenuation you want, and whether linear phase or low latency matters more, I can sketch a concrete design (e.g. in Python/scipy) for the pair.

---

## You

Can you give me both versions (low latency and linear phase) and let the oversampling factor L, the desired attenuation and passband width be user-adjustable parameters that I can set later, i.e. pass as parameters into the design routine?

---

## Claude

### Claude responded: I'll write the design routine, run it to check both versions, and send you the file.

Fitting the low-latency filter with alternating least-squares refinement

I’ll write the design routine, run it to check both versions, and send you the file. I’ll also correct one citation from my earlier answer at the end.

Created a file, updated memory, and 7 more stepsImports are fast, so I'll time the individual steps.

The initial windowed root-raised-cosine reaches only −20 dB of stopband attenuation, which points to a bug in the initial filter, not the LS solve. I’ll inspect it directly.

Ran 2 commands

The Kaiser estimate has 10–30 dB of margin here, so I’ll switch the initial filter to a narrower ideal transition (κ=0.2) and replace the unbounded loops with a bisection on filter length.

Created a file, shared a file, and 28 more stepsOutput was lost when the timeout killed the pipe, so I'll log to a file with unbuffered output.
The rewrite was rejected because the file had changed since I last read it, so I'll read it and write again.
Linear mode works and is exact. The low-latency search fails for L=2, so I'll look at why.

I’ve built and tested `oversampling_pair.py`, but the “low latency” version does not actually have lower latency than the linear-phase one. The roundtrip is exact to ~1e-15 for every setting I ran.

python

```python
from oversampling_pair import design_oversampling_pair, roundtrip
u, d, info = design_oversampling_pair(L=4, atten_db=100, passband=0.9, mode="linear")  # or "lowlatency"
y = roundtrip(x, u, d, L=4, K=info["K"], shaper=np.tanh)   # reference implementation
```

* **Parameters:** `L`, `atten_db`, and `passband` are plain arguments. `passband` is a fraction of the original Nyquist frequency, so 0.9 means 19.8 kHz at 44.1 kHz. The stopband starts at the mirror image, `2 − passband`, which the Nyquist condition forces.
* **Gain convention:** u has passband gain L and d has gain 1. With the identity in between, the chain returns `x[m−K]`, where `K` is in `info["K"]`.

### How it works

u is fixed first, and d is then solved as a least-squares problem with the Nyquist condition imposed as a hard constraint. For fixed u that condition is linear in d, so it holds to machine precision. The solver also searches for the shortest d (linear mode) or smallest `K` (low-latency mode) that meets your attenuation, and checks that both u and d meet it.

* **Linear:** u is a Kaiser-windowed sinc, and u and d are both exactly linear phase.
* **Low latency:** u is the minimum-phase square-root factor of a sinc, so d is not linear phase.

### Test results

| L, atten, passband | mode | K | len(u) / len(d) | stopband u/d (dB) | roundtrip error |
| --- | --- | --: | --- | --- | --- |
| 2, 90, 0.9 | linear | 94 | 119 / 259 | −90.0 / −91.5 | 9e-16 |
| 2, 90, 0.9 | lowlatency | 88 | 176 / 440 | −90.1 / −91.0 | 1e-15 |
| 4, 100, 0.9 | linear | 100 | 261 / 541 | −100.2 / −102.8 | 2e-15 |
| 4, 100, 0.9 | lowlatency | 108 | 430 / 1075 | −100.4 / −102.8 | 1e-15 |
| 8, 120, 0.85 | linear | 79 | 423 / 843 | −120.2 / −123.7 | 2e-15 |
| 8, 120, 0.85 | lowlatency | 89 | 711 / 1778 | −120.0 / −123.7 | 9e-16 |

### Two things you should know

1. **d is about twice as long as u.** The Nyquist constraints use up roughly `(len(u)+len(d))/L` of d’s degrees of freedom. For L=2 with equal lengths there is nothing left to shape its stopband. I couldn’t get equal lengths with exactly linear-phase filters.
2. **The low-latency mode did not reduce latency at full spec** (K = 88/108/89 against 94/100/79 for linear).
  * My first runs showed K≈27 for L=4, but that came from a d with a +117 dB gain bump in the transition band. The design now enforces `max_transition_gain_db=3` to prevent that.
  * You can still pass `latency=K` explicitly to see what attenuation is reachable. For L=4, passband 0.9, I measured about −75 dB at K=100, −44 dB at K=80, and −20 dB at K=45.
  * The real levers on latency are a wider transition band (smaller `passband`) and lower `atten_db`.

If you need genuinely low latency, this exact construction probably isn’t the tool. I haven’t ruled out other approaches (non-minimum-phase u, optimising both filters jointly), but I didn’t find one here.

### Correction to my earlier answer

I cited Mintzer’s two-band paper as 1985. It is from 1977 (IEEE Trans. ASSP). The relevant Mintzer paper for L-th band filters is “On half-band, third-band, and Nth-band FIR filters and their design” (IEEE Trans. ASSP, 1982). I recalled these from memory and didn’t check them against a source.

Files are in /home/claude:

* oversampling_pair.py

Oversampling pairPY4 hours ago

---

## You

OK. Very nice work! Thank you very much. This is actually my second attempt at tackling the problem. Previously, I did it with my GitHub Copilot subscription (also using Sonnet 5.5 at medium effort). As I wanted to look up this old chat today, I had to discover that Copilot deletes old chats after a while - so I unfortunately lost it (which is quite frustrating). That's why I was now trying to reconstruct the solution you gave me back then. I fortunately did save the Python script that was generated. It is this one here:

[https://github.com/RobinSchmidt/RS-MET-Research/blob/master/WithAI/Oversampling/Attempt001/oversampling_pr_design.py](https://github.com/RobinSchmidt/RS-MET-Research/blob/master/WithAI/Oversampling/Attempt001/oversampling_pr_design.py)

It is much shorter than the result you produced now. I think, in that attempt, the parametrization was not in terms of L, attenuation, passband but rather in terms of L and the two lengths directly. Is this the reason why it is so much shorter? Is this old code even correct? Copilot said that it didn't test it (because it can't run code). In the same directory, you will find also a C++ version and a ReadMe.txt with some notes about this first attempt. Copilot also said something about a KKT (Karush-Kuhn-Tucker) system. Can you reconstruct what we did back then? Is this the same approach that you did here or is this different?

---

## Claude

### Claude responded: I'll read your old script first, then the ReadMe and the C++ version from the same directory.

Checking which GitHub file links can be fetched

I’ll read your old script first, then the ReadMe and the C++ version from the same directory.

Read 2 pages, updated tasks

The page fetch only gave me a summary, and the directory listing wasn’t allowed. I’ll try pulling the raw files from GitHub through the shell instead.

Used Add repo, read 3 files, and 8 more steps

Your old script is correct, and it uses the same approach as my new routine. I can’t see the lost Copilot chat, so I’m working only from your script, the C++ version and the ReadMe. I ran the Python and the C++ version; both round-trip to about 1e-15.

### What the old code does

* **`design_u`:** a Kaiser-windowed sinc with cutoff π/L and gain L. That is the same u as my `mode="linear"`.
* **`design_d`:** d is chosen to minimise its stopband energy subject to the exact Nyquist condition `(u*d)[c + L·n] = δ[n]`, where `c` is the centre of `u*d`. This is the problem I solve in `_solve_d`.
* **`stopband_matrix`:** the integral ∫cos((i−j)ω)dω over the stopband, in closed form. That gives d’s stopband energy as dᵀQd, which holds for any real d.

### The KKT system

You want to minimise ½·dᵀQd subject to C·d = e, where C is the stride-L rows of the convolution matrix of u. Introduce Lagrange multipliers λ for the constraints and set the gradient of the Lagrangian to zero. Together with feasibility, this gives one linear system:

```
[ Q  Cᵀ ] [ d ]   [ 0 ]
[ C  0  ] [ λ ] = [ e ]
```

The old code solves this directly. I instead eliminate the constraints with a null-space parametrisation, d = d₀ + Z·y, and solve an unconstrained least-squares problem in y. Mathematically these are the same optimum.

### Why the old script is shorter

Partly the parametrisation: it takes `L`, `N_U`, `N_D` and `TRANSITION` directly. My routine takes `L`, attenuation and passband, so it has to search for the lengths automatically. It also has the minimum-phase mode and diagnostics. The core, meaning u, d and the roundtrip, is about the same size in both.

The other differences are in the objective and the numerics:

* **Objective:** the old version minimises only d’s stopband energy. A flat passband follows from the Nyquist constraint, so that was enough. Mine samples a frequency grid and also has passband and transition-band terms.
* **Numerics:** the old stopband-energy matrix Q has a condition number of about 3e12, regularised with 1e-12·I. It still held the constraints to about 5e-15 at the sizes I tried, but there is no refinement step. My null-space solve with refinement is safer when u is minimum-phase.
* **Quality:** at the same lengths (L=4, Nu=261, Nd=541, passband 0.9) the old KKT method gives d at −100.4 dB and mine gives −102.8 dB. So you lose essentially nothing.

### Things to know about the old script

* **Misleading u stopband figure:** it prints u’s stopband in absolute dB, without dividing by the gain L. The printed −2.5 dB is really −14.5 dB relative to the passband.
* **Poor default sizes:** the defaults (33 and 65 taps for L=4) are a demo and give only about −14.5 dB for u and −25 dB for d. For a given attenuation you need the lengths I mentioned earlier.
* **Weak length guard:** `len(idx) >= n_taps` only catches a degenerate case. Nothing is left to optimise well before that point, and d needs about twice the length of u.
* **Why 8L+1 and 16L+1:** those lengths make the centre `c` a multiple of L, so the latency is a whole number of base-rate samples (12 here).

The ReadMe lists “Why is the downsampling kernel roughly twice as long as the upsampling kernel?” as a follow-up. This session’s analysis answers it. The constraints use up about `(len(u)+len(d))/L` of d’s degrees of freedom, so d needs the extra length to keep any freedom to shape its stopband. The old script’s `N_D = 2·N_U − 1` matches what I found.

The ReadMe also quotes a statement about alternating joint optimisation of u and d. When I tried a naive version of that this session, it stalled with d at about −25 dB, so it isn’t a free improvement.

If you’d like, I can work through the other follow-up questions from your ReadMe, including the matched `u = d = g` filter, in this chat. That would help you export them into the repo.

2 hours ago

---

## You

Yes. That would be very nice. It's really sad that the old chat was lost. I didn't know that Copilot deletes them after a while and have now learned it the hard way.

---

## Claude

### Claude responded: I'll build and test the matched-filter design (the u = d question) first, since several of the other answers depend on what it shows.

Writing a numpy implementation of the symmetric filter design

I’ll build and test the matched-filter design (the `u = d` question) first, since several of the other answers depend on what it shows. I’ll also check which libraries exist for this.

Searched the web, updated tasks, ran a command2 hours ago

---

*Exported with [Inkpour](https://github.com/tronicum/inkpour)*
