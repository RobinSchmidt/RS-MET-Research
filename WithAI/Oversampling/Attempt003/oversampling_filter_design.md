# Oversampling filter design

**User:** @RobinSchmidt

## @RobinSchmidt

I write audio signal processing algorithms for musical purposes. In this context, it is often necessary to use oversampling, i.e. to first upsample an incoming signal to a higher sample rate, then do the actual processing at that higher rate, then downsample it to the original sample rate again. Both, up- and downsampling steps involve lowpass filtering the signal. In the case of upsampling, the lowpass is called anti-imaging lowpass, in the case of downsampling, it's called anti-aliasing lowpass. What I want to create is a pair of FIR upsampling and downsampling filter kernels (let's denote them by u[n] and d[n] respectively), such that an upsampling-downsampling roundtrip is an exact identity operations (up to floating point roundoff errors). For concreteness, let's assume I want to implement a waveshaping distortion effect. So, the processing would be:

  input -> upsample -> waveshaper -> downsample -> output

The goal is that my oversampled distortion algorithm is completely transparent as long as the waveshaper that sits in between the upsampler and the downsampler is itself an identity function (i.e. does nothing). Is there any known literature for this particular type of filter design problem?

## Copilot

Yes—this is a well-known problem. The keywords you want are **perfect-reconstruction (PR) multirate filter design**, **Nyquist(\(L\)) FIR filters**, and **root-Nyquist pairs** (for oversampling factor \(L\)).

For your chain
\[
x \rightarrow \uparrow L \rightarrow u[n] \rightarrow (\text{identity}) \rightarrow d[n] \rightarrow \downarrow L \rightarrow y,
\]
a sufficient/standard PR condition is that the combined high-rate impulse response \(h[n]=(u*d)[n]\) satisfies
\[
h[mL]=\delta[m-m_0]
\]
(possibly with a known gain factor). That gives exact round-trip transparency up to delay/gain. In practice this is designed via **polyphase PR conditions**; if you choose \(d\) as a matched/biorthogonal partner of \(u\), this is the classic interpolation/decimation PR setup.

Good references:
1. **P. P. Vaidyanathan**, *Multirate Systems and Filter Banks* (PR theory, polyphase conditions).
2. **R. E. Crochiere, L. R. Rabiner**, *Multirate Digital Signal Processing*.
3. Communications literature on **FIR Nyquist and root-Nyquist filters** (e.g., raised-cosine/root-raised-cosine and equiripple Nyquist FIR design).

For audio oversampled nonlinear processing, this is exactly the right literature family.

## @RobinSchmidt

How would the design procedure to produce such a pair of filter kernels look like? Are there any open source libraries for that?

## Copilot

Yes. A practical **PR oversampling pair** workflow is:

1. Pick oversampling factor \(L\), latency (tap counts), and specs (\(\delta_p,\delta_s\), transition band).  
   **Important:** with finite FIRs you need a nonzero transition, so your transparent bandwidth is slightly below original Nyquist.

2. Design a **composite Nyquist(\(L\)) FIR** \(h[n]\) (at high rate) such that  
\[
h[mL]=\delta[m-m_0]
\]
and \(h\) is lowpass (anti-imaging/anti-aliasing behavior).  
This is the exact roundtrip condition for identity (up to delay \(m_0\)).

3. Factor \(H(z)\) into \(U(z)D(z)\):  
   - **Root-Nyquist route:** choose \(D(z)=U(z^{-1})\), with \(|U(e^{j\omega})|^2=H(e^{j\omega})\) (spectral factorization).  
   - **Biorthogonal route:** solve polyphase PR equations for \(u,d\) directly.

4. Normalize so \(h[m_0L]=1\), then verify numerically that upsample→\(u\)→\(d\)→downsample gives delayed identity to floating-point precision.

Open-source options:
- **liquid-dsp (C):** has Nyquist/root-Nyquist FIR design utilities (very relevant here).  
- **SciPy/NumPy (Python):** `remez`/`firls`/`firwin2` + `minimum_phase` + `upfirdn`; for strict constrained PR, combine with `cvxpy` or custom LS optimization.  
- **GNU Radio filter tools:** useful FIR design/resampler building blocks.

If you want, I can give you a concrete Python script that designs a \(u,d\) pair for a chosen \(L\), tap budget, and stopband target.

## @RobinSchmidt

Yes, please give me a Python script.

## Copilot

Great—here’s a self-contained script that designs an FIR up/down pair for factor `L` with an **exact discrete PR constraint** at the decimated sample times (identity roundtrip up to delay and floating-point error), while keeping the down filter close to a standard anti-alias lowpass.

```python
#!/usr/bin/env python3
import numpy as np
from scipy.signal import firwin, upfirdn

def design_pr_pair(L=4, span=12, cutoff_frac=0.90, window="kaiser", beta=8.6):
    """
    Design u[n], d[n] for upsample-by-L / downsample-by-L roundtrip.

    L           : oversampling factor
    span        : half-length in original-rate samples (taps = 2*span*L+1)
    cutoff_frac : fraction of ideal cutoff (ideal is 1/L of high-rate Nyquist)
    """
    # Choose odd tap count with center delay exactly on an L-grid.
    # This makes roundtrip delay an integer number of original-rate samples.
    num_taps = 2 * span * L + 1

    # Anti-imaging lowpass for upsampler (high-rate normalized freq, Nyquist=1.0)
    cutoff = cutoff_frac / L
    u = firwin(num_taps, cutoff=cutoff, window=(window, beta) if window == "kaiser" else window)

    # Initial anti-alias candidate: matched filter
    g = u[::-1].copy()

    # Build exact PR constraints on decimated samples of h = u * d:
    # h[delay + mL] = delta[m]
    Nd = num_taps
    Nu = num_taps
    h_len = Nu + Nd - 1
    delay = (Nu - 1 + Nd - 1) // 2  # high-rate samples

    m_min = int(np.ceil(-delay / L))
    m_max = int(np.floor((h_len - 1 - delay) / L))
    ms = np.arange(m_min, m_max + 1)

    A = np.zeros((len(ms), Nd))
    b = np.zeros(len(ms))

    for r, m in enumerate(ms):
        n = delay + m * L
        # h[n] = sum_j d[j] * u[n-j]
        j_lo = max(0, n - (Nu - 1))
        j_hi = min(Nd - 1, n)
        js = np.arange(j_lo, j_hi + 1)
        A[r, js] = u[n - js]
        if m == 0:
            b[r] = 1.0

    # Project g onto affine set {d | A d = b}
    # d = g + A^T (A A^T)^(-1) (b - A g)
    AAT = A @ A.T
    d = g + A.T @ np.linalg.pinv(AAT, rcond=1e-12) @ (b - A @ g)

    return u, d, delay

def verify_roundtrip(u, d, L=4, N=48000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)

    y_hi = upfirdn(u, x, up=L)      # upsample + anti-imaging
    z_hi = y_hi                      # identity "waveshaper"
    y = upfirdn(d, z_hi, down=L)     # anti-alias + downsample

    # Integer roundtrip delay in original-rate samples
    total_delay_hi = (len(u) - 1 + len(d) - 1) // 2
    total_delay_lo = total_delay_hi // L

    y_aligned = y[total_delay_lo:total_delay_lo + N]
    err = y_aligned - x

    # Also check exact PR constraint residual directly
    # A*d-b should be near machine precision if design succeeded.
    h = np.convolve(u, d)
    dec = h[total_delay_hi::L]
    # center element of dec corresponds to m=0
    # (sequence starts at m_min, so just measure off-peak + peak)
    peak = np.max(np.abs(dec))
    off = np.max(np.abs(np.delete(dec, np.argmax(np.abs(dec))))) if len(dec) > 1 else 0.0

    print(f"L={L}, taps={len(u)}")
    print(f"Roundtrip delay (original-rate samples): {total_delay_lo}")
    print(f"RMS error: {np.sqrt(np.mean(err**2)):.3e}")
    print(f"Max abs error: {np.max(np.abs(err)):.3e}")
    print(f"Composite decimated peak (should be ~1): {peak:.12f}")
    print(f"Composite decimated off-peak (should be ~0): {off:.3e}")

if __name__ == "__main__":
    L = 4
    u, d, delay_hi = design_pr_pair(L=L, span=12, cutoff_frac=0.90, window="kaiser", beta=8.6)
    verify_roundtrip(u, d, L=L)

    # Save kernels if desired:
    np.savetxt("u_kernel.txt", u)
    np.savetxt("d_kernel.txt", d)
```

Install deps with `pip install numpy scipy`.  
This gives you a practical PR pair for the identity-in-the-middle case; the transparent bandwidth is limited by the chosen lowpass cutoff/transition (as expected for finite FIR oversampling).

## @RobinSchmidt

Is it possible to design a single filter kernel that can be used for up- and downsampling, i.e. u = d, that still satisfies the PR constraints?

## Copilot

**Yes, but with an important nuance:** the “same-filter both sides” case is the **root-Nyquist** case.

For PR with oversampling factor \(L\), you need the composite \(h=u*d\) to satisfy \(h[mL]=\delta[m-m_0]\). If you set the two stages to the same prototype, this becomes a root condition (roughly “\(u\) convolved with itself” gives a Nyquist(\(L\)) filter). In frequency-domain PR form, a common condition is
\[
\sum_{r=0}^{L-1} |U(e^{j(\omega+2\pi r)/L})|^2 = L
\]
(up to delay/gain conventions), which is exactly the root-Nyquist/power-complementary constraint.

**Practical caveat:** exact FIR root-Nyquist solutions typically give you \(d[n]=u[-n]\) (time-reversed/conjugate partner). If you require literally identical tap order \(u[n]=d[n]\), that is much more restrictive; it is only straightforward when your \(u\) is symmetric in the required way (or you simply reuse one coefficient array and reverse indexing in one stage). So: **same prototype yes, same ordered kernel not always.**

## @RobinSchmidt

I was assuming that u is symmetric anyway. Isn't that the typical case for linear phase FIR filters like a windowed sinc?

## Copilot

**Yes—if \(u[n]\) is symmetric (linear phase Type I/II), then \(u[-n]\) is the same tap sequence up to indexing shift, so in practice you can use the same coefficient array for both stages.** That’s the usual implementation with windowed-sinc-style FIRs.

The key is still the **root-Nyquist condition** for that prototype: using the same symmetric filter twice must make the composite \(h=u*u\) satisfy \(h[mL]=\delta[m-m_0]\) (up to known gain/delay). So symmetry makes “same kernel” feasible, but PR depends on the Nyquist/root-Nyquist design constraint, not symmetry alone.

## @RobinSchmidt

How would one go about designing such a filter? Can you give me a Python script for that, too?

## Copilot

Yes—here’s a script that designs a **single symmetric FIR kernel** `u` for both stages (`u = d`) using nonlinear optimization:

- enforces PR constraints exactly (numerically):  
  \[
  (u*u)[(N-1)+mL] = \delta[m]
  \]
- keeps `u` close to a lowpass shape (for anti-imaging / anti-aliasing usefulness)

```python
#!/usr/bin/env python3
"""
Design one symmetric FIR kernel u for both up/down sampling (u=d)
such that upsample->u->u->downsample is PR (identity up to delay).

Requires: numpy, scipy
    pip install numpy scipy
"""

import argparse
import numpy as np
from scipy.signal import firwin, upfirdn
from scipy.optimize import minimize, NonlinearConstraint


def full_from_unique(a: np.ndarray) -> np.ndarray:
    """
    a[0] = center tap, a[k] = taps at +/-k
    returns odd-length symmetric FIR u
    """
    M = len(a) - 1
    N = 2 * M + 1
    u = np.empty(N, dtype=float)
    u[M] = a[0]
    for k in range(1, M + 1):
        u[M - k] = a[k]
        u[M + k] = a[k]
    return u


def unique_from_full(u: np.ndarray) -> np.ndarray:
    M = (len(u) - 1) // 2
    a = np.empty(M + 1, dtype=float)
    a[0] = u[M]
    a[1:] = u[M + 1 :]
    return a


def pr_moments(u: np.ndarray, L: int) -> np.ndarray:
    """
    Returns [c0, c1, ..., cR], where:
      c0 = sum_n u[n]^2
      ck = sum_n u[n] u[n-kL], k>=1
    PR requires c0=1 and ck=0 for k>=1.
    """
    N = len(u)
    R = (N - 1) // L
    c = np.zeros(R + 1, dtype=float)
    c[0] = np.dot(u, u)
    for k in range(1, R + 1):
        lag = k * L
        c[k] = np.dot(u[lag:], u[:-lag])
    return c


def lowpass_target_mag(f: np.ndarray, fp: float, fs: float) -> np.ndarray:
    """
    f in [0,1], where 1 = Nyquist.
    Cosine transition from fp to fs.
    """
    t = np.zeros_like(f)
    t[f <= fp] = 1.0
    tr = (f > fp) & (f < fs)
    if np.any(tr):
        x = (f[tr] - fp) / (fs - fp)
        t[tr] = 0.5 * (1.0 + np.cos(np.pi * x))
    return t


def design_same_kernel_pr(
    L=4,
    span=12,
    pass_frac=0.90,
    stop_frac=1.10,
    nfft=16384,
    w_stop=8.0,
    w_pass=1.0,
    maxiter=1500,
):
    """
    L         : oversampling factor
    span      : half-length in original-rate samples
                N = 2*span*L + 1 (odd, symmetric, delay aligns to L-grid)
    pass_frac : passband edge as fraction of ideal cutoff (ideal cutoff = 1/L Nyquist units)
    stop_frac : stopband start as fraction of ideal cutoff
    """
    if span < 1:
        raise ValueError("span must be >= 1")

    N = 2 * span * L + 1
    fp = pass_frac / L
    fs = stop_frac / L

    if not (0 < fp < fs < 1):
        raise ValueError("Need 0 < pass_frac/L < stop_frac/L < 1")

    # Initial symmetric lowpass guess near ideal anti-imaging cutoff
    u0 = firwin(N, cutoff=1.0 / L, window=("kaiser", 8.6), scale=False)
    u0 /= np.sqrt(np.dot(u0, u0))  # satisfy c0 roughly
    a0 = unique_from_full(u0)

    f = np.linspace(0.0, 1.0, nfft // 2 + 1)
    tgt = lowpass_target_mag(f, fp, fs)
    w = np.where(f >= fs, w_stop, w_pass)

    def objective(a):
        u = full_from_unique(a)
        U = np.fft.rfft(u, n=nfft)
        mag = np.abs(U)
        e = w * (mag - tgt)
        return np.mean(e * e)

    # Exact PR constraints (numerically): c0=1, ck=0
    R = (N - 1) // L
    target = np.zeros(R + 1, dtype=float)
    target[0] = 1.0
    nlc = NonlinearConstraint(
        lambda a: pr_moments(full_from_unique(a), L),
        lb=target,
        ub=target,
    )

    res = minimize(
        objective,
        a0,
        method="SLSQP",
        constraints=[nlc],
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not res.success:
        raise RuntimeError(
            f"Optimization failed: {res.message}\n"
            f"Try larger span, looser pass/stop edges, or different weights."
        )

    u = full_from_unique(res.x)
    return u, res


def verify_roundtrip(u: np.ndarray, L: int, Nsig=48000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(Nsig)

    y_hi = upfirdn(u, x, up=L)
    y = upfirdn(u, y_hi, down=L)

    delay_lo = (len(u) - 1) // L
    y_aligned = y[delay_lo : delay_lo + Nsig]
    err = y_aligned - x

    h = np.convolve(u, u)
    c = len(u) - 1
    R = (len(u) - 1) // L
    pr_samples = np.array([h[c + m * L] for m in range(-R, R + 1)])
    m0 = R
    off = np.max(np.abs(np.delete(pr_samples, m0))) if len(pr_samples) > 1 else 0.0

    print(f"L={L}, taps={len(u)}, roundtrip delay={delay_lo} input-rate samples")
    print(f"RMS error:     {np.sqrt(np.mean(err**2)):.3e}")
    print(f"Max abs error: {np.max(np.abs(err)):.3e}")
    print(f"h[c] (target 1): {pr_samples[m0]:.12f}")
    print(f"max |h[c+mL]|, m!=0 (target 0): {off:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4, help="oversampling factor")
    p.add_argument("--span", type=int, default=12, help="half-length in input-rate samples")
    p.add_argument("--pass-frac", type=float, default=0.90, help="pass edge / ideal cutoff")
    p.add_argument("--stop-frac", type=float, default=1.10, help="stop edge / ideal cutoff")
    p.add_argument("--w-stop", type=float, default=8.0, help="stopband objective weight")
    p.add_argument("--w-pass", type=float, default=1.0, help="passband objective weight")
    p.add_argument("--nfft", type=int, default=16384)
    p.add_argument("--out", type=str, default="u_same_kernel.txt")
    args = p.parse_args()

    u, res = design_same_kernel_pr(
        L=args.L,
        span=args.span,
        pass_frac=args.pass_frac,
        stop_frac=args.stop_frac,
        nfft=args.nfft,
        w_stop=args.w_stop,
        w_pass=args.w_pass,
    )

    np.savetxt(args.out, u)
    print(f"Saved kernel to {args.out}")
    print(f"Objective: {res.fun:.6e}, iterations: {res.nit}")

    verify_roundtrip(u, args.L)


if __name__ == "__main__":
    main()
```

Run example:

```bash
python design_same_kernel_pr.py --L 4 --span 14 --pass-frac 0.88 --stop-frac 1.12
```

If optimization fails, increase `--span` first (more taps gives the constraints room).
