#!/usr/bin/env python3
"""
prosampler demo: design an oversampling filter pair and look at it.

Scenario: a waveshaper (distortion) plug-in runs at L x oversampling to keep aliasing down.  The up- and
downsampling filters u and d are designed so that, with the waveshaper bypassed, the chain

    x -> zero-stuff by L -> u -> [waveshaper] -> d -> keep every L-th sample -> y

returns the input exactly (a pure delay of `latency` samples, error ~1e-15).

What this script does
  1. designs the filters from the parameters in the block below,
  2. prints a summary (lengths, latency, stopband, ripple, memory, multiplies),
  3. checks the identity roundtrip and runs a short test signal through a tanh waveshaper,
  4. plots impulse, step and magnitude responses of u and d (figure 1),
  4b. optionally (--joint, pair design only) refines u and d jointly and overlays the plain pair in grey.  This is
     EXPERIMENTAL and measured NOT to improve the worst stopband peak with flat passbands (see docs/theory.md),
  5. plots the passband detail and the combined response h = u * d, whose samples at the Nyquist positions are
     exactly 1 and 0 (figure 2).

HOW TO USE: edit the parameters in the PARAMETERS block, or override a few from the command line, e.g.

    python demos/design_and_plot.py --L 8 --atten-db 100 --passband-hz 19000
    python demos/design_and_plot.py --design matched --atten-db 60 --save design.png --no-show
    python demos/design_and_plot.py --objective minimax --atten-db 60 --passband-hz 17000
    python demos/design_and_plot.py --joint-minimax --atten-db 60 --passband-hz 17000
    python demos/design_and_plot.py --joint --atten-db 60 --passband-hz 17000
    python demos/design_and_plot.py --help

Needs matplotlib (pip install prosampler[plots]).  Run it from the repository root, or install the package first
(pip install -e .).
"""
import argparse
import pathlib
import sys

import numpy as np

try:                                              # allow running from a source checkout without installing
    import prosampler as ps
except ImportError:                               # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    import prosampler as ps

# =====================================================================================================================
# PARAMETERS  -  edit these
# =====================================================================================================================
FS = 44100.0            # base (original) sample rate in Hz
L = 4                   # oversampling factor (integer >= 2)
PASSBAND_HZ = 20000.0   # the flat passband must extend to this frequency (converted to a fraction of Nyquist)
ATTEN_DB = 80.0         # required stopband attenuation of u and d, in dB

DESIGN = "pair"         # "pair"    : u and d are different filters (shortest, best attenuation per tap)
                        # "matched" : one kernel g serves as both u and d (less memory, more multiplies)

# --- options for DESIGN = "pair" -------------------------------------------------------------------------------------
PAIR_METHOD = "nullspace"        # solver for d: "nullspace" (robust) or "kkt"
PAIR_OBJECTIVE = "grid"          # "grid" (frequency-grid least squares), "energy" (closed-form stopband energy) or
                                 # "minimax" (linear program: minimal stopband PEAK; d about 5-12 % shorter for the same spec)
TRANSITION_WEIGHT = 0.0          # extra least-squares weight on the transition band.  Keep it 0: for linear-phase pairs
                                 # it does NOT remove the ~+1.1 dB bump of d at the passband edge (it grows to ~+1.6 dB)
                                 # and it costs stopband, so d must get 25-50 % longer to meet the same spec
N_U = None                       # fixed lengths (odd numbers); None = let the library pick the shortest that works.
N_D = None                       # If you set N_D you must also set N_U (or give atten_db and let N_U follow the spec)

JOINT_MINIMAX = False            # True: alternating-LP joint minimax refinement of (u, d) (pair design only)
JOINT_MINIMAX_RIPPLE_DB = None   # allowed passband deviation of each filter in that refinement (None = 2x the start's)
JOINT = False                    # True: refine (u, d) jointly after the pair design (experimental, see below)
JOINT_PASSBAND_WEIGHT = 1.0      # weight of the passband-flatness term; 0 lets u and d ripple against each other
JOINT_ON_WORSE_PEAK = "warn"     # "warn": keep the result and warn if the worst stopband peak got worse; "keep": fall
                                 # back to the plain pair in that case.  The joint objective is stopband ENERGY, not the
                                 # peak, so the peak is often NOT better than the plain pair's.

# --- options for DESIGN = "matched" ----------------------------------------------------------------------------------
MATCHED_K = None                 # None = search the smallest latency K that meets ATTEN_DB; else g has L*K+1 taps

# --- the little time-domain check -------------------------------------------------------------------------------------
TEST_FREQS_HZ = (1000.0, 6000.0) # sine components of the test signal
DRIVE = 4.0                      # gain in front of tanh()
TEST_SECONDS = 0.05

# --- plotting -----------------------------------------------------------------------------------------------------------
SHOW_DETAIL_FIGURE = True        # second figure: passband detail and the Nyquist property of h = u * d
MAX_STEP_SECONDS = None          # step-response x-range in ms (None = whole kernel)
# =====================================================================================================================


def design(fs=FS, L=L, passband_hz=PASSBAND_HZ, atten_db=ATTEN_DB, design=DESIGN, **opts):
    """Design the filters and return a :class:`prosampler.FilterPair`."""
    passband = ps.passband_from_hz(passband_hz, fs)          # fraction of the ORIGINAL Nyquist frequency
    if design == "pair":
        return ps.design_pair(
            L, passband, atten_db,
            n_u=opts.get("n_u", N_U), n_d=opts.get("n_d", N_D),
            method=opts.get("method", PAIR_METHOD), objective=opts.get("objective", PAIR_OBJECTIVE),
            transition_weight=opts.get("transition_weight", TRANSITION_WEIGHT))
    if design == "matched":
        return ps.design_matched(L, passband, atten_db, K=opts.get("K", MATCHED_K))
    raise ValueError("design must be 'pair' or 'matched'")


def refine_joint(fp, weight=JOINT_PASSBAND_WEIGHT, on_worse_peak=JOINT_ON_WORSE_PEAK):
    """Joint optimisation of (u, d) starting from the pair design ``fp``; prints a before/after comparison."""
    out = ps.optimise_joint(fp, passband_weight=weight, on_worse_peak=on_worse_peak)
    j = out.info.get("joint")
    if j is None:                                  # on_worse_peak="keep" returned the input pair
        print("\njoint optimisation: peak not improved, plain pair kept")
        return out
    f = lambda t: f"u {t[0]:.1f} / d {t[1]:.1f} dB"
    r = lambda t: f"u {t[0]:.4f} / d {t[1]:.4f} dB"
    print(f"\njoint optimisation (passband weight {j['passband_weight']:g}); stopband energy x{j['objective_ratio']:.2f}")
    print(f"  stopband peak   before: {f(j['stop_db_before'])}   after: {f(j['stop_db_after'])}")
    print(f"  passband ripple before: {r(j['ripple_db_before'])}   after: {r(j['ripple_db_after'])}")
    return out


def refine_joint_minimax(fp, ripple_db=JOINT_MINIMAX_RIPPLE_DB):
    """Alternating-LP joint minimax refinement of ``(u, d)``; prints a before/after comparison."""
    out = ps.optimise_joint_minimax(fp, ripple_db=ripple_db)
    h = out.info["joint_minimax"]["history"]
    print(f"\njoint minimax: {len(h) - 1} round(s), passband ripple bound {out.info['joint_minimax']['ripple_db']:.4f} dB per filter")
    print(f"  stopband peak   before: u {h[0][0]:.1f} / d {h[0][1]:.1f} dB   after: u {h[-1][0]:.1f} / d {h[-1][1]:.1f} dB")
    return out


def report(fp, fs):
    """Print the numbers that matter."""
    m = fp.metrics()
    print(f"\n=== {fp.method} ===")
    print(f"L = {fp.L}   base rate {fs:g} Hz   oversampled rate {fs * fp.L:g} Hz")
    print(f"passband edge {fp.passband * fs / 2:.0f} Hz (fraction {fp.passband:.4f} of Nyquist);"
          f" stopband starts at {(2 - fp.passband) * fs / 2:.0f} Hz")
    print(f"u: {fp.n_u} taps   d: {fp.n_d} taps   latency: {fp.latency} samples at {fs:g} Hz "
          f"({1000 * fp.latency / fs:.3f} ms)")
    print(f"stopband  u {m['u_stop_db']:.1f} dB   d {m['d_stop_db']:.1f} dB")
    print(f"passband ripple  u {m['u_ripple_db']:.4f} dB   d {m['d_ripple_db']:.4f} dB")
    print(f"transition-band peak of d: {m['d_transition_peak_db']:+.2f} dB")
    print(f"Nyquist error of u*d: {m['nyquist_error']:.1e}   (0 = exact)")
    print(f"memory: {fp.stored_coeffs} coefficients (symmetry exploited)   "
          f"cost: {fp.macs_per_sample} multiplies per base-rate sample (polyphase)")
    if "meets_spec" in fp.info:
        print("meets the requested attenuation:", fp.info["meets_spec"])


def time_domain_check(fp, fs, freqs=TEST_FREQS_HZ, drive=DRIVE, seconds=TEST_SECONDS):
    """Identity roundtrip (waveshaper bypassed) and a tanh run, using the slow-but-simple reference chain."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    y_id = ps.reference_roundtrip(x, fp.u, fp.d, fp.L, fp.latency, shaper=None)
    err = float(np.max(np.abs(y_id - x)))
    y_nl = ps.reference_roundtrip(x, fp.u, fp.d, fp.L, fp.latency, shaper=lambda v: np.tanh(drive * v),
                                  compensate_levels=True)
    print(f"\nidentity roundtrip, max |y - x| = {err:.1e}   (waveshaper bypassed, output aligned by `latency`)")
    # (the peak can slightly exceed 1: the downsampling lowpass rings on the clipped waveform, as any lowpass does)
    if abs(fp.shaper_level - 1) > 1e-6:
        print(f"note: the level at the waveshaper is {fp.shaper_level:.3f} x the input level (matched kernel); "
              f"compensate_levels=True rescales around the shaper so it sees the input level")
    print(f"tanh({drive:g} * x) through the oversampled chain: peak output {np.max(np.abs(y_nl)):.3f}")
    return x, y_id, y_nl


def plot_main(fp, fs, max_step_ms=MAX_STEP_SECONDS, base=None):
    """Figure 1: impulse, step and magnitude responses of u and d."""
    import matplotlib.pyplot as plt

    fs_hi = fs * fp.L
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    pb_hz = fp.passband * fs / 2
    sb_hz = (2 - fp.passband) * fs / 2
    for col, (h, name) in enumerate(((fp.u, "u (upsampling, anti-imaging)"), (fp.d, "d (downsampling, anti-aliasing)"))):
        hb = None if base is None else (base.u, base.d)[col]
        dc = float(np.sum(h))
        t_ms = 1000 * np.arange(len(h)) / fs_hi

        ax = axes[0, col]
        if hb is not None:
            ax.plot(1000 * np.arange(len(hb)) / fs_hi, hb / np.sum(hb), color="0.6", lw=0.9, label="plain pair")
        ax.plot(t_ms, h / dc, lw=0.9)
        ax.set_title(f"{name}\n{len(h)} taps")
        ax.set_ylabel("impulse response\n(normalised to DC gain 1)")
        ax.set_xlabel("time (ms)")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        if hb is not None:
            ax.plot(1000 * np.arange(len(hb)) / fs_hi, np.cumsum(hb) / np.sum(hb), color="0.6", lw=0.9)
        ax.plot(t_ms, np.cumsum(h) / dc, lw=0.9)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.5)
        ax.set_ylabel("step response")
        ax.set_xlabel("time (ms)")
        if max_step_ms:
            ax.set_xlim(0, max_step_ms)
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        w = np.linspace(0, np.pi, 8000)
        mag = np.abs(ps.freqz(h, w)) / abs(dc)
        f_khz = w / np.pi * fs_hi / 2 / 1000
        if hb is not None:
            ax.plot(f_khz, 20 * np.log10(np.abs(ps.freqz(hb, w)) / abs(np.sum(hb)) + 1e-300), color="0.6", lw=0.9,
                    label="plain pair")
        ax.plot(f_khz, 20 * np.log10(mag + 1e-300), lw=0.9, label="joint" if hb is not None else None)
        ax.axvspan(0, pb_hz / 1000, color="C2", alpha=0.12, label="passband")
        ax.axvspan(sb_hz / 1000, fs_hi / 2000, color="C3", alpha=0.08, label="stopband")
        ax.axvline(fs / 2000, color="k", lw=0.6, ls=":", label="original Nyquist")
        ax.set_xlim(0, fs_hi / 2000)
        ax.set_ylim(-160, 10)
        ax.set_ylabel("magnitude (dB)")
        ax.set_xlabel(f"frequency (kHz, at the oversampled rate {fs_hi / 1000:g} kHz)")
        ax.grid(alpha=0.3)
        if col == 0 or hb is not None:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"prosampler: {fp.method}   L = {fp.L}, passband {pb_hz:.0f} Hz at fs = {fs:g} Hz", fontsize=12)
    return fig


def plot_detail(fp, fs, base=None):
    """Figure 2: passband magnitude in detail, and the combined response h = u * d (the Nyquist property)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    w = np.linspace(0, fp.passband * np.pi / fp.L, 2000)
    f = w / np.pi * fs * fp.L / 2
    if base is not None:
        for h, label in ((base.u, "u plain"), (base.d, "d plain")):
            ax1.plot(f / 1000, 20 * np.log10(np.abs(ps.freqz(h, w)) / abs(np.sum(h))), color="0.6", lw=1, label=label)
    for h, label in ((fp.u, "u"), (fp.d, "d")):
        mag = np.abs(ps.freqz(h, w)) / abs(np.sum(h))
        ax1.plot(f / 1000, 20 * np.log10(mag), label=label, lw=1)
    ax1.set_xlabel("frequency (kHz)")
    ax1.set_ylabel("passband magnitude (dB)")
    ax1.set_title("passband detail (each filter relative to its own DC gain)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    h = np.convolve(fp.u, fp.d)
    n0 = fp.L * fp.latency
    n = np.arange(len(h)) - n0
    on_grid = (n % fp.L) == 0
    centre = slice(max(0, n0 - 6 * fp.L), min(len(h), n0 + 6 * fp.L + 1))
    ax2.stem(n[centre][~on_grid[centre]], h[centre][~on_grid[centre]] / fp.L, linefmt="C0-", markerfmt="C0o",
             basefmt="k-", label="other samples (free)")
    ax2.stem(n[centre][on_grid[centre]], h[centre][on_grid[centre]] / fp.L, linefmt="C3-", markerfmt="C3s",
             basefmt="k-", label=f"every {fp.L}-th sample: 1 at the centre, else 0")
    ax2.set_xlabel("sample offset from the centre (oversampled rate)")
    ax2.set_ylabel("h = u * d   (divided by L)")
    ax2.set_title(f"Nyquist({fp.L}) property, max deviation {fp.nyquist_error:.0e}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="Defaults come from the PARAMETERS block at the top of this file.")
    ap.add_argument("--fs", type=float, default=FS, help="base sample rate in Hz")
    ap.add_argument("--L", type=int, default=L, help="oversampling factor")
    ap.add_argument("--passband-hz", type=float, default=PASSBAND_HZ)
    ap.add_argument("--atten-db", type=float, default=ATTEN_DB)
    ap.add_argument("--design", choices=("pair", "matched"), default=DESIGN)
    ap.add_argument("--joint", action="store_true", default=JOINT,
                    help="refine u and d jointly (pair design only; experimental, usually no better peak stopband)")
    ap.add_argument("--objective", choices=("grid", "energy", "minimax"), default=PAIR_OBJECTIVE,
                    help="objective of the partner design for --design pair (minimax = linear program, minimal peak)")
    ap.add_argument("--joint-minimax", action="store_true", default=JOINT_MINIMAX,
                    help="alternating-LP joint minimax refinement of u and d (pair design only)")
    ap.add_argument("--save", metavar="PREFIX", help="save figures as PREFIX.png and PREFIX_detail.png")
    ap.add_argument("--no-show", action="store_true", help="do not open plot windows")
    args = ap.parse_args(argv)

    if args.no_show or args.save:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")

    fp = design(args.fs, args.L, args.passband_hz, args.atten_db, args.design, objective=args.objective)
    base = None
    if args.joint or args.joint_minimax:
        if args.design != "pair":
            ap.error("--joint / --joint-minimax need --design pair")
        report(fp, args.fs)
        joint = refine_joint_minimax(fp) if args.joint_minimax else refine_joint(fp)
        if joint is not fp:
            m = joint.metrics()
            joint.info["meets_spec"] = max(m["u_stop_db"], m["d_stop_db"]) <= -args.atten_db   # re-evaluate
            base, fp = fp, joint
    report(fp, args.fs)
    time_domain_check(fp, args.fs)

    figs = [plot_main(fp, args.fs, base=base)]
    if SHOW_DETAIL_FIGURE:
        figs.append(plot_detail(fp, args.fs, base=base))
    if args.save:
        figs[0].savefig(f"{args.save}.png", dpi=130)
        if len(figs) > 1:
            figs[1].savefig(f"{args.save}_detail.png", dpi=130)
        print(f"\nsaved {args.save}.png" + (f" and {args.save}_detail.png" if len(figs) > 1 else ""))
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return fp


if __name__ == "__main__":
    main()
