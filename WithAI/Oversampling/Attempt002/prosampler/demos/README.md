# prosampler demos

## `design_and_plot.py` — design a filter pair and look at it

A complete, small, realistic scenario: a distortion effect at 44.1 kHz that runs its waveshaper at 4x oversampling.
The script

1. designs the up-/downsampling filters `u` and `d` from a handful of parameters,
2. prints lengths, latency (in samples and ms), stopband attenuation, passband ripple, memory and multiplies,
3. verifies the identity roundtrip (waveshaper bypassed) and pushes a test signal through a `tanh` waveshaper,
4. plots **impulse, step and magnitude responses** of `u` and `d` (figure 1),
5. plots the passband detail and the combined response `h = u * d`, whose samples at every L-th position are
   exactly 1 (centre) and 0 (figure 2) — the property that makes the roundtrip exact.

Run it from the repository root (needs matplotlib: `pip install -e .[plots]`):

```
python demos/design_and_plot.py                       # default design, opens two plot windows
python demos/design_and_plot.py --design matched      # single kernel u = d = g
python demos/design_and_plot.py --L 8 --atten-db 100 --passband-hz 19000
python demos/design_and_plot.py --joint --atten-db 60 --passband-hz 17000   # experimental, see below
python demos/design_and_plot.py --no-show --save out  # writes out.png and out_detail.png
python demos/design_and_plot.py --help
```

### Changing the design

Open the file: everything you would want to change is in the **PARAMETERS** block at the top.

| Parameter | Meaning | Default |
|---|---|---|
| `FS` | base sample rate in Hz | 44100 |
| `L` | oversampling factor | 4 |
| `PASSBAND_HZ` | flat passband must reach this frequency | 20000 |
| `ATTEN_DB` | required stopband attenuation of `u` and `d` | 80 |
| `DESIGN` | `"pair"` (two kernels) or `"matched"` (one kernel `g`) | `"pair"` |
| `PAIR_METHOD`, `PAIR_OBJECTIVE` | solver (`nullspace`/`kkt`) and objective (`grid`/`energy`) for `d` | nullspace, grid |
| `N_U`, `N_D` | fixed lengths instead of "shortest that meets the spec" | `None` |
| `MATCHED_K` | fixed latency (`g` has `L*K+1` taps) instead of a search | `None` |
| `JOINT`, `JOINT_PASSBAND_WEIGHT`, `JOINT_ON_WORSE_PEAK` | `--joint`: refine `u` and `d` jointly after the pair design (pair only) | off, 1.0, `"warn"` |
| `TRANSITION_WEIGHT` | leave at 0 (see `docs/usage.md`, section 4) | 0 |
| `TEST_FREQS_HZ`, `DRIVE`, `TEST_SECONDS` | the little time-domain check | 1 kHz + 6 kHz, 4x, 50 ms |

The functions `design()`, `report()`, `time_domain_check()`, `plot_main()` and `plot_detail()` are importable, so you
can also copy the script and reuse the pieces.

### What to expect (defaults: L = 4, 20 kHz passband at 44.1 kHz, 80 dB)

* Pair: `u` 217 taps, `d` 457 taps, latency 84 samples (1.9 ms at 44.1 kHz), both about -80 dB; Nyquist error ~1e-17.
* The transition band between 20 kHz and 24.1 kHz is narrow, so the filters are long. Measured pair lengths
  (L = 4, fs = 44.1 kHz):

  | passband | attenuation | `u` taps | `d` taps | latency (samples) |
  |---|---|---|---|---|
  | 20 kHz | 80 dB | 217 | 457 | 84 |
  | 18 kHz | 80 dB | 121 | 241 | 45 |
  | 16 kHz | 80 dB | 81 | 161 | 30 |
  | 20 kHz | 60 dB | 163 | 351 | 64 |
  | 18 kHz | 60 dB | 83 | 175 | 32 |
* `--design matched` needs a few minutes above ~70 dB (it solves a nonconvex problem; see `docs/theory.md`).
  Try `--atten-db 60` first.

### Things worth knowing when you adapt the script

* The passband is always given as a fraction of the **original** Nyquist frequency inside the library;
  `ps.passband_from_hz(f_hz, fs_hz)` converts.
* The step response is shown normalised to DC gain 1; `u` itself has DC gain `L` (pair) or `sqrt(L)` (matched).
* **Matched kernels change the level the waveshaper sees** (`1/sqrt(L)` of the input level; `fp.shaper_level`).
  The identity roundtrip is unaffected, but a nonlinearity is not, so the demo calls
  `reference_roundtrip(..., compensate_levels=True)`. Do the same (scale by `1/shaper_level` before and by
  `shaper_level` after the waveshaper) in your own implementation.
* `reference_roundtrip` uses full convolutions: fine for a 50 ms test, far too slow for real signals. A polyphase
  implementation is in `docs/usage.md`, section 3.

### `--joint` (experimental)

Runs `optimise_joint` on the pair design, prints before/after stopband peak and passband ripple, and overlays the
plain pair in grey in both figures. **It is not an improvement in peak stopband attenuation**: the joint objective is
stopband *energy* (which does drop, typically to 30-60 %), but with flat passbands the worst stopband *peak* is
usually the same or worse than that of the plain pair (e.g. L=4, 17 kHz, 60 dB: u -60.4 -> -53.0 dB). Without the
passband term (`JOINT_PASSBAND_WEIGHT = 0`) the peak can gain a few dB, but `u` and `d` then ripple against each other
by up to several dB. A peak (minimax) objective would be needed for a real gain. With `JOINT_ON_WORSE_PEAK = "keep"`
the demo falls back to the plain pair whenever the peak got worse.
