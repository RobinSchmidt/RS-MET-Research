# Context handoff for session 3: prosampler (Attempt002)

You are continuing a long collaboration with Robin Schmidt (audio DSP developer) on the Python library **`prosampler`**:
FIR oversampling filter pairs for waveshaping with *exactly transparent* round trip. Read this file, then
`oversampling_research_summary.md` (findings, pitfalls, open questions) and `prosampler/docs/{usage,theory}.md` in the repo.

## 0. Where things are

- Repo: `RobinSchmidt/rs-met-research` (GitHub), directory `WithAI/Oversampling/Attempt002`
  (`prosampler/` = the package, `oversampling_research_summary.md` next to it). Use `mcp__claude-code-remote__add_repo` and
  clone it. **The user updates the repo files himself** from the zips/files you send; a fresh clone is the latest state.
  He also edits `prosampler/README.md` by hand: always start from his version (`git fetch; git show origin/master:<path>`)
  when you rebuild the zip, never overwrite it.
- Deliver with `SendUserFile`: `prosampler.zip` (build from the `Attempt002` directory:
  `zip -qr /home/claude/prosampler.zip prosampler -x '*/__pycache__/*' '*.pyc' '*/.pytest_cache/*' '*.png'`) and the updated
  `oversampling_research_summary.md`. **No `.png` files in the zip or the repo** (figures are only for looking at in the
  chat; the demo scripts may write them elsewhere).
- Run the tests with `cd prosampler && PYTHONPATH=src python3 -m pytest -q tests` (88 pass, about 10 minutes: run detached,
  see section 6). Docs examples in ```python blocks are executed by `tests/test_docs.py` (use ```py for slow examples).

## 1. The idea

Pipeline: input -> zero-stuff by L -> `u` -> nonlinearity -> `d` -> keep every L-th sample. With the identity as nonlinearity
the output equals the input (delayed by `latency`) iff `h = u*d` is Nyquist(L): `h[L*latency + L*k] = delta[k]`. This is
**linear in `d` for fixed `u`**. Both filters are linear phase (symmetric); `u` has DC gain `L`, `d` has DC gain 1; the passband
is a fraction of the *original* Nyquist frequency (e.g. 0.85); the stopband starts at `2 - passband`. A centred pair needs
`n_u + n_d - 2` to be a multiple of `2L` and both lengths odd (total length steps by `2L`).
A *matched* kernel `g` (`u = d = g`, DC gain about `sqrt(L)`) makes the Nyquist condition quadratic.

## 2. Library (all exported from `prosampler`)

| Function | Purpose |
|---|---|
| `design_pair(L, passband, atten_db, ...)`, `design_partner` | Kaiser-sinc `u` + optimal partner `d` ("plain pair"); `objective="grid"/"energy"/"minimax"` |
| `design_matched`, `optimise_matched` | matched `g`, energy objective (SLSQP + Gauss-Newton), spec-driven length search |
| `design_partner_minimax`, `optimise_joint_minimax` | minimax LP partner; alternating minimax (cannot move the roll-off split) |
| `optimise_joint_slp(pair, ripple_db=0.1, transition_max_db="ripple", ...)` | simultaneous minimax of `u` and `d` by sequential LP (linearised Nyquist, trust region, Gauss-Newton projection, elastic ripple/cap) |
| `search_pair(L, passband, atten_db, ...)` | shortest total length meeting a spec (scan or `strategy="adaptive"`; `resolution="coarse"/"medium"/"fine"`; `n_jobs`; progress lines with ETA; `rescue` extra starts) |
| `best_pair_for_length(L, passband, total, ...)` | best stopband for a *given* total length (scans the `u`/`d` split) |
| `optimise_matched_slp`, `search_matched` | minimax matched kernel; shortest `K` (secant + bracket over `K`) |
| `TransitionShape` | prescribed roll-off; a dead end for removing the bump, kept for exploration |
| `optimise_joint`, `rebalance`, `tie`, `compare`, `DEFAULT_RIPPLE_DB` | experimental joint/tying code; table of pairs; the constant 0.1 |

**Defaults (changed on the user's request):** `ripple_db = 0.1` dB peak-to-peak per filter and `transition_max_db = "ripple"`
(bump cap = ripple/2, i.e. the bump may not exceed the gain the passband ripple reaches anyway) for `search_pair`,
`best_pair_for_length`, `optimise_joint_slp`, `optimise_matched_slp`, `search_matched`. `ripple_db=None` gives the old relative behaviour
(`ripple_factor` times the plain pair's ripple; for the SLP twice the start's), `transition_max_db=None` no cap.
`ripple_db` is **peak-to-peak** like `metrics()`; the bound is `+-ripple_db/2` around the nominal gain and is elastic (a start with
more ripple is driven down with weight x100, a start with less may use it).

## 3. Key results (all L = 4, passband 0.85, worst stopband peak target 60 dB unless stated)

- Shortest pair found (bump cap in brackets): plain pair 314 taps (99/215, ripple 0.016 dB, bump +1.17 dB); cap +0.5 dB 274 (103/171);
  cap +0.2 dB 274; cap +0.1 dB 282 (103/179); cap 0 dB 290 (107/183, -60.8 dB); cap ripple/2 at 0.016 dB ripple 290 (-61.9 dB);
  0.05 dB ripple 274 (103/171, -61.4 dB); **0.1 dB ripple, cap ripple/2: 258 taps (99/159), -62.2 dB**, bump +0.05 dB (250 misses: -59.5 dB).
- Ripple sweep at 290 taps (`best_pair_for_length`, cap ripple/2), best worst stopband: 0.008 dB -60.6; 0.016 -61.9; 0.03 -63.4;
  0.05 -66.2 (109/181); 0.1 -70.7 (115/175); 0.2 -73.0 (117/173). Gain per doubling 1.2/1.7/3.9/4.5/2.3 dB (mean 2.6).
  The best split is about 37 % of the total in `u`.
- Matched: energy-optimal 233 taps; **minimax `search_matched` 181 taps (K = 45), -60.5 dB, 6 probes, 3 min on 2 cores**. For a matched
  kernel ripple and bump are about 1e-5 dB (forced by Nyquist: `A^2(w) + A^2(2pi/L - w)` is nearly constant), so the bounds never bind.
  Comparison (stored coefficients / multiplies per sample / latency): plain pair 157/312/39; 0.1 dB pair (258 taps) 129/256/32; matched
  minimax 90/358/45. Matching saves memory (-30 %), costs arithmetic (+40 %) and latency (+13).
- The starting window does not matter: Kaiser, Dolph-Chebyshev (60/70/80 dB), Blackman-Harris, Remez `u` all end within 0.2 dB after the SLP
  (Blackman 1 dB worse). Extra starts (`rescue`) did not improve any failing length. A split step of 2 around the winner at 266 taps found no pass
  (flat plateau within 0.3 dB). Keep Kaiser.
- Perception: ~1 dB (0.3-0.5 dB in careful tests) is the just-noticeable level change, so 0.1 dB ripple is inaudible; also, in the linear path the
  ripples of `u` and `d` cancel exactly (the pair is Nyquist); they only matter for the nonlinear path.

## 4. Pitfalls and lessons (details in section 4 of the summary)

- Energy objectives do not improve the peak; minimax is needed. Alternating minimax cannot move the roll-off split; SLP can.
- Compare designs at **equal ripple** (ripple is a free resource, about 2.6 dB per doubling). `ripple_db` is peak-to-peak (an early bug used +-r).
- The search acceptance test compares against the *reference ripple* (an early version let the ripple drift to 1.7x and reported 266 instead of 274).
- Results are the shortest length *found* (local optima scatter by 1-2 dB between neighbours); do not claim optimality.
- Minimax LP numerics: HiGHS tolerance 1e-10, rescale, floor -160 dB, solver fallbacks, early infeasibility check for `transition_max_db`.
- Honest reporting matters to the user: report corrections and results that contradict expectations.

## 5. Open items (numbered list in section 6 of the summary; done items are marked there)

Top candidates for the next session:
1. **#1 Measure what matters:** real alias suppression with a waveshaper (tanh, folding) on tones/noise, exact-PR pairs against conventional
   independently designed filters of equal length. The only item that tests whether all this matters in practice.
2. #14 Other specs (0.9 / 80 dB; L = 2, 3, 8) for the length / ripple / bump studies (cheap: `search_pair`, `best_pair_for_length`, `search_matched`).
3. Housekeeping: #19 tidy-ups (remove `TransitionShape` when agreed, SLP option in `design_matched`), #17 warm starts, #15 fair plain baseline at 0.1 dB,
   #16 bump penalty option, #13 tail-penalty idea (speculative, needs a new optimiser), #9 efficient implementation / C++.
Marked in the summary: #2 and #7 DONE; #4 and #8 PARTLY TOUCHED; the others open.

## 6. Working notes for the sandbox

- The tool call limit is about 10 minutes: run anything longer **detached** (`(PYTHONPATH=src setsid nohup python3 script.py > log 2>&1 < /dev/null &)`)
  and poll with `sleep` below that limit. A job started with plain `nohup` inside a call that times out gets killed.
- Timings (2 cores): a fine adaptive `search_pair` about 15-30 min; `best_pair_for_length` about 9-11 min; `search_matched` 0.85 / 60 dB 3 min;
  one SLP candidate about 1 min at ~300 taps; the full test suite about 10 min.
- Do not edit a cloned file the user has also edited without checking `git fetch` first; the clone in a new sandbox is stale until you fetch.
- Update docs with every finding (`docs/usage.md`, `docs/theory.md`, the summary), keep the test suite green, resend zip + summary.
- User preferences saved in memory: zip updates start from his README; no `.png` in zip/repo. He likes to ask "why / how does it work" questions
  and appreciates plain answers about cost (code complexity against compute), plus honest statements when something does not work.
