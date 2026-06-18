"""
Simulated-annealing Monte-Carlo recovery of the projectile path (least action).
Pure Python + matplotlib. Runs until the path converges to the true trajectory
within MC_TOL, then stops.

  blue line + dots   = current path (the N+1 nodes)
  dashed dark grey   = initial straight-line guess (where the path started)
  dashed yellow      = target parabola (the actual trajectory)
  green / grey marks  = tried nudges this frame (accepted / rejected)
  right panel        = action vs iterations, descending to the least-action floor.

============================ HOW TO RUN ============================
Edit RUN_MODE below (or set the MC_MODE environment variable):
    RUN_MODE = "live"   -> open an interactive window
    RUN_MODE = "save"   -> render a VIDEO file to SAVE_PATH (.mp4 or .mov)
Command-line equivalents:
    python3 mc_least_action.py                       # uses RUN_MODE below
    MC_MODE=live  python mc_least_action.py          # force live
    MC_MODE=save  MC_OUT=least_action.mov python3 mc_least_action.py
Saving a video needs ffmpeg on PATH (apt-get install ffmpeg / brew install ffmpeg).
====================================================================
"""

import os
import numpy as np
import matplotlib

# ----------------------------- run mode ------------------------------------
RUN_MODE = os.environ.get("MC_MODE", "live")          # "live" or "save"
SAVE_PATH = os.environ.get("MC_OUT", "projectile_monte_carlo.mp4")   # real video (.mp4 / .mov)
# ---------------------------------------------------------------------------

if RUN_MODE == "save":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------------
# Locked-in physics
# ---------------------------------------------------------------------------
V0, ANGLE, H0, G, M = 10.0, 45.0, 2.0, 9.8, 1.0
theta = np.radians(ANGLE)
VX = V0 * np.cos(theta)
VY = V0 * np.sin(theta)
T = (VY + np.sqrt(VY ** 2 + 2 * G * H0)) / G
RANGE = VX * T

# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------
N = int(os.environ.get("MC_N", 21))
DT = T / N
XS = np.linspace(0.0, RANGE, N + 1)
TS = XS / VX
Y_PARABOLA = H0 + VY * TS - 0.5 * G * TS ** 2

KE_C = 0.5 * M / DT
PE_C = 0.5 * M * G * DT


def action(y):
    dy = np.diff(y)
    return float(np.sum(0.5 * M * (VX ** 2 + (dy / DT) ** 2)
                        - M * G * (y[:-1] + y[1:]) / 2.0) * DT)


# ---------------------------------------------------------------------------
# Monte-Carlo / annealing settings
# ---------------------------------------------------------------------------
TOL = float(os.environ.get("MC_TOL", 0.02))         # converge within this (metres)
SIGMA0 = float(os.environ.get("MC_SIGMA", 0.20))    # initial nudge size
SEED = int(os.environ.get("MC_SEED", 1))
SWEEPS = int(os.environ.get("MC_SWEEPS", 3))        # sweeps per frame
TAU0 = float(os.environ.get("MC_TAU0", 1.0))        # hot start
COOL = float(os.environ.get("MC_COOL", 0.985))      # geometric cooling / frame
TAU_MIN = 1e-5
MAXFRAMES = int(os.environ.get("MC_MAXFRAMES", 6000))
GHOST_SHOW = 24
ACC_HI, ACC_LO = 0.35, 0.25

INTERVAL = 80                                        # ms per frame
OUTPUT_FPS = int(os.environ.get("MC_FPS", 30))       # editor-friendly output rate
INTRO_FRAMES = max(1, round(1000 / INTERVAL))        # ~1 s intro before sim
X_INIT = 5000                                        # initial fixed x-window (iterations)
Y_PAD = 0.6                                          # right-panel vertical headroom

rng = np.random.default_rng(SEED)
y = H0 * (1.0 - np.arange(N + 1) / N)
y[0], y[N] = H0, 0.0
Y_INIT = y.copy()
S = action(y)
S0 = S
S_TARGET = action(Y_PARABOLA)
MAXDEV0 = float(np.max(np.abs(Y_INIT - Y_PARABOLA)))

sigma = SIGMA0
total_props = 0
steps_hist = []
S_hist = []
S_max = S0
state = {"converged": False}


def run_frame(tau, sig):
    global S, total_props
    n_props = SWEEPS * (N - 1)
    tried = []
    stride = max(1, n_props // GHOST_SHOW)
    acc = 0
    k = 0
    for _ in range(SWEEPS):
        for _ in range(N - 1):
            i = rng.integers(1, N)
            d = rng.normal(0.0, sig)
            a = y[i - 1]; b = y[i]; c = y[i + 1]; bn = b + d
            dS = (KE_C * ((bn - a) ** 2 + (c - bn) ** 2 - (b - a) ** 2 - (c - b) ** 2)
                  - 2.0 * PE_C * (bn - b))
            accept = (dS < 0.0) or (rng.random() < np.exp(-dS / max(tau, 1e-12)))
            if k % stride == 0 and len(tried) < GHOST_SHOW:
                tried.append((i, a, bn, c, accept))
            if accept:
                y[i] = bn; S += dS; acc += 1
            k += 1
    total_props += n_props
    return tried, acc / n_props


# ---------------------------------------------------------------------------
# Figure  (equal left/right halves; right graph is the same width but shorter)
# ---------------------------------------------------------------------------
plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white"})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.2))
fig.patch.set_facecolor("#0e0e0e")
for ax in (axL, axR):
    ax.set_facecolor("#0e0e0e")

axL.set_position([0.040, 0.000, 0.440, 0.860])       # left half, full height
axR.set_position([0.545, 0.180, 0.430, 0.700])       # right half, shorter & centred

axL.set_xlim(-0.3, RANGE + 0.3)
axL.set_ylim(-2.0, Y_PARABOLA.max() + 3.0)
axL.axhline(0.0, color="#9A9A9A", lw=1.5, alpha=0.5)
axL.plot(XS, Y_INIT, color="#5A5A5A", lw=1.4, ls="--", alpha=0.85, zorder=1)
axL.plot(XS, Y_PARABOLA, color="#FFF1A6", lw=1.5, ls="--", alpha=0.6, zorder=2)
(path_line,) = axL.plot(XS, y.copy(), color="#4FC3F7", lw=1.8,
                        marker="o", markersize=4.0, mfc="#4FC3F7",
                        mec="#4FC3F7", zorder=6)
path_line.set_visible(False)
axL.axis("off")

axR.set_xlim(0, X_INIT)
axR.set_ylim(S_TARGET - Y_PAD, S0 + Y_PAD)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.spines["left"].set_color("white")
axR.spines["bottom"].set_color("white")
axR.set_xlabel("Iterations")
axR.set_ylabel("action  S")
axR.axhline(S_TARGET, color="#FFF1A6", ls="--", lw=1.4, alpha=0.85)
(conv_line,) = axR.plot([], [], color="#4FC3F7", lw=1.6)

title = fig.suptitle("", color="white", fontsize=13)

ghost_artists = []


def set_title(tau, props, s, md):
    title.set_text(f"T = {tau:.4f}     Iterations = {props}     "
                   f"S = {s:.3f} Js     Max|deviation| = {md:.3f} m")


def update(frame):
    global ghost_artists, sigma, S_max
    for art in ghost_artists:
        art.remove()
    ghost_artists = []

    mc = frame - INTRO_FRAMES

    if mc < 0:                                   # intro: dashed references only
        path_line.set_visible(False)
        set_title(TAU0, 0, S0, MAXDEV0)
        return [path_line, conv_line, title]

    if mc == 0:                                  # reveal starting straight line
        path_line.set_visible(True)
        path_line.set_ydata(y)
        if not steps_hist:
            steps_hist.append(0); S_hist.append(S)
        conv_line.set_data(steps_hist, S_hist)
        axR.set_ylim(S_TARGET - Y_PAD, S_max + Y_PAD)
        set_title(TAU0, total_props, S, MAXDEV0)
        return [path_line, conv_line, title]

    tau = max(TAU0 * COOL ** mc, TAU_MIN)         # mc >= 1: run the simulation
    tried, ar = run_frame(tau, sigma)
    if ar > ACC_HI:
        sigma = min(sigma * 1.08, 0.5)
    elif ar < ACC_LO:
        sigma = max(sigma * 0.92, 1e-3)

    for (i, a, bn, c, acc) in tried:
        col = "#3FBF6F" if acc else "#8A8A8A"
        al = 0.9 if acc else 0.5
        (gl,) = axL.plot([XS[i - 1], XS[i], XS[i + 1]], [a, bn, c],
                         color=col, lw=1.2, alpha=al, zorder=3)
        (gd,) = axL.plot([XS[i]], [bn], marker="o", markersize=4.0,
                         color=col, alpha=min(al + 0.1, 1.0), zorder=4)
        ghost_artists += [gl, gd]

    path_line.set_ydata(y)
    steps_hist.append(total_props)
    S_hist.append(S)
    S_max = max(S_max, S)
    conv_line.set_data(steps_hist, S_hist)
    axR.set_xlim(0, max(X_INIT, total_props))            # fixed x-window first, then live
    axR.set_ylim(S_TARGET - Y_PAD, S_max + Y_PAD)         # grow-to-fit: never clips the curve

    maxdev = float(np.max(np.abs(y - Y_PARABOLA)))
    set_title(tau, total_props, S, maxdev)

    if maxdev < TOL and not state["converged"]:
        state["converged"] = True
    return [path_line, conv_line, title] + ghost_artists


def frame_source():
    yield 0
    f = 0
    while not state["converged"] and f < MAXFRAMES + INTRO_FRAMES:
        f += 1
        yield f


def main():
    anim = animation.FuncAnimation(fig, update, frames=frame_source,
                                   interval=INTERVAL, blit=False, repeat=False,
                                   cache_frame_data=False,
                                   save_count=MAXFRAMES + INTRO_FRAMES + 2)

    if RUN_MODE == "save":
        ext = os.path.splitext(SAVE_PATH)[1].lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=16)
        else:                                  # .mp4 / .mov / .m4v ... -> real video via ffmpeg
            writer = animation.FFMpegWriter(
                fps=OUTPUT_FPS,
                codec="libx264",
                bitrate=8000,
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "high",
                    "-level", "4.1",
                    "-movflags", "+faststart",
                ],
            )
        anim.save(SAVE_PATH, writer=writer)
        print("saved %s  converged=%s S=%.3f target=%.3f maxdev=%.3f Speak=%.2f frames~%d"
              % (SAVE_PATH, state["converged"], S, S_TARGET,
                 float(np.max(np.abs(y - Y_PARABOLA))),
                 (max(S_hist) if S_hist else S0), len(steps_hist)))
    else:
        plt.show()


if __name__ == "__main__":
    main()
