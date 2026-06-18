"""
Simulated-annealing Monte-Carlo recovery of a KEPLER ORBIT arc (least action).
Pure Python + matplotlib. Same method as the projectile demo, in 2-D:

  * The path is N+1 nodes (x_i, y_i) in the orbital plane; the two endpoints
    are fixed, all interior nodes are free (BOTH x and y move).
  * Action  S = sum [ 1/2 ( (dx/dt)^2 + (dy/dt)^2 )  +  mu / r ] dt
    (units: m = 1, GM = mu = 1).  The Sun sits at the focus r = 0.
  * Move: nudge one random node by (dx,dy) ~ Normal(0, sigma).
  * Accept (Metropolis): always if dS < 0, else with probability exp(-dS/T).
  * Cool T geometrically; auto-shrink/grow sigma to hold a healthy accept rate.
  * Stop when the path is within MC_TOL of the true orbit arc.

NOTE: Kepler's 1/r action is NON-CONVEX, so (unlike the projectile parabola)
convergence is approximate and seed-dependent; the chord visibly inflates into
the elliptical arc and settles near the analytic orbit.

============================ HOW TO RUN ============================
    RUN_MODE = "live"  -> interactive window      (or MC_MODE=live)
    RUN_MODE = "save"  -> video to SAVE_PATH       (or MC_MODE=save, MC_OUT=...)
    python3 kepler_action.py
    MC_MODE=save MC_OUT=kepler.mp4 python3 mc_kepler.py
Saving a video needs ffmpeg on PATH.
====================================================================
"""

import os
import numpy as np
import matplotlib

RUN_MODE = os.environ.get("MC_MODE", "live")
SAVE_PATH = os.environ.get("MC_OUT", "kepler_least_action_davinci.mp4")
if RUN_MODE == "save":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------------
# Orbit (normalised: m = 1, GM = mu = 1, semi-major axis a = 1)
# ---------------------------------------------------------------------------
MU, A, E = 1.0, 1.0, 0.3
NMEAN = np.sqrt(MU / A ** 3)          # mean motion (= 1 here)
B = A * np.sqrt(1 - E ** 2)


def kepler_E(M):
    Es = np.atleast_1d(M).astype(float)
    out = Es + E * np.sin(Es)
    for _ in range(60):
        out = out - (out - E * np.sin(out) - Es) / (1 - E * np.cos(out))
    return out


def orbit(t):
    Ecc = kepler_E(NMEAN * np.atleast_1d(t))
    return A * (np.cos(Ecc) - E), B * np.sin(Ecc)


# full ellipse (for reference) and the target arc
E_FULL = np.linspace(0, 2 * np.pi, 400)
ELL_X = A * (np.cos(E_FULL) - E)
ELL_Y = B * np.sin(E_FULL)

E1, E2 = 2 * np.pi / 3, 4 * np.pi / 3          # arc over the far (aphelion) side
T1 = E1 - E * np.sin(E1)
T2 = E2 - E * np.sin(E2)

N = int(os.environ.get("MC_N", 100))
DT = (T2 - T1) / N
TS = np.linspace(T1, T2, N + 1)
XC, YC = orbit(TS)                              # analytic arc (target)

KE_C = 0.5 / DT


def action(X, Y):
    dx = np.diff(X); dy = np.diff(Y); r = np.sqrt(X ** 2 + Y ** 2)
    return float(np.sum(KE_C * (dx ** 2 + dy ** 2)
                        + 0.5 * (MU / r[:-1] + MU / r[1:]) * DT))


def node_terms(X, Y, i):
    a1 = (X[i] - X[i - 1]) ** 2 + (Y[i] - Y[i - 1]) ** 2
    a2 = (X[i + 1] - X[i]) ** 2 + (Y[i + 1] - Y[i]) ** 2
    return KE_C * (a1 + a2) + MU / np.sqrt(X[i] ** 2 + Y[i] ** 2) * DT


# ---------------------------------------------------------------------------
# MC / annealing settings
# ---------------------------------------------------------------------------
TOL = float(os.environ.get("MC_TOL", 0.08))
SIGMA0 = float(os.environ.get("MC_SIGMA", 0.10))
SEED = int(os.environ.get("MC_SEED", 1))
SWEEPS = int(os.environ.get("MC_SWEEPS", 6))
TAU0 = float(os.environ.get("MC_TAU0", 1.0))
COOL = float(os.environ.get("MC_COOL", 0.994))
TAU_MIN = 1e-6
MAXFRAMES = int(os.environ.get("MC_MAXFRAMES", 3000))
GHOST_SHOW = 10
ACC_HI, ACC_LO = 0.40, 0.25

INTERVAL = 70
OUTPUT_FPS = int(os.environ.get("MC_FPS", 30))
INTRO_FRAMES = max(1, round(1000 / INTERVAL))
X_INIT = 30000
Y_PAD = 0.4

rng = np.random.default_rng(SEED)
X = np.linspace(XC[0], XC[-1], N + 1)
Y = np.linspace(YC[0], YC[-1], N + 1)
S = action(X, Y)
S0 = S
S_CL = action(XC, YC)
MAXDEV0 = float(np.max(np.hypot(X - XC, Y - YC)))

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
    for k in range(n_props):
        i = rng.integers(1, N)
        dx, dy = rng.normal(0.0, sig, 2)
        old = node_terms(X, Y, i)
        ox, oy = X[i], Y[i]
        X[i] = ox + dx; Y[i] = oy + dy
        dS = node_terms(X, Y, i) - old
        accept = (dS < 0.0) or (rng.random() < np.exp(-dS / max(tau, 1e-12)))
        if accept:
            S += dS; acc += 1
        else:
            X[i], Y[i] = ox, oy
        if k % stride == 0 and len(tried) < GHOST_SHOW:
            tried.append((X[i], Y[i], accept))
    total_props += n_props
    return tried, acc / n_props


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white"})
fig = plt.figure(figsize=(13, 6.4))
fig.patch.set_facecolor("#0e0e0e")
axL = fig.add_axes([0.035, 0.06, 0.46, 0.86]); axL.set_facecolor("#0e0e0e")
axR = fig.add_axes([0.575, 0.30, 0.40, 0.40]); axR.set_facecolor("#0e0e0e")

axL.set_xlim(-2.2, 1.4); axL.set_ylim(-1.7, 1.7); axL.set_aspect("equal")
axL.axis("off")
axL.plot(ELL_X, ELL_Y, color="#FFF1A6", lw=1.3, ls="--", alpha=0.45, zorder=1)   # true orbit
axL.plot([XC[0], XC[-1]], [YC[0], YC[-1]], "o", color="#E8C84F",
         markersize=7, zorder=5)                                                  # fixed endpoints
axL.plot([0], [0], marker="*", color="#FFD23F", markersize=20, zorder=6)          # Sun (focus)
axL.plot(X, Y, color="#5A5A5A", lw=1.3, ls="--", alpha=0.85, zorder=2)            # initial chord
(path_line,) = axL.plot(X.copy(), Y.copy(), color="#4FC3F7", lw=1.8,
                        marker="o", markersize=3.2, mfc="#4FC3F7",
                        mec="#4FC3F7", zorder=7)
path_line.set_visible(False)

axR.set_xlim(0, X_INIT); axR.set_ylim(S_CL - Y_PAD, S0 + Y_PAD)
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)
axR.spines["left"].set_color("white"); axR.spines["bottom"].set_color("white")
axR.set_xlabel("Iterations"); axR.set_ylabel("action  S")
axR.axhline(S_CL, color="#FFF1A6", ls="--", lw=1.4, alpha=0.85)
(conv_line,) = axR.plot([], [], color="#4FC3F7", lw=1.6)

title = fig.suptitle("", color="white", fontsize=13)
ghost_artists = []


def set_title(tau, props, s, md):
    title.set_text(f"T = {tau:.4f}     Iterations = {props}     "
                   f"S = {s:.3f}     max deviation = {md:.3f}")


def update(frame):
    global ghost_artists, sigma, S_max
    for art in ghost_artists:
        art.remove()
    ghost_artists = []

    mc = frame - INTRO_FRAMES
    if mc < 0:
        path_line.set_visible(False)
        set_title(TAU0, 0, S0, MAXDEV0)
        return [path_line, conv_line, title]

    if mc == 0:
        path_line.set_visible(True)
        path_line.set_data(X, Y)
        if not steps_hist:
            steps_hist.append(0); S_hist.append(S)
        conv_line.set_data(steps_hist, S_hist)
        set_title(TAU0, total_props, S, MAXDEV0)
        return [path_line, conv_line, title]

    tau = max(TAU0 * COOL ** mc, TAU_MIN)
    tried, ar = run_frame(tau, sigma)
    if ar > ACC_HI:
        sigma = min(sigma * 1.06, 0.6)
    elif ar < ACC_LO:
        sigma = max(sigma * 0.94, 5e-4)

    for (gx, gy, acc) in tried:
        col = "#3FBF6F" if acc else "#8A8A8A"
        (gd,) = axL.plot([gx], [gy], marker="o", markersize=3.4,
                         color=col, alpha=0.7, zorder=4)
        ghost_artists.append(gd)

    path_line.set_data(X, Y)
    steps_hist.append(total_props); S_hist.append(S)
    S_max = max(S_max, S)
    conv_line.set_data(steps_hist, S_hist)
    axR.set_xlim(0, max(X_INIT, total_props))
    axR.set_ylim(S_CL - Y_PAD, S_max + Y_PAD)

    maxdev = float(np.max(np.hypot(X - XC, Y - YC)))
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
        else:
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
        print("saved %s converged=%s S=%.3f S_cl=%.3f maxdev=%.3f frames~%d"
              % (SAVE_PATH, state["converged"], S, S_CL,
                 float(np.max(np.hypot(X - XC, Y - YC))), len(steps_hist)))
    else:
        plt.show()


if __name__ == "__main__":
    main()
