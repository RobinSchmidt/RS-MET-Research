"""Plot the pairs built by build_bump_caps.py (default directory bump_caps_data) -> bump_caps.png."""
import sys
CAPS = sys.argv[1] if len(sys.argv) > 1 else "bump_caps_data"
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, prosampler as ps
L, P = 4, 0.85
names = [("plain", "plain pair (314 taps)", "0.6"), ("cap0.2", "cap +0.2 dB", "C0"), ("cap0.1", "cap +0.1 dB", "C2"), ("cap0.0", "cap 0 dB", "C3")]
sets = []
for k, lab, col in names:
    f = CAPS + f"/{k}.npz"
    if os.path.exists(f):
        z = np.load(f); sets.append((lab + f" [{len(z['u'])}/{len(z['d'])}]", col, z["u"], z["d"]))
def mag(h, x):
    return 20 * np.log10(np.abs(ps.freqz(h, x * np.pi / L)) / abs(np.sum(h)) + 1e-30)
fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]], dtype=object)
x = np.linspace(0, L, 6000)
for lab, col, u, d in sets:
    ax[0, 0].plot(x, mag(u, x), color=col, lw=0.8, label=lab); ax[0, 1].plot(x, mag(d, x), color=col, lw=0.8, label=lab)
ax[0, 0].set_title("u (relative to its DC gain), full range"); ax[0, 1].set_title("d, full range")
for a in ax[0][:2]:
    a.set_ylim(-90, 5); a.axhline(-60, color="k", lw=0.5, ls="--"); a.axvline(2 - P, color="0.7", lw=0.5); a.set_xlabel("frequency / original Nyquist"); a.set_ylabel("dB"); a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8)
ax[0, 2].axis("off"); rows = []
for k, lab, col in names:
    f = CAPS + f"/{k}.npz"
    if os.path.exists(f):
        z = np.load(f); mm = ps.FilterPair(z["u"], z["d"], L, int(z["latency"]), P).metrics()
        rows.append(f"{lab}\n   taps {len(z['u'])}+{len(z['d'])} = {len(z['u'])+len(z['d'])},  stopband {max(mm['u_stop_db'], mm['d_stop_db']):.1f} dB\n   bump {max(mm['u_transition_peak_db'], mm['d_transition_peak_db']):+.2f} dB,  ripple {max(mm['u_ripple_db'], mm['d_ripple_db']):.4f} dB")
ax[0, 2].text(0.0, 0.95, "\n\n".join(rows), va="top", family="monospace", fontsize=8.5)
xt = np.linspace(0.7, 1.3, 4000)
for lab, col, u, d in sets:
    ax[1, 0].plot(xt, mag(d, xt), color=col, lw=1.2, label="d " + lab); ax[1, 0].plot(xt, mag(u, xt), color=col, lw=1.0, ls=":")
ax[1, 0].set_title("transition band: d (solid), u (dotted)"); ax[1, 0].set_ylim(-40, 1.6); ax[1, 0].axhline(0, color="k", lw=0.5)
ax[1, 0].axvline(P, color="0.7", lw=0.5); ax[1, 0].axvline(2 - P, color="0.7", lw=0.5)
xz = np.linspace(0.75, 1.0, 3000)
for lab, col, u, d in sets:
    ax[1, 1].plot(xz, mag(d, xz), color=col, lw=1.4); ax[1, 1].plot(xz, mag(u, xz), color=col, lw=1.0, ls=":")
ax[1, 1].set_ylim(-1.5, 1.4); ax[1, 1].axhline(0, color="k", lw=0.5); ax[1, 1].axvline(P, color="0.7", lw=0.5)
ax[1, 1].set_title("passband edge detail: d (solid), u (dotted)")
xp = np.linspace(0, P, 3000)
for lab, col, u, d in sets:
    ax[1, 2].plot(xp, mag(d, xp), color=col, lw=1.0); ax[1, 2].plot(xp, mag(u, xp), color=col, lw=1.0, ls=":")
ax[1, 2].set_title("passband ripple: d (solid), u (dotted)"); ax[1, 2].set_ylim(-0.012, 0.012)
for a in ax[1]:
    a.set_xlabel("frequency / original Nyquist"); a.set_ylabel("dB"); a.grid(alpha=0.3)
ax[1, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("0.85 / 60 dB, L = 4: shortest pairs found for different bump caps (equal passband ripple)")
fig.savefig(CAPS + "/bump_caps.png", dpi=130)
