"""Build the shortest pairs for bump caps 0 / +0.1 / +0.2 dB (0.85 / 60 dB, L = 4) and save them as .npz in CAPS.
Takes about 30 minutes (the +0.1 dB adaptive fine search on two cores).  Then run plot_bump_caps.py."""
import os, sys
CAPS = sys.argv[1] if len(sys.argv) > 1 else "bump_caps_data"
os.makedirs(CAPS, exist_ok=True)
import numpy as np, prosampler as ps
L, P, A = 4, 0.85, 60
plain = ps.design_pair(L, P, A); m = plain.metrics(); r_ref = max(m["u_ripple_db"], m["d_ripple_db"])
np.savez(CAPS + "/plain.npz", u=plain.u, d=plain.d, latency=plain.latency)
def save(name, fp):
    mm = fp.metrics(); np.savez(f"{CAPS}/{name}.npz", u=fp.u, d=fp.d, latency=fp.latency)
    print("SAVED", name, fp.n_u, fp.n_d, "worst", max(mm["u_stop_db"], mm["d_stop_db"]), "bump", max(mm["u_transition_peak_db"], mm["d_transition_peak_db"]), "ripple", max(mm["u_ripple_db"], mm["d_ripple_db"]), flush=True)
for name, cap, nu, nd in (("cap0.0", 0.0, 107, 183), ("cap0.2", 0.2, 103, 171)):
    st = ps.design_pair(L, P, A, n_u=nu, n_d=nd, objective="minimax")
    save(name, ps.optimise_joint_slp(st, ripple_db=r_ref, transition_max_db=cap, iterations=30))
fp = ps.search_pair(L, P, A, transition_max_db=0.1, resolution="fine", strategy="adaptive", n_jobs=2)
save("cap0.1", fp)
print("DONE", fp.info["search"]["elapsed_s"], len(fp.info["search"]["table"]), flush=True)
