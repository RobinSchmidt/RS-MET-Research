import json, numpy as np, prosampler as ps
out = []
for r in (0.008, 0.016, 0.03, 0.05, 0.1, 0.2):
    fp = ps.best_pair_for_length(4, 0.85, 290, ripple_db=r, transition_max_db="ripple", n_jobs=2, progress=False)
    m = fp.metrics(); b = fp.info["best_for_length"]
    row = dict(ripple_req=r, n_u=fp.n_u, n_d=fp.n_d, worst_db=max(m["u_stop_db"], m["d_stop_db"]), u_stop=m["u_stop_db"], d_stop=m["d_stop_db"],
               bump=max(m["u_transition_peak_db"], m["d_transition_peak_db"]), ripple=max(m["u_ripple_db"], m["d_ripple_db"]),
               table=b["table"], elapsed=b["elapsed_s"])
    out.append(row); json.dump(out, open("/tmp/caps/sweep.json", "w"))
    np.savez(f"/tmp/caps/sweep_{r}.npz", u=fp.u, d=fp.d, latency=fp.latency)
    print("ROW", r, fp.n_u, fp.n_d, round(row["worst_db"], 2), round(row["bump"], 4), round(row["ripple"], 4), round(row["elapsed"]), flush=True)
