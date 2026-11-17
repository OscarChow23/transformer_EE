"""Does S3 lean towards its own spectrum (sample flag used) or the mixed all-sample spectrum (dominated by S1)?
Toy peak pull for S3 events with the 'crowd' = S3-only weighted spectrum vs all-samples weighted spectrum,
for the actual r05/r06 pooled weights; compare with measured S3 peak pull."""
import numpy as np
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
def toy(y_events, y_crowd, w_crowd, sig):
    h, _ = np.histogram(y_crowd, ns["bins"], weights=w_crowd); xc = ns["xc"]
    grid = np.arange(0.3, 12, 0.02); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post); est[i] = xc[np.searchsorted(c, c[-1] / 2)]
    t = ns["rng"].choice(y_events, 400000); m = t * (1 + sig * ns["rng"].standard_normal(t.size)); r = np.interp(m, grid, est) / t - 1
    return 100 * (np.median(r[(t > 1) & (t < 1.5)]) - np.median(r[(t > 3) & (t < 4)]))
MEAS = {("fhc", 0.05): 10.4, ("fhc", 0.01): 4.7, ("rhc", 0.05): 7.5, ("rhc", 0.01): 3.5}
for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz"); y, s = d["y"], d["s"]
    sig = ns["SIG"][horn]
    for a, run in ((0.05, "r05"), (0.01, "r06")):
        w = ns["FlatSpectraWeights"](y, bins=500, alpha=a).getweight(y)
        own = toy(y[s == 3], y[s == 3], w[s == 3], sig); mixed = toy(y[s == 3], y, w, sig)
        print(f"{horn.upper()} {run}: measured S3 peak pull {MEAS[(horn, a)]:+5.1f} | toy, crowd = S3 only {own:+5.1f} | toy, crowd = all samples mixed {mixed:+5.1f}")
