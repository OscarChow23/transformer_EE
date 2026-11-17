"""Follow-up to flag_test.py: compare the measured S3 bias curve (peak_metrics.json, test set) with the toy curves
for 'crowd = S3 only' and 'crowd = all samples mixed', bin by bin (median est/true - 1, %)."""
import json, numpy as np
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
PM = {("fhc", "r05"): "/data/nova/chow/pcnumu-runs/r05_v3data-allsamples_v3features/fhc/scorecard/peak_metrics.json",
      ("rhc", "r05"): "/data/nova/chow/pcnumu-runs/r05_v3data-allsamples_v3features/rhc/scorecard/peak_metrics.json",
      ("fhc", "r06"): "/data/nova/chow/pcnumu-runs/r06_flat-alpha001/fhc/scorecard/peak_metrics.json",
      ("rhc", "r06"): "/data/nova/chow/pcnumu-runs/r06_flat-alpha001/rhc/scorecard/peak_metrics.json"}
def toy_curve(y_events, y_crowd, w_crowd, sig, edges):
    h, _ = np.histogram(y_crowd, ns["bins"], weights=w_crowd); xc = ns["xc"]
    grid = np.arange(0.3, 12, 0.02); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post); est[i] = xc[np.searchsorted(c, c[-1] / 2)]
    t = ns["rng"].choice(y_events, 600000); m = t * (1 + sig * ns["rng"].standard_normal(t.size)); r = np.interp(m, grid, est) / t - 1
    return [100 * np.median(r[(t >= a) & (t < b)]) for a, b in edges]
for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz"); y, s = d["y"], d["s"]
    for run, a in (("r05", 0.05), ("r06", 0.01)):
        bc = json.load(open(PM[(horn, run)]))["by_sample"]["3"]["bias_curve"]
        edges = [(b["lo"], b["hi"]) for b in bc if b["lo"] >= 1.0 and b["hi"] <= 5.0]
        meas = [b["median_r_pct"] for b in bc if b["lo"] >= 1.0 and b["hi"] <= 5.0]
        w = ns["FlatSpectraWeights"](y, bins=500, alpha=a).getweight(y)
        own = toy_curve(y[s == 3], y[s == 3], w[s == 3], ns["SIG"][horn], edges)
        mix = toy_curve(y[s == 3], y, w, ns["SIG"][horn], edges)
        # remove each curve's overall offset (mean over bins) so only the shape is compared
        sh = lambda v: np.array(v) - np.mean(v)
        rms_own = np.sqrt(np.mean((sh(meas) - sh(own)) ** 2)); rms_mix = np.sqrt(np.mean((sh(meas) - sh(mix)) ** 2))
        print(f"\n{horn.upper()} {run} S3 bias curve (%), true E bin: measured | toy own | toy mixed")
        for (lo, hi), m_, o, x in zip(edges, meas, own, mix):
            if lo in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5): print(f"  {lo:4.2f}-{hi:4.2f}: {m_:+6.1f} | {o:+6.1f} | {x:+6.1f}")
        print(f"  shape mismatch (rms after removing offsets): own {rms_own:.2f}  mixed {rms_mix:.2f}")
