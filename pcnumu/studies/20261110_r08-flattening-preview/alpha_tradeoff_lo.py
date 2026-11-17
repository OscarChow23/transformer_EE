"""Follow-up: does S3's lower edge (1.3 GeV in the 4 Nov recipe) drive the toy peak pull? Also reports the two halves."""
import numpy as np
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
def halves(y, w, sig):
    h, _ = np.histogram(y, ns["bins"], weights=w); xc = ns["xc"]
    grid = np.arange(0.3, 12, 0.02); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post)
        est[i] = xc[np.searchsorted(c, c[-1] / 2)] if c[-1] > 0 else m
    t = ns["rng"].choice(y, 400000); m = t * (1 + sig * ns["rng"].standard_normal(t.size)); r = np.interp(m, grid, est) / t - 1
    return 100 * np.median(r[(t > 1) & (t < 1.5)]), 100 * np.median(r[(t > 3) & (t < 4)])
for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz")
    y_all, s = d["y"], d["s"]; y = y_all[s == 3]; sig = ns["SIG"][horn]
    print(f"--- {horn.upper()} S3   (lean at true 1-1.5 GeV / lean at 3-4 GeV; peak pull = difference)")
    for a in (0.01,):
        w = ns["FlatSpectraWeights"](y_all, bins=500, alpha=a).getweight(y_all)[s == 3]; a1, a3 = halves(y, w, sig)
        print(f"r06 pooled α0.01                               {a1:+5.1f} / {a3:+5.1f}   peak pull {a1-a3:+5.1f}")
    for lo in (1.3, 1.0, 0.7, 0.5):
        for hi, tp in ((5, 3), (10, 2)):
            for a in (0.05, 0.02, 0.001):
                w = ns["persample"](y, lo, hi, tp, a); mx, ess = ns["describe"](y, w); a1, a3 = halves(y, w, sig)
                print(f"from {lo} to {hi:>2} GeV, ease {tp}, α{a:<6} weight≤{mx:5.1f}x ESS {ess:4.1f}%  {a1:+5.1f} / {a3:+5.1f}   peak pull {a1-a3:+5.1f}")
