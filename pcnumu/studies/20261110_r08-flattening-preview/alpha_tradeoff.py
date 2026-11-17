"""Planning (no training): put r05/r06 (pooled FlatSpectraWeights) and per-sample alpha options on one scale for S3.
Strength of reweighting: ESS within S3 and largest weight vs a peak event. Lean: toy peak pull, i.e. events drawn
from S3's natural spectrum, inputs blurred by sigma, estimate = median of (training spectrum x blur);
peak pull = median(est/true-1) at true 1-1.5 GeV minus at 3-4 GeV (same definition as the scorecard)."""
import numpy as np, sys
sys.path.insert(0, "/data/nova/chow/transformerEE-pcnumu-loop")
from transformer_ee.utils.weights import FlatSpectraWeights
src = open("plot_regimes.py").read(); ns = {}; exec(src.split("rows = []")[0], ns); persample = ns["weights"]
SIG = {"fhc": .1518, "rhc": .1724}; LO = 1.3
rng = np.random.default_rng(20261110)
bins = np.arange(0, 40.0001, 0.05); xc = 0.5 * (bins[1:] + bins[:-1])

def toy_peak_pull(y, w, sig):
    h, _ = np.histogram(y, bins, weights=w)
    grid = np.arange(0.3, 12, 0.02); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post)
        est[i] = xc[np.searchsorted(c, c[-1] / 2)] if c[-1] > 0 else m
    t = rng.choice(y, 400000); m = t * (1 + sig * rng.standard_normal(t.size))
    r = np.interp(m, grid, est) / t - 1
    a = np.median(r[(t > 1) & (t < 1.5)]); b = np.median(r[(t > 3) & (t < 4)])
    return 100 * (a - b)

def describe(y, w):
    pk = w[(y > 1.8) & (y < 2.2)].mean()
    return w.max() / pk, 100 * w.sum() ** 2 / (w ** 2).sum() / w.size

for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz")
    y_all, s = d["y"], d["s"]; y = y_all[s == 3]; sig = SIG[horn]
    rows = [("no reweighting", np.ones_like(y))]
    for name, a in (("r05 pooled α0.05", 0.05), ("r06 pooled α0.01", 0.01)):
        rows.append((name, FlatSpectraWeights(y_all, bins=500, alpha=a).getweight(y_all)[s == 3]))
    for a in (0.2, 0.1, 0.05, 0.02, 0.01, 0.001):
        rows.append((f"per-sample α{a}, flat to 5 GeV, ease off to 8", persample(y, LO, 5.0, 3.0, a)))
    print(f"--- {horn.upper()} S3 (σ {sig:.3f})")
    for name, w in rows:
        mx, ess = describe(y, w)
        print(f"{name:42s} biggest weight {mx:5.1f}x  ESS {ess:4.1f}%  toy peak pull {toy_peak_pull(y, w, sig):+5.1f}")
