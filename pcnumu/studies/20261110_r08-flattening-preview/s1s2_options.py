"""Same S3 method (alpha_tradeoff.py) applied to S1 and S2: strength (largest weight, ESS) and toy lean per option.
sigma = r05 trueE resolution per sample. Lean reported at true 1-1.5 GeV and 3-4 GeV and their difference (peak pull)."""
import numpy as np
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
SIG = {"fhc": {1: .0675, 2: .1013, 3: .1518}, "rhc": {1: .0612, 2: .0920, 3: .1724}}
def halves(y, w, sig):
    h, _ = np.histogram(y, ns["bins"], weights=w); xc = ns["xc"]
    grid = np.arange(0.3, 12, 0.01); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post); est[i] = xc[np.searchsorted(c, c[-1] / 2)]
    t = ns["rng"].choice(y, 400000); m = t * (1 + sig * ns["rng"].standard_normal(t.size)); r = np.interp(m, grid, est) / t - 1
    a, b = 100 * np.median(r[(t > 1) & (t < 1.5)]), 100 * np.median(r[(t > 3) & (t < 4)]); return a, b
for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz"); y_all, s = d["y"], d["s"]
    for smp in (1, 2, 3):
        y = y_all[s == smp]; sig = SIG[horn][smp]; print(f"--- {horn.upper()} S{smp} (σ {sig:.3f})")
        rows = [("no reweighting", np.ones_like(y))]
        for n, a in (("r05 pooled", 0.05), ("r06 pooled", 0.01)):
            rows.append((n, ns["FlatSpectraWeights"](y_all, bins=500, alpha=a).getweight(y_all)[s == smp]))
        for lo in (1.0, 0.7):
            for a in (0.1, 0.05, 0.02):
                rows.append((f"per-sample {lo}-5 GeV ease 3, α{a}", ns["persample"](y, lo, 5.0, 3.0, a)))
        for n, w in rows:
            mx, ess = ns["describe"](y, w); a1, a3 = halves(y, w, sig)
            print(f"  {n:34s} weight≤{mx:6.1f}x  ESS {ess:4.1f}%  lean 1-1.5 {a1:+5.1f}  3-4 {a3:+5.1f}  peak pull {a1-a3:+5.1f}")
