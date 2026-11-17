import numpy as np
def capped(y, lo, hi, alpha, width=0.1):
    e = np.arange(lo, hi + 1e-9, width); c, _ = np.histogram(y, e)
    w = 1.0 / (c + max(alpha * c.max(), 1))
    plateau = np.median(w[-10:])                      # weight level at the top of the flat range
    i = np.clip(np.digitize(y, e) - 1, 0, len(w) - 1)
    out = w[i]; out[y >= hi] = plateau; out[y < lo] = w[0]
    return out * (y.size / out.sum())                 # mean event weight 1 within the sample
for h in ["fhc", "rhc"]:
    d = np.load("/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_" + h + ".npz"); y, s = d["y"], d["s"]
    print(f"\n== {h}: per-sample flat in [0.5, cap], above cap = plateau weight, alpha 0.01, sample mix kept")
    print(f"{'smp':>3} {'cap':>4} {'ESS':>6} {'w>10GeV':>8} {'w 5-10':>7} {'raw>10':>7} {'w max/med':>9} {'flat 1-cap':>10}")
    for k in (1, 2, 3):
        yk = y[s == k]
        for cap in (6, 8, 10):
            w = capped(yk, 0.5, cap, 0.01)
            e = np.arange(1.0, min(cap, 5.0 if k == 1 else cap) + 1e-9, 0.5); hh, _ = np.histogram(yk, e, weights=w)
            print(f"{k:>3} {cap:>4} {w.sum()**2/(w**2).sum()/w.size:>6.1%} {w[yk>10].sum()/w.sum():>8.1%} {w[(yk>5)&(yk<10)].sum()/w.sum():>7.1%} {np.mean(yk>10):>7.1%} {w.max()/np.median(w):>9.1f} {hh.max()/hh.min():>10.2f}")
