"""What a tolerated pull X implies (planning, no training). Rule of thumb: pull ~ sigma^2 * E * |dlnN/dE|.
Target spectrum per sample = raw spectrum with its log-slope limited to s_max(E) = X / (sigma^2 E),
filled in from the peak outwards (never cut down, only raised), over [lo, 5 GeV]; weight = target/raw.
sigma = r05 trueE resolution per sample (scorecard, 0.5-6 GeV), treated as constant relative width."""
import numpy as np
SIG = {"fhc": {1: .0675, 2: .1013, 3: .1518}, "rhc": {1: .0612, 2: .0920, 3: .1724}}
LO = {"fhc": {1: 1.0, 2: 1.3, 3: 1.3}, "rhc": {1: 0.9, 2: 1.3, 3: 1.3}}
HI, DE = 5.0, 0.1

def target_weights(y, lo, sig, X):
    e = np.arange(lo, HI + 1e-9, DE); c, _ = np.histogram(y, e); c = np.maximum(c, 1)
    xc = 0.5 * (e[1:] + e[:-1]); ln = np.log(c); t = ln.copy(); p = ln.argmax()
    smax = X / (sig ** 2 * xc)
    for i in range(p + 1, len(t)):          # above peak: may fall at most smax per GeV
        t[i] = max(ln[i], t[i - 1] - smax[i] * DE)
    for i in range(p - 1, -1, -1):          # below peak (rising side), same limit
        t[i] = max(ln[i], t[i + 1] - smax[i] * DE)
    wb = np.exp(t - ln)                      # boost relative to a peak event (peak bin = 1)
    w = np.ones_like(y, dtype=float)         # outside [lo, HI]: like a peak event
    k = (y >= lo) & (y < HI); w[k] = wb[np.clip(np.digitize(y[k], e) - 1, 0, len(wb) - 1)]
    return w, xc, wb

if __name__ == "__main__":
    for horn in ("fhc", "rhc"):
        d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz")
        for smp in (1, 2, 3):
            y = d["y"][d["s"] == smp]
            for X in (0.03, 0.05):
                w, xc, wb = target_weights(y, LO[horn][smp], SIG[horn][smp], X)
                ess = w.sum() ** 2 / (w ** 2).sum() / w.size
                prof = " ".join(f"{E}:{wb[np.searchsorted(xc, E)]:.1f}" for E in (1.2, 1.5, 2.5, 3, 3.5, 4, 4.5, 4.9))
                print(f"{horn} S{smp} σ{SIG[horn][smp]:.3f} X{int(X*100)}%: max boost {w.max():5.1f}x  ESS {100*ess:4.1f}%  boost@E {prof}")
