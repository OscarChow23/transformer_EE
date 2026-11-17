"""r08 planning preview (no training): per-sample weights and weighted spectra for three flattening regimes.

Weights follow pcnumu/eval/flattening_taper_scan.py:taper_weights, with one change agreed 10 Nov (Q9):
events below the lower edge get weight 1 instead of the first flat bin's weight.
Run: ~/miniforge3/envs/transformer_ee/bin/python -I plot_regimes.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = "/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{}.npz"
OUT = "/data/nova/chow/pcnumu-runs/analysis/r08-flattening-preview/"
LO = {"fhc": {1: 1.0, 2: 1.3, 3: 1.3}, "rhc": {1: 0.9, 2: 1.3, 3: 1.3}}
# (label, alpha, {sample: (edge, taper)}); "cliff" = edge where the 0.1 GeV count first drops below 10% of peak
REGIMES = {
    "fhc": [("4 Nov edges, α 0.001", 0.001, {1: (5, 1), 2: (6, 2), 3: (10, 2)}),
            ("4 Nov edges, α 0.02", 0.02, {1: (5, 1), 2: (6, 2), 3: (10, 2)}),
            ("edge at cliff (10% of peak), α 0.001", 0.001, {1: (3.0, 1), 2: (3.0, 1), 3: (3.0, 1)})],
    "rhc": [("4 Nov edges, α 0.001", 0.001, {1: (5, 1), 2: (6, 2), 3: (10, 2)}),
            ("4 Nov edges, α 0.02", 0.02, {1: (5, 1), 2: (6, 2), 3: (10, 2)}),
            ("edge at cliff (10% of peak), α 0.001", 0.001, {1: (3.0, 1), 2: (4.5, 1), 3: (4.0, 1)})],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz reference palette slots 1-3


def weights(y, lo, hi, taper, alpha, width=0.1):
    e = np.arange(lo, hi + taper + 1e-9, width)
    c, _ = np.histogram(y, e)
    wb = 1.0 / (c + max(alpha * c.max(), 1))
    w = wb[np.clip(np.digitize(y, e) - 1, 0, len(wb) - 1)]
    core = (y >= lo) & (y < hi)
    w = w * (core.sum() / w[core].sum())  # flat part averages 1 per event
    tz = (y >= hi) & (y < hi + taper)
    f = (y[tz] - hi) / taper
    w[tz] = w[tz] ** (1 - f)  # log-linear taper back to 1
    w[(y >= hi + taper) | (y < lo)] = 1.0
    return w * (y.size / w.sum())


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum() / w.size


rows = []
for horn in ("fhc", "rhc"):
    d = np.load(CACHE.format(horn))
    y_all, s_all = d["y"], d["s"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    bins = np.arange(0, 12.0001, 0.1)
    for j, smp in enumerate((1, 2, 3)):
        y = y_all[s_all == smp]
        h0, _ = np.histogram(y, bins)
        ax[1, j].stairs(h0, bins, color="#8a8a85", lw=1.5, label="unweighted")
        peak_w = {}
        for (lab, a, edges), col in zip(REGIMES[horn], COLORS):
            hi, tp = edges[smp]
            w = weights(y, LO[horn][smp], hi, tp, a)
            pk = w[(y > 1.8) & (y < 2.2)].mean()
            # weight relative to a peak event: what "boost" means
            xs = np.arange(0.05, 12, 0.1)
            idx = np.clip(np.digitize(y, bins) - 1, 0, len(xs) - 1)
            mean_w = np.bincount(idx, w, len(xs)) / np.maximum(np.bincount(idx, minlength=len(xs)), 1)
            ok = np.bincount(idx, minlength=len(xs)) > 0
            ax[0, j].plot(xs[ok], mean_w[ok] / pk, color=col, lw=2)
            hw, _ = np.histogram(y, bins, weights=w)
            ax[1, j].stairs(hw, bins, color=col, lw=2, label=lab)
            hiE = y >= 3
            rows.append(f"{horn} S{smp} | {lab} | max boost vs peak {w.max() / pk:.1f}x | ESS {100 * ess(w):.1f}% "
                        f"| weight share ≥3 GeV {100 * w[hiE].sum() / w.sum():.1f}% (events {100 * hiE.mean():.1f}%)")
        ax[0, j].set_yscale("log")
        ax[0, j].axhline(1, color="#8a8a85", lw=1, ls=":")
        ax[0, j].set_title(f"{horn.upper()} sample {smp}  ({y.size:,} train events)")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("true neutrino energy (GeV)")
        ax[1, j].set_xlim(0, 12)
        for a_ in ax[:, j]:
            a_.grid(alpha=0.25, lw=0.5)
    ax[0, 0].set_ylabel("event weight ÷ weight at 2 GeV peak")
    ax[1, 0].set_ylabel("events per 0.1 GeV (weighted)")
    ax[1, 2].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{horn.upper()} per-sample flattening regimes (training rows; preview, no training)")
    fig.tight_layout()
    fig.savefig(OUT + f"regimes_{horn}.png", dpi=120)
open(OUT + "regimes_summary.txt", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
