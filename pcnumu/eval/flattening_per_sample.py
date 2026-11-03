"""Preview per-sample FlatSpectraWeights: one weighter (own alpha) per sample, then combine.

Not what training does today (create_weighter builds ONE weighter on the pooled first target).
This only previews the distribution the loss would see if each sample were flattened on its own.

For each sample k and each alpha: w_k = FlatSpectraWeights(y_k, bins, range, alpha).getweight(y_k),
then rescaled so sum(w_k) = N_k (sample mix and mean event weight 1 are kept; the overall scale is a
separate choice). "Core" range of a sample = [p1, min(p99, xmax)] of its trueE; flatness = max/min of
the weighted density in 0.5 GeV bins over the core.

Usage: python -I pcnumu/eval/flattening_per_sample.py --cache <train_<horn>.npz> --out <dir> --tag fhc
       [--range 0,10] [--bins 500] [--target-flatness 1.5]
The cache is the npz written by flattening_scan.py --cache (arrays y = trueE, s = sample, train rows).
"""

import argparse
import csv
import os
import sys

import numpy as np

SAMPLES = {1: "Sample 1 (contained)", 2: "Sample 2 (peripheral, μ contained)",
           3: "Sample 3 (peripheral, μ uncontained)"}
COLORS = {1: "#2a6fdb", 2: "#e8643c", 3: "#1aa37a", 0: "#222222"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--alphas", default="0,0.001,0.0025,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--bins", type=int, default=500)
    ap.add_argument("--range", default=None, help="lo,hi for the weighter histogram (default: each sample's min-max)")
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--target-flatness", type=float, default=1.5)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    d = np.load(a.cache)
    y, s = d["y"], d["s"]
    rng = tuple(float(v) for v in a.range.split(",")) if a.range else None
    rtag = f"range {rng[0]:g}–{rng[1]:g} GeV" if rng else "range = each sample's full min–max"
    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]
    edges = np.linspace(0, a.xmax, int(a.xmax * 10) + 1)
    fl_edges = np.arange(0, a.xmax + 1e-9, 0.5)
    cmap = plt.get_cmap("viridis")

    def weights(mask, alpha, bins=a.bins, r=rng):
        yk = y[mask]
        w = FlatSpectraWeights(yk, bins=bins, range=r, alpha=alpha).getweight(yk)
        return w * (yk.size / w.sum())

    core, W, rows = {}, {}, []
    for k in SAMPLES:
        m = s == k
        lo, hi = np.percentile(y[m], [1, 99])
        core[k] = (lo, min(hi, a.xmax))
        for al in alphas:
            w = weights(m, al)
            W[(k, al)] = w
            yk = y[m]
            h, _ = np.histogram(yk, bins=fl_edges, weights=w)
            c = 0.5 * (fl_edges[1:] + fl_edges[:-1])
            sel = (c > core[k][0]) & (c < core[k][1]) & (h > 0)
            rows.append({"sample": k, "alpha": al, "n": yk.size,
                         "core_lo": round(core[k][0], 2), "core_hi": round(core[k][1], 2),
                         "flatness_max_over_min": h[sel].max() / h[sel].min(),
                         "ess_frac": w.sum() ** 2 / (w ** 2).sum() / w.size,
                         "w_max_over_median": w.max() / np.median(w),
                         "weight_frac_above_xmax": w[yk > a.xmax].sum() / w.sum()})
    # auto pick: largest alpha with flatness <= target (largest = least weight spread)
    pick = {}
    for k in SAMPLES:
        ok = [r for r in rows if r["sample"] == k and r["flatness_max_over_min"] <= a.target_flatness]
        pick[k] = max(ok, key=lambda r: r["alpha"])["alpha"] if ok else min(alphas)
    for r in rows:
        r["picked"] = int(pick[r["sample"]] == r["alpha"])
    with open(os.path.join(a.out, "per_sample_metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})

    title = f"{a.tag} · per-sample FlatSpectraWeights, bins={a.bins}, {rtag}"

    # 1) per sample: weighted shape vs alpha (linear 0-xmax, log 0-30), + metrics
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for j, k in enumerate(SAMPLES):
        m = s == k
        yk = y[m]
        for row, (ed, logy) in enumerate(((edges, False), (np.linspace(0, 30, 121), True))):
            ax = axs[row, j]
            h0, _ = np.histogram(yk, bins=ed)
            ax.stairs(h0 / h0.sum(), ed, color="0.6", ls="--", lw=1, label="unweighted")
            for i, al in enumerate(alphas):
                h, _ = np.histogram(yk, bins=ed, weights=W[(k, al)])
                ax.stairs(h / h.sum(), ed, color=cmap(i / (len(alphas) - 1)),
                          lw=2.2 if al == pick[k] else 1.1, label=f"α={al:g}" + (" ◀ pick" if al == pick[k] else ""))
            ax.axvspan(*core[k], color="0.85", alpha=0.35, lw=0)
            if logy:
                ax.set_yscale("log")
                ax.axvline(a.xmax, color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.25)
            ax.set_xlabel("true trueE [GeV]", fontsize=8)
            ax.set_ylabel("fraction / bin (area-normalized)", fontsize=8)
        axs[0, j].set_title(f"{SAMPLES[k]}  N={m.sum():,}\ngrey band = core [p1, min(p99,{a.xmax:g})]", fontsize=9.5)
        axs[0, j].legend(fontsize=7, ncol=2)
        ax = axs[2, j]
        rk = [r for r in rows if r["sample"] == k]
        xa = [max(r["alpha"], 3e-4) for r in rk]
        ax.plot(xa, [r["flatness_max_over_min"] for r in rk], "o-", color=COLORS[k], label="flatness max/min (core, 0.5 GeV)")
        ax.axhline(a.target_flatness, color=COLORS[k], ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax2 = ax.twinx()
        ax2.plot(xa, [r["ess_frac"] for r in rk], "s--", color="0.3", ms=3, label="ESS / N")
        ax2.plot(xa, [r["weight_frac_above_xmax"] for r in rk], "^:", color="0.55", ms=3, label=f"weight frac > {a.xmax:g} GeV")
        ax2.set_ylim(0, 1)
        ax.axvline(pick[k] if pick[k] > 0 else 3e-4, color="crimson", lw=0.8)
        ax.set_xlabel("alpha (0 at 3e-4)", fontsize=8)
        ax.set_title(f"pick α={pick[k]:g} (largest with flatness ≤ {a.target_flatness:g})", fontsize=9)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, which="both")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "per_sample_alpha.png"), dpi=110)
    plt.close(fig)

    # 2) combined: per-sample picks (sample mix kept) vs pooled weighters
    wc = np.empty_like(y)
    for k in SAMPLES:
        wc[s == k] = W[(k, pick[k])]
    combos = [(f"per-sample α={{1:{pick[1]:g}, 2:{pick[2]:g}, 3:{pick[3]:g}}}", wc)]
    for al in (0.01, 0.05):
        wp = FlatSpectraWeights(y, bins=a.bins, range=rng, alpha=al).getweight(y)
        combos.append((f"pooled α={al:g}", wp * (y.size / wp.sum())))
    fig, axs = plt.subplots(len(combos), 2, figsize=(14, 3.6 * len(combos)))
    for i, (name, w) in enumerate(combos):
        for col, (ed, logy) in enumerate(((edges, False), (np.linspace(0, 30, 121), True))):
            ax = axs[i, col]
            for k in SAMPLES:
                m = s == k
                h, _ = np.histogram(y[m], bins=ed, weights=w[m])
                ax.stairs(h, ed, color=COLORS[k], lw=1.2, label=SAMPLES[k])
            h, _ = np.histogram(y, bins=ed, weights=w)
            ax.stairs(h, ed, color=COLORS[0], lw=1.5, label="All")
            if logy:
                ax.set_yscale("log")
                ax.axvline(a.xmax, color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"weighted events / {ed[1]-ed[0]:.2g} GeV", fontsize=8)
        ess = w.sum() ** 2 / (w ** 2).sum() / w.size
        shares = ", ".join(f"s{k} {w[s == k].sum() / w.sum():.0%}" for k in SAMPLES)
        axs[i, 0].set_title(f"{name}   (ESS {ess:.0%}; weight share {shares}; "
                            f"> {a.xmax:g} GeV: {w[y > a.xmax].sum() / w.sum():.0%})", fontsize=9, loc="left")
    axs[0, 0].legend(fontsize=7.5)
    for ax in axs[-1]:
        ax.set_xlabel("true trueE [GeV]")
    fig.suptitle(f"{title}\nall rows normalized to mean event weight 1 (sample mix kept); left 0–{a.xmax:g} linear, right 0–30 log",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "combined.png"), dpi=110)
    plt.close(fig)
    print("picks", pick, "->", a.out)


if __name__ == "__main__":
    main()
