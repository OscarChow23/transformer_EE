"""Scan FlatSpectraWeights (alpha x bins) on the training rows of a trial config, before training.

Reads only the target / sample / split columns of the config's data_path, keeps split == "train"
(as load.py does), builds the weighter with the repo's FlatSpectraWeights on the first target,
and writes per horn:
  grid_<target>.png       weighted spectrum per sample, rows = alpha, cols = bins
  alpha_overlay_<target>.png   pooled + per-sample weighted spectra for each alpha at fixed bins
  metrics.csv / metrics.png    mean event weight, ESS, weight range, sample shares, flatness ratios

Usage (conda env transformer_ee, PYTHONPATH=<repo>):
  python -I pcnumu/eval/flattening_scan.py --config <effective_config.json> --out <dir>
"""

import argparse
import json
import os
import sys

import numpy as np

SAMPLES = {1: "Sample 1 (contained)", 2: "Sample 2 (peripheral, μ contained)",
           3: "Sample 3 (peripheral, μ uncontained)"}
COLORS = {1: "#2a6fdb", 2: "#e8643c", 3: "#1aa37a", 0: "#222222"}


def load_train(cfg, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["y"], d["s"]
    import polars as pl  # pylint: disable=C0415
    tgt = cfg["target"][0]
    sc = cfg.get("split_column", "split")
    df = (pl.scan_csv(cfg["data_path"]).select([tgt, "sample", sc])
          .filter(pl.col(sc) == "train").collect())
    y = df.get_column(tgt).to_numpy().astype(np.float64)
    s = df.get_column("sample").to_numpy().astype(np.int64)
    if cache:
        np.savez(cache, y=y, s=s)
    return y, s


def metrics(y, s, w, alpha, bins, edges_plot):
    out = {"alpha": alpha, "bins": bins, "bin_width_GeV": (y.max() - y.min()) / bins,
           "mean_event_weight": w.mean(), "ess_frac": w.sum() ** 2 / (w ** 2).sum() / w.size,
           "w_max_over_min": w.max() / w.min(), "w_p99_over_p1": np.percentile(w, 99) / np.percentile(w, 1)}
    tot = w.sum()
    for k in SAMPLES:
        m = s == k
        out[f"share_s{k}_raw"] = m.mean()
        out[f"share_s{k}_weighted"] = w[m].sum() / tot
    # flatness: weighted counts in the 1.5-2.5 GeV peak region vs 4-5 GeV (per GeV), pooled and sample 3
    def dens(mask, lo, hi):
        sel = mask & (y >= lo) & (y < hi)
        return w[sel].sum() / (hi - lo)
    allm = np.ones_like(s, bool)
    for tag, m in (("all", allm), ("s3", s == 3), ("s1", s == 1)):
        out[f"peak_over_4to5_{tag}"] = dens(m, 1.5, 2.5) / max(dens(m, 4, 5), 1e-12)
        out[f"peak_over_8to10_{tag}"] = dens(m, 1.5, 2.5) / max(dens(m, 8, 10), 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--alphas", default="0,0.001,0.0025,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--bins", default="50,100,250,500,1000")
    ap.add_argument("--overlay-bins", type=int, default=500)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    cfg = json.load(open(a.config))
    tgt = cfg["target"][0]
    kw0 = {k: v for k, v in cfg["weight"].get("kwargs", {}).items() if k not in ("alpha", "bins")}
    os.makedirs(a.out, exist_ok=True)
    y, s = load_train(cfg, a.cache)
    print(f"{tgt}: {y.size} train rows, range [{y.min():.3f}, {y.max():.3f}] GeV, "
          f"samples {dict(zip(*np.unique(s, return_counts=True)))}")
    alphas = [float(x) for x in a.alphas.split(",")]
    binss = [int(x) for x in a.bins.split(",")]
    edges = np.linspace(0, a.xmax, int(a.xmax * 10) + 1)
    title_tag = f"{os.path.basename(os.path.dirname(os.path.abspath(a.config)))} · {tgt} · {y.size:,} train rows"

    W, rows = {}, []
    for al in alphas:
        for b in binss:
            w = FlatSpectraWeights(y, bins=b, alpha=al, **kw0).getweight(y)
            W[(al, b)] = w
            rows.append(metrics(y, s, w, al, b, edges))
    import csv  # pylint: disable=C0415
    with open(os.path.join(a.out, "metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    def step(ax, mask, w, k, lw=1.1, **kws):
        h, _ = np.histogram(y[mask], bins=edges, weights=None if w is None else w[mask])
        ax.stairs(h, edges, color=COLORS[k], lw=lw, **kws)

    # 1) grid alpha x bins, per-sample weighted spectra (each panel own y-scale)
    nr, nc = len(alphas) + 1, len(binss)
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.0 * nr), sharex=True)
    axs = np.atleast_2d(axs)
    for j in range(nc):
        ax = axs[0, j]
        for k in SAMPLES:
            step(ax, s == k, None, k)
        step(ax, np.ones_like(s, bool), None, 0, lw=1.3)
        ax.set_title(f"bins = {binss[j]}  (Δ={(y.max()-y.min())/binss[j]*1000:.0f} MeV)", fontsize=9)
    axs[0, 0].set_ylabel("unweighted", fontsize=9)
    for i, al in enumerate(alphas, start=1):
        for j, b in enumerate(binss):
            ax = axs[i, j]
            w = W[(al, b)]
            for k in SAMPLES:
                step(ax, s == k, w, k)
            step(ax, np.ones_like(s, bool), w, 0, lw=1.3)
            r = rows[(i - 1) * nc + j]
            ax.text(0.98, 0.95, f"⟨w⟩={r['mean_event_weight']:.3f}\nESS={r['ess_frac']:.1%}\n"
                    f"s3 pk/4-5={r['peak_over_4to5_s3']:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, family="monospace")
        axs[i, 0].set_ylabel(f"α = {al:g}", fontsize=9)
    for ax in axs.flat:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axs[-1]:
        ax.set_xlabel(f"true {tgt} [GeV]", fontsize=8)
    handles = [plt.Line2D([], [], color=COLORS[k], label=SAMPLES[k]) for k in SAMPLES]
    handles.append(plt.Line2D([], [], color=COLORS[0], label="All samples"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"FlatSpectraWeights scan, weighted events / 0.1 GeV (shown 0–{a.xmax:g}; weights use full range)\n{title_tag}",
                 y=1.035, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(a.out, f"grid_{tgt}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2) alpha overlay at fixed bins: shape only (normalized to unit area in 0-xmax)
    b = a.overlay_bins
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [(0, "All samples", np.ones_like(s, bool))] + [(k, SAMPLES[k], s == k) for k in SAMPLES]
    for ax, (k, name, m) in zip(axs.flat, panels):
        h0, _ = np.histogram(y[m], bins=edges)
        ax.stairs(h0 / h0.sum(), edges, color="0.6", lw=1.0, ls="--", label="unweighted")
        for i, al in enumerate(alphas):
            h, _ = np.histogram(y[m], bins=edges, weights=W[(al, b)][m])
            ax.stairs(h / h.sum(), edges, color=cmap(i / max(1, len(alphas) - 1)), lw=1.3, label=f"α={al:g}")
        ax.set_title(name, fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_ylabel("fraction / 0.1 GeV (area-normalized)", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel(f"true {tgt} [GeV]")
    axs[0, 0].legend(fontsize=7.5, ncol=2)
    fig.suptitle(f"Weighted shape vs alpha at bins={b}\n{title_tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"alpha_overlay_{tgt}.png"), dpi=120)
    plt.close(fig)

    # 3) metrics vs alpha, one line per bins
    keys = [("mean_event_weight", "mean event weight ⟨w⟩ (effective LR scale)", False),
            ("ess_frac", "effective sample size / N", False),
            ("w_p99_over_p1", "event weight p99 / p1", True),
            ("share_s3_weighted", "sample-3 share of total weight", False),
            ("peak_over_4to5_s3", "sample 3: density 1.5–2.5 / 4–5 GeV", True),
            ("peak_over_8to10_all", "all: density 1.5–2.5 / 8–10 GeV", True)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    xa = [max(al, 3e-4) for al in alphas]
    for ax, (key, lab, logy) in zip(axs.flat, keys):
        for j, bb in enumerate(binss):
            v = [r[key] for r in rows if r["bins"] == bb]
            ax.plot(xa, v, "o-", ms=3.5, color=cmap(j / max(1, len(binss) - 1)), label=f"bins={bb}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if key == "share_s3_weighted":
            ax.axhline(rows[0]["share_s3_raw"], color="0.5", ls="--", lw=1, label="unweighted")
        if key.startswith("peak_over"):
            ax.axhline(1, color="0.5", ls=":", lw=1)
        ax.axvline(0.01, color="crimson", lw=0.8, alpha=0.5)
        ax.axvline(0.05, color="0.4", lw=0.8, alpha=0.5)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("alpha (0 plotted at 3e-4)", fontsize=8)
        ax.grid(alpha=0.25, which="both")
    axs[0, 0].legend(fontsize=7.5)
    fig.suptitle(f"Weight diagnostics vs alpha (grey line r05 α=0.05, red r06 plan α=0.01)\n{title_tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "metrics.png"), dpi=120)
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
