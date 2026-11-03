"""Preview the proposed per-sample flattening scheme (findings 2026-11-03), before any loader change.

Per sample k, with its own flat range [lo_k, hi_k] (physics ranges, default S1 0.5-5, S2 0.5-6, S3 0.5-8 GeV):
  - inside the range: w = 1 / (count + max(alpha * max_count, 1)) in 0.1 GeV bins (FlatSpectraWeights form),
    rescaled to mean 1 over the in-range events;
  - below lo_k: the first bin's weight;
  - at or above hi_k: weight 1, so the tail keeps its natural share of the sample (neither boosted nor cut).
The sample mix is kept (each sample's weights have mean 1 per event).
Compared against: per-sample flattening over the full range (each sample's min-max, repo class) and pooled
FlatSpectraWeights (alpha 0.01 and 0.05, 500 bins), all rescaled to mean event weight 1.

Usage: python -I pcnumu/eval/flattening_scheme_preview.py --cache <train_<horn>.npz> --tag fhc --out <dir>
The cache holds y = trueE and s = sample of the train rows (written by flattening_scan.py --cache).
"""

import argparse
import csv
import os
import sys

import numpy as np

SAMPLES = {1: "Sample 1 (contained)", 2: "Sample 2 (peripheral, μ contained)",
           3: "Sample 3 (peripheral, μ uncontained)"}
COLORS = {1: "#2a6fdb", 2: "#e8643c", 3: "#1aa37a", 0: "#222222"}


def scheme_weights(y, lo, hi, alpha, width=0.1):
    e = np.arange(lo, hi + 1e-9, width)
    c, _ = np.histogram(y, e)
    wb = 1.0 / (c + max(alpha * c.max(), 1))
    i = np.clip(np.digitize(y, e) - 1, 0, len(wb) - 1)
    w = wb[i]
    tail = y >= hi
    core = ~tail
    w[core] *= core.sum() / w[core].sum()
    w[tail] = 1.0
    return w


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum() / w.size


def flatness(y, w, lo, hi):
    e = np.arange(lo, hi + 1e-9, 0.5)
    h, _ = np.histogram(y, e, weights=w)
    h = h[h > 0]
    return h.max() / h.min() if h.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--ranges", default="1:0.5-5,2:0.5-6,3:0.5-8")
    ap.add_argument("--alphas", default="0.001,0.0025,0.005,0.01,0.02,0.05")
    ap.add_argument("--alpha", type=float, default=0.01, help="alpha used in the combined comparison")
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    d = np.load(a.cache)
    y, s = d["y"], d["s"]
    rng = {int(k): tuple(float(v) for v in r.split("-")) for k, r in (x.split(":") for x in a.ranges.split(","))}
    alphas = [float(x) for x in a.alphas.split(",")]
    os.makedirs(a.out, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    rows = []
    for k in SAMPLES:
        yk = y[s == k]
        lo, hi = rng[k]
        p1 = float(np.percentile(yk, 1))
        for al in alphas:
            w = scheme_weights(yk, lo, hi, al)
            rows.append({"sample": k, "alpha": al, "flat_lo": lo, "flat_hi": hi, "p1_GeV": round(p1, 2),
                         "flatness_phys_range": flatness(yk, w, lo, hi),
                         "flatness_populated": flatness(yk, w, max(lo, np.ceil(p1 * 2) / 2), hi),
                         "ess_frac": ess(w), "w_max_over_median": w.max() / np.median(w),
                         "events_below_p1_maxweight": w[yk < p1].max() if (yk < p1).any() else np.nan,
                         "weight_frac_above_hi": w[yk >= hi].sum() / w.sum(),
                         "weight_frac_above_10": w[yk > 10].sum() / w.sum(),
                         "event_frac_above_10": float(np.mean(yk > 10))})
    with open(os.path.join(a.out, "scheme_metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})

    title = f"{a.tag} · proposed scheme: per-sample flat over the physics range, tail at natural share"

    # 1) per sample vs alpha
    ed_lin, ed_log = np.linspace(0, 10, 101), np.linspace(0, 30, 121)
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for j, k in enumerate(SAMPLES):
        yk = y[s == k]
        lo, hi = rng[k]
        for row, (ed, logy) in enumerate(((ed_lin, False), (ed_log, True))):
            ax = axs[row, j]
            h0, _ = np.histogram(yk, ed)
            ax.stairs(h0 / h0.sum(), ed, color="0.6", ls="--", lw=1, label="unweighted")
            for i, al in enumerate(alphas):
                h, _ = np.histogram(yk, ed, weights=scheme_weights(yk, lo, hi, al))
                ax.stairs(h / h.sum(), ed, color=cmap(i / (len(alphas) - 1)),
                          lw=2.2 if al == a.alpha else 1.1, label=f"α={al:g}")
            ax.axvspan(lo, hi, color="0.85", alpha=0.35, lw=0)
            ax.grid(alpha=0.25)
            ax.set_xlabel("true trueE [GeV]", fontsize=8)
            ax.set_ylabel("fraction / bin (area-normalized)", fontsize=8)
            if logy:
                ax.set_yscale("log")
        axs[0, j].set_title(f"{SAMPLES[k]}  N={yk.size:,}\ngrey band = flat range {lo:g}–{hi:g} GeV; "
                            f"above it weight 1 (natural share)", fontsize=9)
        axs[0, j].legend(fontsize=7, ncol=2)
        ax = axs[2, j]
        rk = [r for r in rows if r["sample"] == k]
        xa = [r["alpha"] for r in rk]
        ax.plot(xa, [r["flatness_phys_range"] for r in rk], "o-", color=COLORS[k], label=f"flatness {lo:g}–{hi:g} GeV")
        ax.plot(xa, [r["flatness_populated"] for r in rk], "o--", color=COLORS[k], alpha=0.5,
                label=f"flatness from p1 ({rk[0]['p1_GeV']:g} GeV)")
        ax.axhline(1.5, color=COLORS[k], ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.plot(xa, [r["ess_frac"] for r in rk], "s--", color="0.3", ms=3, label="ESS / N (this sample)")
        ax2.plot(xa, [r["weight_frac_above_10"] for r in rk], "^:", color="0.55", ms=3, label="weight share > 10 GeV")
        ax2.axhline(rk[0]["event_frac_above_10"], color="0.55", lw=0.8, ls="-", alpha=0.5)
        ax2.set_ylim(0, 1)
        ax.axvline(a.alpha, color="crimson", lw=0.8)
        ax.set_xlabel("alpha", fontsize=8)
        ax.set_title(f"thin grey line = event share > 10 GeV ({rk[0]['event_frac_above_10']:.1%})", fontsize=8.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, which="both")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "scheme_per_sample_alpha.png"), dpi=110)
    plt.close(fig)

    # 2) combined comparison at one alpha
    def mean1(w):
        return w * (w.size / w.sum())
    w_scheme = np.empty_like(y)
    w_full = np.empty_like(y)
    for k in SAMPLES:
        m = s == k
        w_scheme[m] = scheme_weights(y[m], *rng[k], a.alpha)
        wf = FlatSpectraWeights(y[m], bins=500, alpha=a.alpha).getweight(y[m])
        w_full[m] = wf * (m.sum() / wf.sum())
    combos = [(f"proposed: per-sample, flat over physics range, tail natural (α={a.alpha:g})", w_scheme),
              (f"per-sample, flat over full range (α={a.alpha:g} each)", w_full),
              ("pooled α=0.01 (r06 plan)", mean1(FlatSpectraWeights(y, bins=500, alpha=0.01).getweight(y))),
              ("pooled α=0.05 (r05)", mean1(FlatSpectraWeights(y, bins=500, alpha=0.05).getweight(y)))]
    summ = []
    fig, axs = plt.subplots(len(combos), 2, figsize=(14, 3.5 * len(combos)))
    for i, (name, w) in enumerate(combos):
        for col, (ed, logy) in enumerate(((ed_lin, False), (ed_log, True))):
            ax = axs[i, col]
            for k in SAMPLES:
                m = s == k
                h, _ = np.histogram(y[m], ed, weights=w[m])
                ax.stairs(h, ed, color=COLORS[k], lw=1.2, label=SAMPLES[k])
            if logy:
                ax.set_yscale("log")
                ax.axvline(10, color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"weighted events / {ed[1] - ed[0]:.2g} GeV", fontsize=8)
        r = {"scheme": name, "ess_all": ess(w), "weight_frac_above_10_all": w[y > 10].sum() / w.sum()}
        for k in SAMPLES:
            m = s == k
            r[f"s{k}_weight_share"] = w[m].sum() / w.sum()
            r[f"s{k}_weight_frac_above_10"] = w[m & (y > 10)].sum() / w[m].sum()
            r[f"s{k}_ess"] = ess(w[m])
        summ.append(r)
        axs[i, 0].set_title(f"{name}   ESS {r['ess_all']:.0%}; weight > 10 GeV: all {r['weight_frac_above_10_all']:.0%}, "
                            f"S3 {r['s3_weight_frac_above_10']:.0%}; shares S1/S2/S3 "
                            f"{r['s1_weight_share']:.0%}/{r['s2_weight_share']:.0%}/{r['s3_weight_share']:.0%}",
                            fontsize=8.5, loc="left")
    axs[0, 0].legend(fontsize=7.5)
    for ax in axs[-1]:
        ax.set_xlabel("true trueE [GeV]")
    fig.suptitle(f"{a.tag} · weighted training spectrum by sample (each scheme: mean event weight 1); "
                 "left 0–10 linear, right 0–30 log", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "scheme_vs_others.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(a.out, "scheme_vs_others.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0]))
        wr.writeheader()
        for r in summ:
            wr.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    # 3) combined total (all samples summed) per scheme: shape, flatness and ESS of the whole training set
    windows = [(1.0, 5.0), (1.3, 8.0)]
    fig, axs = plt.subplots(len(combos) + 1, 2, figsize=(14, 3.2 * (len(combos) + 1)))
    tot_rows = []
    for i, (name, w) in enumerate(combos):
        r = {"scheme": name, "ess_all": ess(w), "ess_all_events": ess(w) * w.size,
             "weight_frac_above_10_all": w[y > 10].sum() / w.sum()}
        for lo, hi in windows:
            r[f"total_flatness_{lo:g}-{hi:g}"] = flatness(y, w, lo, hi)
        tot_rows.append(r)
        for col, (ed, logy) in enumerate(((ed_lin, False), (ed_log, True))):
            ax = axs[i, col]
            bottom = np.zeros(len(ed) - 1)
            for k in SAMPLES:
                m = s == k
                h, _ = np.histogram(y[m], ed, weights=w[m])
                ax.stairs(bottom + h, ed, baseline=bottom, fill=True, color=COLORS[k], alpha=0.35, lw=0,
                          label=SAMPLES[k] if i == 0 and col == 0 else None)
                bottom = bottom + h
            ax.stairs(bottom, ed, color=COLORS[0], lw=1.6, label="total" if i == 0 and col == 0 else None)
            for lo, hi in windows:
                ax.axvline(lo, color="0.5", ls=":", lw=0.8)
                ax.axvline(hi, color="0.5", ls=":", lw=0.8)
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"weighted events / {ed[1] - ed[0]:.2g} GeV", fontsize=8)
        axs[i, 0].set_title(f"{name}\ncombined: ESS {r['ess_all']:.1%} ({r['ess_all_events']:,.0f} events); "
                            + "; ".join(f"flatness {lo:g}–{hi:g} GeV {r[f'total_flatness_{lo:g}-{hi:g}']:.2f}"
                                        for lo, hi in windows)
                            + f"; weight > 10 GeV {r['weight_frac_above_10_all']:.1%}", fontsize=8.5, loc="left")
    for col, (ed, logy) in enumerate(((ed_lin, False), (ed_log, True))):
        ax = axs[-1, col]
        for i, (name, w) in enumerate(combos):
            h, _ = np.histogram(y, ed, weights=w)
            ax.stairs(h / h.sum(), ed, lw=1.4, color=plt.get_cmap("tab10")(i), label=name)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_ylabel("total, fraction / bin", fontsize=8)
        ax.set_xlabel("true trueE [GeV]")
    axs[-1, 0].set_title("all schemes: combined total, area-normalized", fontsize=9, loc="left")
    axs[-1, 0].legend(fontsize=7)
    axs[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"{a.tag} · combined total of all samples (stacked) per scheme; dotted lines = flatness windows "
                 + ", ".join(f"{lo:g}–{hi:g}" for lo, hi in windows) + " GeV", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "combined_total.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(a.out, "combined_total.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(tot_rows[0]))
        wr.writeheader()
        for r in tot_rows:
            wr.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
