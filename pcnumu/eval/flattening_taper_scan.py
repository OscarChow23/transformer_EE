"""Scan the per-sample flattening scheme over upper edge x taper width (findings 2026-11-03).

Per sample k: flat over [lo_k, hi] (0.1 GeV bins, w_flat = 1/(count + alpha*max count) on the histogram of
[lo_k, hi + taper]), scaled so the events in [lo_k, hi) have mean weight 1; in [hi, hi + taper] the weight
moves from w_flat to 1 on a log scale, w = w_flat**(1 - f), f = (E - hi)/taper; above hi + taper: weight 1
(natural share); below lo_k: the first bin's weight. Each sample is then rescaled to mean event weight 1.
Combined ESS uses the identity ESS_all = N / sum_k(N_k / ESS_k) (each sample has mean weight 1).

Usage: python -I pcnumu/eval/flattening_taper_scan.py --cache <train_<horn>.npz> --tag fhc --out <dir>
       [--lo 1:1.0,2:1.3,3:1.3] [--candidates "5.5/1,7/1,10/2;..."]
"""

import argparse
import csv
import itertools
import os

import numpy as np

SAMPLES = {1: "Sample 1 (contained)", 2: "Sample 2 (peripheral, μ contained)",
           3: "Sample 3 (peripheral, μ uncontained)"}
COLORS = {1: "#2a6fdb", 2: "#e8643c", 3: "#1aa37a", 0: "#222222"}
EDGES = {1: [5.0, 5.5, 6.0], 2: [6.0, 7.0], 3: [8.0, 10.0, 12.0]}
TAPERS = [0.0, 1.0, 2.0, 4.0]


def taper_weights(y, lo, hi, taper, alpha, width=0.1):
    e = np.arange(lo, hi + taper + 1e-9, width)
    if len(e) < 2:
        e = np.array([lo, lo + width])
    c, _ = np.histogram(y, e)
    wb = 1.0 / (c + max(alpha * c.max(), 1))
    i = np.clip(np.digitize(y, e) - 1, 0, len(wb) - 1)
    w = wb[i]
    core = y < hi
    w = w * (core.sum() / w[core].sum())
    if taper > 0:
        tz = (y >= hi) & (y < hi + taper)
        f = (y[tz] - hi) / taper
        w[tz] = w[tz] ** (1 - f)
    w[y >= hi + taper] = 1.0
    return w * (y.size / w.sum()), c


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum() / w.size


def flatness(y, w, lo, hi, step=0.5):
    e = np.arange(lo, hi + 1e-9, step)
    h, _ = np.histogram(y, e, weights=w)
    h = h[h > 0]
    return h.max() / h.min() if h.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lo", default="1:1.0,2:1.3,3:1.3")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--candidates", default="", help="'S1hi/S1taper,S2hi/S2taper,S3hi/S3taper;...' for the total plots")
    a = ap.parse_args()
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    d = np.load(a.cache)
    y, s = d["y"], d["s"]
    lo = {int(k): float(v) for k, v in (x.split(":") for x in a.lo.split(","))}
    os.makedirs(a.out, exist_ok=True)
    N = {k: int((s == k).sum()) for k in SAMPLES}

    # per-sample grid
    rows, W = [], {}
    for k in SAMPLES:
        yk = y[s == k]
        for hi, tp in itertools.product(EDGES[k], TAPERS):
            w, c = taper_weights(yk, lo[k], hi, tp, a.alpha)
            W[(k, hi, tp)] = w
            rows.append({"sample": k, "lo": lo[k], "hi": hi, "taper": tp, "n": yk.size,
                         "min_events_per_bin": int(c.min()),
                         "w_max_over_median": w.max() / np.median(w),
                         "ess_frac": ess(w), "ess_events": ess(w) * yk.size,
                         "weight_frac_above_10": w[yk > 10].sum() / w.sum(),
                         "event_frac_above_10": float(np.mean(yk > 10)),
                         "flatness_window": flatness(yk, w, lo[k], 5.0),
                         "flatness_flat_range": flatness(yk, w, lo[k], hi)})
    with open(os.path.join(a.out, "taper_scan_per_sample.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({kk: (f"{v:.5g}" if isinstance(v, float) else v) for kk, v in r.items()})

    # combined: every edge set, same taper for all samples
    def comb(choice):
        inv = sum(N[k] / next(r for r in rows if r["sample"] == k and r["hi"] == choice[k][0]
                               and r["taper"] == choice[k][1])["ess_frac"] for k in SAMPLES)
        return sum(N.values()) / inv
    crow = []
    for e1, e2, e3 in itertools.product(EDGES[1], EDGES[2], EDGES[3]):
        for tp in TAPERS:
            ch = {1: (e1, tp), 2: (e2, tp), 3: (e3, tp)}
            wall = np.concatenate([W[(k, ch[k][0], tp)] for k in SAMPLES])
            yall = np.concatenate([y[s == k] for k in SAMPLES])
            ce = comb(ch)
            crow.append({"S1_hi": e1, "S2_hi": e2, "S3_hi": e3, "taper": tp, "ess_frac": ce,
                         "ess_events": ce * y.size, "w_max_over_median": wall.max() / np.median(wall),
                         "weight_frac_above_10": wall[yall > 10].sum() / wall.sum(),
                         "total_flatness_1-5": flatness(yall, wall, 1.0, 5.0)})
    with open(os.path.join(a.out, "taper_scan_combined.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(crow[0]))
        wr.writeheader()
        for r in crow:
            wr.writerow({kk: (f"{v:.5g}" if isinstance(v, float) else v) for kk, v in r.items()})

    # trade-off plot: per sample ESS (fraction and events), max weight/median, weight > 10 GeV vs edge; combined
    tcol = {tp: plt.get_cmap("plasma")(i / len(TAPERS)) for i, tp in enumerate(TAPERS)}
    fig, axs = plt.subplots(3, 4, figsize=(18, 11))
    for j, k in enumerate(SAMPLES):
        for tp in TAPERS:
            rk = [r for r in rows if r["sample"] == k and r["taper"] == tp]
            x = [r["hi"] for r in rk]
            lab = "step" if tp == 0 else f"taper {tp:g} GeV"
            axs[0, j].plot(x, [100 * r["ess_frac"] for r in rk], "o-", color=tcol[tp], label=lab)
            axs[1, j].plot(x, [r["w_max_over_median"] for r in rk], "o-", color=tcol[tp], label=lab)
            axs[2, j].plot(x, [100 * r["weight_frac_above_10"] for r in rk], "o-", color=tcol[tp], label=lab)
            for xx, r in zip(x, rk):
                axs[0, j].annotate(f"{r['ess_events'] / 1e3:.0f}k", (xx, 100 * r["ess_frac"]), fontsize=6.5,
                                   xytext=(3, 3), textcoords="offset points", color=tcol[tp])
                axs[1, j].annotate(f"min {r['min_events_per_bin']}", (xx, r["w_max_over_median"]), fontsize=6.5,
                                   xytext=(3, -9), textcoords="offset points", color=tcol[tp])
        axs[2, j].axhline(100 * rows[[r["sample"] for r in rows].index(k)]["event_frac_above_10"], color="0.5",
                          ls="--", lw=1, label="event share > 10 GeV")
        axs[0, j].set_title(f"{SAMPLES[k]} (N={N[k]:,}, flat from {lo[k]:g} GeV)", fontsize=9.5)
        axs[0, j].set_ylabel("ESS % of sample (labels: effective events)", fontsize=8)
        axs[1, j].set_ylabel("max weight / median (labels: fewest events per 0.1 GeV bin)", fontsize=8)
        axs[1, j].set_yscale("log")
        axs[2, j].set_ylabel("% of sample weight above 10 GeV", fontsize=8)
        for i in range(3):
            axs[i, j].set_xlabel("upper edge of the flat range [GeV]", fontsize=8)
            axs[i, j].grid(alpha=0.25)
        axs[0, j].legend(fontsize=7)
        axs[2, j].legend(fontsize=7)
    # combined panels: categorical edge sets
    sets = list(itertools.product(EDGES[1], EDGES[2], EDGES[3]))
    xs = np.arange(len(sets))
    for tp in TAPERS:
        cr = [next(r for r in crow if (r["S1_hi"], r["S2_hi"], r["S3_hi"]) == st and r["taper"] == tp) for st in sets]
        lab = "step" if tp == 0 else f"taper {tp:g} GeV"
        axs[0, 3].plot(xs, [100 * r["ess_frac"] for r in cr], "o-", ms=3, color=tcol[tp], label=lab)
        axs[1, 3].plot(xs, [r["w_max_over_median"] for r in cr], "o-", ms=3, color=tcol[tp], label=lab)
        axs[2, 3].plot(xs, [100 * r["weight_frac_above_10"] for r in cr], "o-", ms=3, color=tcol[tp], label=lab)
    for i in range(3):
        axs[i, 3].set_xticks(xs)
        axs[i, 3].set_xticklabels([f"{a1:g}/{a2:g}/{a3:g}" for a1, a2, a3 in sets], rotation=90, fontsize=6.5)
        axs[i, 3].set_xlabel("upper edges S1/S2/S3 [GeV] (same taper for all)", fontsize=8)
        axs[i, 3].grid(alpha=0.25)
    axs[0, 3].set_title(f"Combined (N={y.size:,}); S1 is 74% of the weight", fontsize=9.5)
    axs[0, 3].set_ylabel("combined ESS %", fontsize=8)
    axs[1, 3].set_ylabel("combined max weight / median", fontsize=8)
    axs[1, 3].set_yscale("log")
    axs[2, 3].set_ylabel("% of all weight above 10 GeV", fontsize=8)
    axs[0, 3].legend(fontsize=7)
    fig.suptitle(f"{a.tag} · per-sample flattening: upper edge × taper width (alpha {a.alpha:g}; "
                 "taper = weights falling from the flat level to 1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "taper_tradeoff.png"), dpi=110)
    plt.close(fig)

    # combined totals for candidates
    if a.candidates:
        import sys  # pylint: disable=C0415
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
        combos = []
        for cand in a.candidates.split(";"):
            parts = [tuple(float(v) for v in p.split("/")) for p in cand.split(",")]
            ch = dict(zip(SAMPLES, parts))
            w = np.empty_like(y)
            for k in SAMPLES:
                m = s == k
                w[m] = taper_weights(y[m], lo[k], ch[k][0], ch[k][1], a.alpha)[0]
            name = "edges/taper " + ", ".join(f"S{k} {ch[k][0]:g}/{ch[k][1]:g}" for k in SAMPLES) + f" GeV (α={a.alpha:g})"
            combos.append((name, w))
        for al in (0.01, 0.05):
            wp = FlatSpectraWeights(y, bins=500, alpha=al).getweight(y)
            combos.append((f"pooled α={al:g}" + (" (r06 plan)" if al == 0.01 else " (r05)"), wp * (y.size / wp.sum())))
        ed_lin, ed_log = np.linspace(0, 12, 121), np.linspace(0, 30, 121)
        fig, axs = plt.subplots(len(combos) + 1, 2, figsize=(14, 3.1 * (len(combos) + 1)))
        tot = []
        for i, (name, w) in enumerate(combos):
            r = {"scheme": name, "ess_all": ess(w), "ess_all_events": ess(w) * y.size,
                 "weight_frac_above_10_all": w[y > 10].sum() / w.sum(), "total_flatness_1-5": flatness(y, w, 1.0, 5.0)}
            for k in SAMPLES:
                m = s == k
                r[f"s{k}_ess"] = ess(w[m])
                r[f"s{k}_ess_events"] = ess(w[m]) * m.sum()
            tot.append(r)
            for col, ed in enumerate((ed_lin, ed_log)):
                ax = axs[i, col]
                bottom = np.zeros(len(ed) - 1)
                for k in SAMPLES:
                    m = s == k
                    h, _ = np.histogram(y[m], ed, weights=w[m])
                    ax.stairs(bottom + h, ed, baseline=bottom, fill=True, color=COLORS[k], alpha=0.35, lw=0,
                              label=SAMPLES[k] if i == 0 and col == 0 else None)
                    bottom = bottom + h
                ax.stairs(bottom, ed, color=COLORS[0], lw=1.5, label="total" if i == 0 and col == 0 else None)
                ax.axvline(5, color="0.5", ls=":", lw=0.8)
                if col == 1:
                    ax.set_yscale("log")
                    ax.axvline(10, color="0.5", ls=":", lw=0.8)
                ax.grid(alpha=0.25)
                ax.set_ylabel(f"weighted events / {ed[1] - ed[0]:.2g} GeV", fontsize=8)
            axs[i, 0].set_title(f"{name}\ncombined ESS {r['ess_all']:.1%} ({r['ess_all_events']:,.0f}); S3 ESS "
                                f"{r['s3_ess']:.1%} ({r['s3_ess_events']:,.0f}); total flatness 1–5 GeV "
                                f"{r['total_flatness_1-5']:.2f}; weight > 10 GeV {r['weight_frac_above_10_all']:.1%}",
                                fontsize=8.5, loc="left")
        for col, ed in enumerate((ed_lin, ed_log)):
            ax = axs[-1, col]
            for i, (name, w) in enumerate(combos):
                h, _ = np.histogram(y, ed, weights=w)
                ax.stairs(h / h.sum(), ed, lw=1.4, color=plt.get_cmap("tab10")(i), label=name)
            if col == 1:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel("true trueE [GeV]")
            ax.set_ylabel("total, fraction / bin", fontsize=8)
        axs[-1, 0].legend(fontsize=6.5)
        axs[0, 0].legend(fontsize=7.5, loc="upper right")
        fig.suptitle(f"{a.tag} · combined total (samples stacked) for candidates vs pooled", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "candidates_combined_total.png"), dpi=110)
        plt.close(fig)
        with open(os.path.join(a.out, "candidates_combined_total.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(tot[0]))
            wr.writeheader()
            for r in tot:
                wr.writerow({kk: (f"{v:.5g}" if isinstance(v, float) else v) for kk, v in r.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
