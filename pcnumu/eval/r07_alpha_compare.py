"""Compare pooled FlatSpectraWeights alpha on the r07 training rows (half of S1) against r05 and r06.

r05 = all train rows, alpha 0.05; r06 = all train rows, alpha 0.01; r07 = train rows with half of S1
(hashed, seed 20261104) at each alpha. bins 500, no range, as r05. Per case: ESS, weight shares,
weight above 10 GeV (pooled and within S3), flatness ratios (density 1.5-2.5 GeV over 4-5 / 8-10 GeV),
mean event weight. Writes compare_<horn>.csv and compare_<horn>.png.

Usage: python -I pcnumu/eval/r07_alpha_compare.py --rows <preview/rows_<horn>.npz> --horn fhc --out <dir>
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r07_s1_subsample_preview import unit_hash, SAMPLES, COLORS  # noqa: E402  pylint: disable=C0413


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--horn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.1,0.05,0.03,0.02,0.01,0.005")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--sub-seed", type=int, default=20261104)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    d = np.load(a.rows)
    y, s, sp, key = d["y"], d["s"], d["sp"], d["key"]
    full = sp == 0
    red = full & ((s != 1) | (unit_hash(key, a.sub_seed) < a.keep))
    alphas = [float(x) for x in a.alphas.split(",")]
    cases = [("r05 (all S1, α 0.05)", full, 0.05), ("r06 (all S1, α 0.01)", full, 0.01)]
    cases += [(f"r07 (S1 {a.keep:.0%}, α {al:g})", red, al) for al in alphas]

    def dens(yy, w, lo, hi):
        sel = (yy >= lo) & (yy < hi)
        return w[sel].sum() / (hi - lo)

    rows, W = [], {}
    for name, m, al in cases:
        yy, ss = y[m], s[m]
        w = FlatSpectraWeights(yy, bins=500, alpha=al).getweight(yy)
        W[name] = (yy, ss, w)
        m3 = ss == 3
        r = {"case": name, "alpha": al, "train_n": int(m.sum()), "mean_w": w.mean(),
             "ess_frac": w.sum() ** 2 / (w ** 2).sum() / w.size,
             "w_gt10": w[yy > 10].sum() / w.sum(),
             "s3_w_gt10_within_s3": w[m3 & (yy > 10)].sum() / w[m3].sum(),
             "all_peak_over_4to5": dens(yy, w, 1.5, 2.5) / dens(yy, w, 4, 5),
             "all_peak_over_8to10": dens(yy, w, 1.5, 2.5) / dens(yy, w, 8, 10),
             "s3_peak_over_4to5": dens(yy[m3], w[m3], 1.5, 2.5) / dens(yy[m3], w[m3], 4, 5),
             "s1_peak_over_4to5": dens(yy[ss == 1], w[ss == 1], 1.5, 2.5) / dens(yy[ss == 1], w[ss == 1], 4, 5)}
        for k in SAMPLES:
            r[f"s{k}_w_share"] = w[ss == k].sum() / w.sum()
        r["s3_ess"] = w[m3].sum() ** 2 / (w[m3] ** 2).sum()
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"compare_{a.horn}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    # before (unweighted) and after (weighted) spectra: r05 reference, then r07 at each alpha
    show = [c[0] for c in cases if c[0].startswith("r05") or c[0].startswith("r06")] + \
           [c[0] for c in cases if c[0].startswith("r07")]
    unw = {"r05/r06 rows, unweighted": (y[full], s[full], np.ones(full.sum())),
           "r07 rows, unweighted": (y[red], s[red], np.ones(red.sum()))}
    panels = list(unw.items()) + [(n, W[n]) for n in show]
    edges_lin, edges_log = np.arange(0, 10.0001, 0.1), np.arange(0, 30.0001, 0.25)
    nc = 2
    fig, axs = plt.subplots(len(panels), nc, figsize=(13, 2.6 * len(panels)))
    for i, (name, (yy, ss, w)) in enumerate(panels):
        for j, (ed, logy) in enumerate(((edges_lin, False), (edges_log, True))):
            ax = axs[i, j]
            for k in (1, 2, 3, 0):
                mk = np.ones_like(ss, bool) if k == 0 else ss == k
                h, _ = np.histogram(yy[mk], ed, weights=w[mk])
                ax.stairs(h / w.sum(), ed, color=COLORS[k], lw=1.4 if k == 0 else 1.1,
                          label=(SAMPLES.get(k, "All") if i == 0 and j == 0 else None))
            if logy:
                ax.set_yscale("log")
                ax.axvline(10, color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
        r = next((r for r in rows if r["case"] == name), None)
        txt = name if r is None else (f"{name}:  ESS {r['ess_frac']:.0%} · S1/S2/S3 {r['s1_w_share']:.0%}/"
                                      f"{r['s2_w_share']:.0%}/{r['s3_w_share']:.0%} · >10 GeV {r['w_gt10']:.0%} "
                                      f"(within S3 {r['s3_w_gt10_within_s3']:.0%}) · peak/4–5 all {r['all_peak_over_4to5']:.2f}, "
                                      f"S3 {r['s3_peak_over_4to5']:.2f}")
        axs[i, 0].set_title(txt, fontsize=8.5, loc="left")
    axs[0, 0].legend(fontsize=7.5)
    for ax in axs[-1]:
        ax.set_xlabel("trueE [GeV]")
    fig.suptitle(f"{a.horn.upper()}: share of train weight per 0.1 GeV (left, 0–10) and 0.25 GeV (right, log 0–30); "
                 f"pooled FlatSpectraWeights, bins 500", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(os.path.join(a.out, f"compare_{a.horn}.png"), dpi=100)
    plt.close(fig)
    for r in rows:
        print(f"{a.horn} {r['case']:24} ESS {r['ess_frac']:.1%}  S1/S2/S3 {r['s1_w_share']:.0%}/{r['s2_w_share']:.0%}/"
              f"{r['s3_w_share']:.0%}  >10 {r['w_gt10']:.1%}  S3>10|S3 {r['s3_w_gt10_within_s3']:.3f}  "
              f"pk/4-5 all {r['all_peak_over_4to5']:.2f} S1 {r['s1_peak_over_4to5']:.2f} S3 {r['s3_peak_over_4to5']:.2f}  "
              f"pk/8-10 all {r['all_peak_over_8to10']:.2f}  S3 effN {r['s3_ess']:.0f}  <w> {r['mean_w']:.3f}")


if __name__ == "__main__":
    main()
