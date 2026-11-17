"""p1: reconstructed vs true neutrino energy, S3 test events, r05 (counts, log z).

    python -I scripts/p1_response.py --horn fhc|rhc

Black points: median E_reco per rs.PERF_BINS true-energy bin; vertical lines: 16th-84th percentile
of E_reco in that bin (event spread, not an uncertainty). Quantiles use all events in the true bin,
including those whose E_reco falls outside the 0-6 GeV frame.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common as cm  # noqa: E402
from common import rs  # noqa: E402

MODEL, SAMPLE = "r05", 3
H2_BINS = np.round(np.arange(0, 6.0 + 1e-9, 0.1), 2)
ZLIM = (1, 3e3)  # fixed colour range, shared by FHC and RHC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    a = ap.parse_args()
    rs.use()
    d = cm.load(a.horn)
    m = d["sample"] == SAMPLE
    t, p = d["trueE"][m], d["pred"][MODEL][m]
    n_all = int(m.sum())
    in_true = (t >= 0) & (t < 6)
    out_reco = int(np.sum(in_true & ((p < 0) | (p >= 6))))
    h, _, _ = np.histogram2d(t, p, bins=[H2_BINS, H2_BINS])
    if h.max() > ZLIM[1]:
        sys.exit(f"max bin content {h.max()} above fixed z range {ZLIM[1]}")

    e = np.asarray(rs.PERF_BINS)
    centres = 0.5 * (e[1:] + e[:-1])
    ns, med, q16, q84 = [], [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        sel = (t >= lo) & ((t < hi) if hi < e[-1] else (t <= hi))
        ns.append(int(sel.sum()))
        q = np.quantile(p[sel], [0.16, 0.5, 0.84]) if sel.sum() else [np.nan] * 3
        q16.append(q[0]), med.append(q[1]), q84.append(q[2])
    ns, med, q16, q84 = map(np.asarray, (ns, med, q16, q84))
    ok = ns >= rs.MIN_PER_BIN

    fig, ax = plt.subplots(figsize=(6.4 * 1.12, 5.6))
    hm = np.where(h > 0, h, np.nan).T
    im = ax.pcolormesh(H2_BINS, H2_BINS, hm, norm=LogNorm(*ZLIM), cmap="viridis", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Events", loc="top")
    cb.ax.tick_params(direction="in", which="both")
    ax.plot([0, 6], [0, 6], ls="--", color="black", lw=1.2)
    ax.vlines(centres[ok], q16[ok], q84[ok], color="black", lw=1.4)
    ax.plot(centres[ok], med[ok], "o", color="black", ms=4.5)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    rs.labels(ax, rs.E_TRUE, rs.E_RECO)
    rs.tag(ax, f"{rs.HORN[a.horn]} · S3", loc="upper left")
    handles = [Line2D([], [], color="black", marker="o", ls="none", ms=4.5, label="r05 median"),
               Line2D([], [], color="black", lw=1.4, label="16–84% interval"),
               Line2D([], [], color="black", ls="--", lw=1.2, label=r"$E_\nu^{\rm reco} = E_\nu^{\rm true}$")]
    ax.legend(handles=handles, loc="lower right", fontsize=11, frameon=True, fancybox=False, framealpha=0.85,
              facecolor="white", edgecolor="none", borderpad=0.5)
    hi_e = ok & (e[:-1] >= 4.5)
    dev_hi = float(np.max(np.abs(med[hi_e] / centres[hi_e] - 1))) * 100
    n_clip = int(np.sum(ok & (q84 > 6.0)))
    dropped = [f"{e[i]:.2f}-{e[i + 1]:.2f} GeV (N = {ns[i]})" for i in np.flatnonzero(~ok)]
    c = d["cards"][MODEL]
    rs.save(fig, os.path.join(cm.ROOT, a.horn), "p1_s3_response_r05",
            f"2D histogram of reconstructed vs true neutrino energy (raw event counts per bin, log colour scale, "
            f"0.1 x 0.1 GeV bins, 0-6 GeV on both axes); dashed line = identity.\n"
            f"Events: {rs.HORN[a.horn]} test split, sample S3 (peripheral, muon uncontained), N = {n_all:,}, of which "
            f"{int(in_true.sum()):,} have true E in 0-6 GeV; {out_reco:,} of these have E_reco outside 0-6 GeV (not in the frame).\n"
            f"Model: r05 (alpha = 0.05), {c['model_dir']} (result.csv sha256 {d['hashes'][MODEL]['result.csv'][:12]}).\n"
            f"Black points: median E_reco in 0.25 GeV true-energy bins (0.5-6 GeV); vertical lines: 16th-84th "
            f"percentile of E_reco (event spread, not an uncertainty), from all events in the true bin incl. "
            f"those outside the frame. Bins with < {rs.MIN_PER_BIN} events not drawn"
            f"{': ' + ', '.join(dropped) if dropped else ''}.\n"
            f"{n_clip} of the 16-84% lines extend above 6 GeV and are clipped at the frame. Above 4.5 GeV the median "
            f"lies within {dev_hi:.1f}% of the identity line (median / bin centre).\n"
            f"Claim: the r05 S3 median response lies above the identity line below 1.75 GeV and below it "
            f"between 2.25 and 3.5 GeV true energy.")
    cm.manifest(a.horn, "p1_s3_response_r05", d, dict(
        sample=SAMPLE, model=MODEL, n_sample=n_all, n_true_0to6_reco_outside_0to6=out_reco,
        h2_bins_gev=list(H2_BINS), max_bin_content=float(h.max()),
        per_bin={f"{e[i]:.2f}-{e[i + 1]:.2f}": dict(n=int(ns[i]), drawn=bool(ok[i]),
                                                    median=float(med[i]), q16=float(q16[i]), q84=float(q84[i]))
                 for i in range(len(ns))}))
    print("max bin", h.max(), "outside", out_reco, "of", n_all)


if __name__ == "__main__":
    main()
