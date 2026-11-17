"""p5 (bias vs true E), p6 (resolution vs true E) and, for S3, p56 (stacked) for r05/r06/r07.

    python -I scripts/perf_vs_trueE.py --horn fhc|rhc [--sample 1|2|3]

r = pred/true - 1 on trueE; bias = median r; resolution = (q84 - q16)/2; rs.PERF_BINS on true E
(bins [lo, hi), last bin closed at 6.0 GeV). Errors: 68% central interval of rs.N_BOOT bootstrap
resamples (seed rs.BOOT_SEED), the same resampled event indices for all three models.
"""
import argparse
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import common as cm
from common import rs

YLIM = {  # from all drawn points incl. 68% intervals, both horns (manifest per_bin); shared FHC/RHC
    "bias": {1: (-16, 6), 2: (-16, 8), 3: (-8, 24)},  # data S1 -13.3..+1.1, S2 -12.3..+3.0, S3 -5.0..+15.9 %
    "res": {1: (0, 10.4), 2: (0, 31), 3: (0, 38)},  # data S1 4.6-7.9, S2 7.6-26.8, S3 13.4-33.2 %
    "p56_bias": (-8, 24), "p56_res": (0, 38),
}
LEGEND = {"bias": {1: "lower left", 2: "lower left", 3: "upper right"},
          "res": {1: "lower right", 2: "lower left", 3: "lower right"}}
Y_BIAS, Y_RES = "Median fractional residual (%)", "Resolution (%)"
DEFS = ("r = E_reco/E_true - 1 per event; median fractional residual = median of r; resolution = (q84 - q16)/2 "
        "of r (16th/84th percentiles); in %. Bins: 0.25 GeV in true neutrino energy over 0.5-6 GeV (last bin "
        "closed at 6 GeV), points at bin centres.")
OFFSET = {"r05": -0.03, "r06": 0.0, "r07": 0.03}


def bin_index(t):
    e = np.asarray(rs.PERF_BINS)
    i = np.searchsorted(e, t, side="right") - 1
    i[t == e[-1]] = len(e) - 2
    i[(t < e[0]) | (t > e[-1])] = -1
    return i


def compute(d, sample):
    t = d["trueE"]
    m_s = d["sample"] == sample
    ib = bin_index(t)
    nb = len(rs.PERF_BINS) - 1
    out = {r: dict(bias=np.full(nb, np.nan), res=np.full(nb, np.nan), bias_lo=np.full(nb, np.nan),
                   bias_hi=np.full(nb, np.nan), res_lo=np.full(nb, np.nan), res_hi=np.full(nb, np.nan))
           for r in cm.SCORECARDS}
    ns = np.zeros(nb, int)
    for b in range(nb):
        sel = np.flatnonzero(m_s & (ib == b))
        ns[b] = sel.size
        if sel.size < rs.MIN_PER_BIN:
            continue
        rng = np.random.default_rng([rs.BOOT_SEED, sample, b])
        idx = rng.integers(0, sel.size, size=(rs.N_BOOT, sel.size), dtype=np.int32) \
            if sel.size * rs.N_BOOT < 4e8 else None
        for r in cm.SCORECARDS:
            x = d["pred"][r][sel] / t[sel] - 1.0
            q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84])
            out[r]["bias"][b], out[r]["res"][b] = q50, (q84 - q16) / 2
            bq = np.empty((rs.N_BOOT, 3))
            if idx is not None:
                for c in range(0, rs.N_BOOT, 100):
                    bq[c:c + 100] = np.quantile(x[idx[c:c + 100]], [0.16, 0.5, 0.84], axis=1).T
            else:  # large bins: regenerate indices chunk-wise from a per-chunk seed (same for all models)
                for c in range(0, rs.N_BOOT, 50):
                    ii = np.random.default_rng([rs.BOOT_SEED, sample, b, c]).integers(0, sel.size, (50, sel.size))
                    bq[c:c + 50] = np.quantile(x[ii], [0.16, 0.5, 0.84], axis=1).T
            bb, br = bq[:, 1], (bq[:, 2] - bq[:, 0]) / 2
            out[r]["bias_lo"][b], out[r]["bias_hi"][b] = np.quantile(bb, [0.16, 0.84])
            out[r]["res_lo"][b], out[r]["res_hi"][b] = np.quantile(br, [0.16, 0.84])
    return out, ns


def draw(ax, st, key, ok, centres):
    for r in cm.SCORECARDS:
        v = 100 * st[r][key][ok]
        err = np.vstack([v - 100 * st[r][key + "_lo"][ok], 100 * st[r][key + "_hi"][ok] - v])
        rs.model_points(ax, centres[ok], v, err, r, offset=OFFSET[r], ms=5)
    ax.set_xlim(*rs.WINDOW)


def check_inside(st, key, ok, lim):
    for r in cm.SCORECARDS:
        lo, hi = 100 * st[r][key + "_lo"][ok], 100 * st[r][key + "_hi"][ok]
        if lo.min() <= lim[0] or hi.max() >= lim[1]:
            raise SystemExit(f"{r} {key}: [{lo.min():.1f}, {hi.max():.1f}] outside y range {lim}")


def legend(ax, **kw):
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, lab, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    ap.add_argument("--sample", type=int, default=3, choices=[1, 2, 3])
    a = ap.parse_args()
    rs.use()
    d = cm.load(a.horn)
    st, ns = compute(d, a.sample)
    e = np.asarray(rs.PERF_BINS)
    centres = 0.5 * (e[1:] + e[:-1])
    ok = ns >= rs.MIN_PER_BIN
    s = a.sample
    H = rs.HORN[a.horn]
    tagtxt = f"{H} · S{s}\n{cm.TAG_SAMPLE[s].split('(')[1].rstrip(')')}"
    out_dir = os.path.join(cm.ROOT, a.horn)
    n_win = int(ns.sum())
    dropped = [f"{e[i]:.2f}-{e[i + 1]:.2f} GeV (N = {ns[i]})" for i in np.flatnonzero(~ok)]
    common_cap = (
        f"Events: {H} test split, sample S{s}, true E 0.5-6 GeV, N = {n_win:,} "
        f"({ok.sum()} of {len(ok)} bins drawn; bins with < {rs.MIN_PER_BIN} events not drawn"
        f"{': ' + ', '.join(dropped) if dropped else ''}); same events for all models.\n"
        f"Models: {cm.model_lines(d)}.\n"
        f"Error bars: 68% central interval of {rs.N_BOOT} bootstrap resamples of the test events (seed "
        f"{rs.BOOT_SEED}), same resampled events for all three models; statistical only, no training-seed "
        f"variation. x offsets of ±0.03 GeV between models are cosmetic.\n")

    R = list(cm.SCORECARDS)
    mid = ok & (e[:-1] >= 1.0) & (e[1:] <= 5.0)

    def spread(key):
        v = np.array([100 * st[r][key][mid] for r in R])
        sp = v.max(0) - v.min(0)
        return sp.max(), np.median(sp)

    def at(key, i):
        return [100 * st[r][key][i] for r in R]

    sb, mb = spread("bias")
    sr, mr = spread("res")
    if s == 3:
        low = ok & (e[1:] <= 1.5)
        closer = all(abs(st["r07"]["bias"][i]) < min(abs(st["r05"]["bias"][i]), abs(st["r06"]["bias"][i]))
                     for i in np.flatnonzero(low))
        if not closer:
            raise SystemExit("S3 low-energy bias statement no longer true")
        lo_r = {r: 100 * st[r]["bias"][low] for r in R}
        desc_bias = (f"Below 1.5 GeV the r07 median residual ({lo_r['r07'].min():+.1f} to {lo_r['r07'].max():+.1f}%) "
                     f"is closer to zero than r05 ({lo_r['r05'].min():+.1f} to {lo_r['r05'].max():+.1f}%) and r06 "
                     f"({lo_r['r06'].min():+.1f} to {lo_r['r06'].max():+.1f}%) in every drawn bin.")
        n05 = int(sum(min(R, key=lambda r: st[r]["res"][i]) == "r05" for i in np.flatnonzero(mid)))
        desc_res = (f"Between 1 and 5 GeV r05 has the smallest resolution in {n05} of {int(mid.sum())} bins; "
                    f"model differences reach {sr:.1f} percentage points (median over bins {mr:.1f}).")
    else:
        i5, i6 = np.flatnonzero(ok & (e[:-1] >= 5.0))[[0, -1]]
        a5, a6 = at("bias", i5), at("bias", i6)
        desc_bias = (f"Between 1 and 5 GeV the three curves differ by {mb:.1f} percentage points in the median bin and "
                     f"by at most {sb:.1f}. Above 5 GeV all three fall steeply: from {min(a5):+.1f} to {max(a5):+.1f}% "
                     f"at {e[i5]:.2f}-{e[i5 + 1]:.2f} GeV to {min(a6):+.1f} to {max(a6):+.1f}% at "
                     f"{e[i6]:.2f}-{e[i6 + 1]:.2f} GeV.")
        desc_res = (f"The three curves have similar shape; between 1 and 5 GeV they differ by {mr:.1f} percentage "
                    f"points in the median bin and by at most {sr:.1f}. No ranking between models is claimed "
                    f"beyond what the error bars show.")

    # ---- p5: bias ----
    fig, ax = rs.figure(size=(6.4, 4.8))
    ax.axhline(0, color="black", ls=":", lw=1.0)
    draw(ax, st, "bias", ok, centres)
    ax.set_ylim(*YLIM["bias"][s])
    check_inside(st, "bias", ok, YLIM["bias"][s])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    rs.labels(ax, rs.E_TRUE, Y_BIAS)
    rs.tag(ax, tagtxt)
    legend(ax, loc=LEGEND["bias"][s])
    name5 = f"p5_s{s}_bias_vs_trueE"
    rs.save(fig, out_dir, name5,
            f"Median fractional residual vs true neutrino energy, S{s}; dotted line = zero. {DEFS}\n"
            + common_cap + desc_bias)

    # ---- p6: resolution ----
    fig, ax = rs.figure(size=(6.4, 4.8))
    draw(ax, st, "res", ok, centres)
    ax.set_ylim(*YLIM["res"][s])
    check_inside(st, "res", ok, YLIM["res"][s])
    rs.labels(ax, rs.E_TRUE, Y_RES)
    rs.tag(ax, tagtxt)
    legend(ax, loc=LEGEND["res"][s])
    name6 = f"p6_s{s}_resolution_vs_trueE"
    rs.save(fig, out_dir, name6,
            f"Resolution vs true neutrino energy, S{s}. {DEFS}\n" + common_cap + desc_res)

    names = [name5, name6]
    # ---- p56: stacked, S3 only ----
    if s == 3:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(hspace=0))
        a1.axhline(0, color="black", ls=":", lw=1.0)
        draw(a1, st, "bias", ok, centres)
        draw(a2, st, "res", ok, centres)
        a1.set_ylim(*YLIM["p56_bias"])
        a2.set_ylim(*YLIM["p56_res"])
        check_inside(st, "bias", ok, YLIM["p56_bias"])
        check_inside(st, "res", ok, YLIM["p56_res"])
        rs.labels(a1, None, Y_BIAS.replace("fractional ", "fractional\n"))  # two lines: fits the half-height pad
        rs.labels(a2, rs.E_TRUE, Y_RES)
        a1.tick_params(labelbottom=False)
        rs.tag(a1, tagtxt)
        legend(a2, loc="lower right")
        fig.align_ylabels([a1, a2])
        pp = {r: d["cards"][r]["_peak"]["by_sample"]["3"]["peak_pull"] for r in R}
        rs.save(fig, out_dir, "p56_s3_bias_resolution",
                f"Top: median fractional residual vs true neutrino energy, S3 (dotted line = zero). Bottom: "
                f"resolution vs true neutrino energy, S3. {DEFS}\n" + common_cap + desc_bias + " " + desc_res + "\n"
                f"The contrast between the median residual at 1.0-1.5 GeV and at 3.0-4.0 GeV (peak pull, from "
                f"peak_metrics.json) is smaller for r06 and r07 than for r05: {pp['r05']:.1f} -> {pp['r06']:.1f} -> "
                f"{pp['r07']:.1f} percentage points ({H}, r05 -> r06 -> r07).")
        names.append("p56_s3_bias_resolution")

    per_bin = {f"{e[i]:.2f}-{e[i + 1]:.2f}": dict(
        n=int(ns[i]), drawn=bool(ok[i]),
        **{r: {k + "_pct": round(100 * st[r][k][i], 4) for k in
               ("bias", "bias_lo", "bias_hi", "res", "res_lo", "res_hi")}
           for r in cm.SCORECARDS if ok[i]}) for i in range(len(ns))}
    for nm in names:
        cm.manifest(a.horn, nm, d, dict(sample=s, n_window=n_win, bins_gev=list(rs.PERF_BINS),
                                        bootstrap=dict(n=rs.N_BOOT, seed=rs.BOOT_SEED, joint_across_models=True,
                                                       interval="68% central"),
                                        per_bin=per_bin, sanity_vs_scorecard=d["sanity"]))
    for i in range(len(ns)):
        if ok[i]:
            print(f"{centres[i]:.3f} n={ns[i]:7d} " + "  ".join(
                f"{r}: b {100 * st[r]['bias'][i]:+6.2f} r {100 * st[r]['res'][i]:5.2f} "
                f"(±{50 * (st[r]['res_hi'][i] - st[r]['res_lo'][i]):.2f})" for r in cm.SCORECARDS))


if __name__ == "__main__":
    main()
