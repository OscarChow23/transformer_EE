"""Figures for the peak-pull metric study. Reads the caches and the json/npz written by the other scripts.

    python -I figures.py          # writes figures/*.png
"""
import itertools
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

GREY, INK, SURF = "#8a8a85", "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7",
                     "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 4, "text.color": INK, "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
SNAME = {"1": "S1 contained", "2": "S2 peripheral, muon contained", "3": "S3 peripheral, muon uncontained"}
X = M.CENTRES


def bins(t, v, fn):
    return np.array([fn(v[(t >= a) & (t < b)]) if ((t >= a) & (t < b)).sum() >= 20 else np.nan
                     for a, b in zip(M.EDGES[:-1], M.EDGES[1:])])


def shade(ax):
    for w in (M.LOW, M.MODE, M.HIGH):
        ax.axvspan(*w, color="#f0efec", lw=0, zorder=0)
    ax.axhline(0, color=GREY, lw=0.8)


def curves(horn):
    d = C.load(horn)
    fig, axs = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True)
    for i, s in enumerate(C.SAMPLES):
        m = d["smp"] == s
        t = d["t"][m]
        sig = C.SIGMA[horn][s]

        def three(run):
            p = d[f"p_{run}"][m]
            x = M.rho(t, p)
            c = np.median(x[(t >= M.MODE[0]) & (t < M.MODE[1])])
            sgn = np.where(t < 2.0, 1.0, -1.0)
            y = sgn * (x - c)
            return (100 * bins(t, x, np.mean), 100 * bins(t, p / t - 1, np.median),
                    100 * bins(t, y, lambda v: (v > sig).mean() - (v < -sig).mean()))

        r07 = np.array([three(r) for r in C.R07])
        for j in range(3):
            ax = axs[i, j]
            shade(ax)
            ax.fill_between(X, r07[:, j].min(0), r07[:, j].max(0), color=GREY, alpha=0.3, lw=0,
                            label="r07 ×4 identical retrains (min–max)")
        for run in C.MAIN:
            y = three(run)
            for j in range(3):
                axs[i, j].plot(X, y[j], "o-", color=C.COL[run], label=run)
        axs[i, 0].set_ylabel(f"{SNAME[s]}\n\npoints (%)")
        if i == 0:
            axs[0, 0].set_title("mean ln(pred/true) per bin  (sees the tail)")
            axs[0, 1].set_title("median pred/true − 1 per bin  (v1 bias curve)")
            axs[0, 2].set_title("net fraction of events beyond 1σ towards 2 GeV, after removing\nthe run's level at the peak (σ = sample's r05 resolution)")
    for ax in axs[-1]:
        ax.set_xlabel("true neutrino energy (GeV)")
    axs[1, 1].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle(f"{horn.upper()}: per-bin curves on the frozen test set. Bands: low 1.0–1.5, peak 1.75–2.25, high 3.0–4.0 GeV", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, f"curves_{horn}.png"), dpi=115)
    plt.close(fig)


def prior_prediction(horn):
    d = C.load(horn)
    s = "3"
    m = d["smp"] == s
    t = d["t"][m]
    sig = C.SIGMA[horn][s]
    pk = (X > M.MODE[0]) & (X < M.MODE[1])
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.9), sharey=True)
    for ax, run in zip(axs, ["r05", "r06", "r07", "r08", "r09"]):
        shade(ax)
        B = C.predicted_pull(horn, run, s, sig)
        b = 100 * bins(t, B(t), np.mean)
        for r in (C.R07 if run == "r07" else [run]):
            y = 100 * bins(t, M.rho(t, d[f"p_{r}"][m]), np.mean)
            ax.plot(X, y - np.nanmean(y[pk]), "o-", color=C.COL[run], lw=1.2 if run == "r07" else 2,
                    label="measured" if r in (run,) else None)
        ax.plot(X, b - b[pk].mean(), "--", color=INK, lw=1.6, label="predicted from the run's\nweighted S3 training spectrum (toy)")
        ax.set_title(f"{run}" + (" (4 identical retrains)" if run == "r07" else ""))
        ax.set_xlabel("true neutrino energy (GeV)")
    axs[0].set_ylabel("mean ln(pred/true) minus its value\nat the peak (points)")
    axs[0].legend(fontsize=7.5)
    fig.suptitle(f"{horn.upper()} S3: bias curve relative to the run's own level at the peak, against what the training spectrum predicts", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, f"prior_prediction_{horn}.png"), dpi=115)
    plt.close(fig)


def uncertainty(horn):
    e = json.load(open(os.path.join(C.HERE, "ensemble_results.json")))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (wn, sign, lab) in zip(axs, (("low", 1, "true 1.0–1.5 GeV"), ("mode", 1, "true 1.75–2.25 GeV (peak)"), ("high", -1, "true 3.0–4.0 GeV"))):
        r = e[f"{horn}_S3_{wn}"]
        x = r["spread_median_pct"]
        ax.axhline(0, color=GREY, lw=0.8)
        y7 = np.array([sign * q["mean_log"] for q in r["r07_loo"]])
        s7 = np.array([q["mean_log"] for q in r["r07_loo_sd"]])
        ax.fill_between(x, y7 - s7, y7 + s7, color=GREY, alpha=0.3, lw=0)
        ax.plot(x, y7, "o-", color=GREY, label="r07 (mean of 4, band ± sd)")
        for run in C.MAIN:
            ax.plot(x, [sign * q["mean_log"] for q in r[run]], "o-", color=C.COL[run], label=run)
        ax.set_title(lab)
        ax.set_xlabel("event uncertainty score: spread of 4 identical r07 trainings (%),\nmedian of each fifth of the events")
    axs[0].set_ylabel("mean ln(pred/true) (points)\npositive = overestimated")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"{horn.upper()} S3: estimate error against event uncertainty. A prior moves the uncertain events (right) and leaves the certain ones (left)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, f"uncertainty_{horn}.png"), dpi=115)
    plt.close(fig)


def pairs(horn):
    c = np.load(os.path.join(C.HERE, "pairs_curves.npz"))
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
    null = np.array([c[f"{horn}_S3_{x}-{y}"] for x, y in itertools.combinations(C.R07, 2)])
    null = null - np.nanmean(null, axis=1, keepdims=True)
    env = np.nanmax(np.abs(null), axis=0)
    for ax, (a, b) in zip(axs, (("r05", "r06"), ("r05", "r08"), ("r05", "r09"), ("r08", "r09"))):
        y, toy = c[f"{horn}_S3_{a}-{b}"], c[f"{horn}_S3_{a}-{b}_toy"]
        ax.axhline(0, color=GREY, lw=0.8)
        ax.fill_between(X, -env, env, color=GREY, alpha=0.3, lw=0, label="largest shift between two identical\nr07 retrains (6 pairs, offsets removed)")
        ax.plot(X, y - np.nanmean(y), "o-", color=C.COL[a], label="measured")
        ax.plot(X, toy - np.nanmean(toy), "--", color=INK, lw=1.6, label="predicted from the two\ntraining spectra (toy)")
        ax.set_title(f"{a} minus {b}")
        ax.set_xlabel("true neutrino energy (GeV)")
    axs[0].set_ylabel("mean ln(pred_A/pred_B) per bin,\noverall offset removed (points)")
    axs[0].legend(fontsize=7.5)
    fig.suptitle(f"{horn.upper()} S3: how far the same test events move when the training spectrum changes", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, f"pairs_{horn}.png"), dpi=115)
    plt.close(fig)


def injection():
    r = json.load(open(os.path.join(C.HERE, "injection_results.json")))
    mets = [("v1_peak_pull", "v1 peak pull"), ("nf_r_sig_low", "proposal as written: net fraction, low side"),
            ("nfc_sig_low", "net fraction after removing peak level, low side"),
            ("mean_log_low_ref", "low vs peak (mean ln ratio)"), ("median_low_ref", "low vs peak (median)")]
    tests = [("a", "no pull"), ("a2", "no pull,\nreal error shape"), ("bL1.0", "r05-size\npull"), ("c", "pull in 30%\nof events"),
             ("d1", "tilt"), ("d2", "leakage"), ("e", "scale\n×1.03"), ("f", "resolution\n×1.5"), ("g", "pull + high-side\noverestimates")]
    fig, axs = plt.subplots(2, len(mets), figsize=(18, 6.6))
    for i, cfg in enumerate(("fhc_S3", "fhc_S1")):
        for j, (k, name) in enumerate(mets):
            ax = axs[i, j]
            v = [r[cfg][t]["value"][k] for t, _ in tests]
            se = [r[cfg][t]["se"][k] for t, _ in tests]
            col = [GREY, GREY, "#2a78d6", "#2a78d6", "#eb6834", "#eb6834", "#eb6834", "#eb6834", "#2a78d6"]
            ax.bar(range(len(tests)), v, yerr=se, color=col, width=0.62, error_kw=dict(lw=1, ecolor=INK))
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_xticks(range(len(tests)))
            ax.set_xticklabels([n for _, n in tests], fontsize=6.3, rotation=90)
            ax.set_title(name, fontsize=8.5)
            ax.grid(axis="x", visible=False)
        axs[i, 0].set_ylabel(f"{cfg.replace('_', ' ').upper()}\nreading (points)")
    fig.suptitle("Injection tests: grey = should read 0; blue = pull injected; orange = no pull injected, so any difference from the first grey bar is a false reading. Error bars: bootstrap at the real sample size", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, "injection.png"), dpi=115)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(C.FIGS, exist_ok=True)
    for h in C.HORNS:
        curves(h)
        prior_prediction(h)
        uncertainty(h)
        pairs(h)
    injection()
    print("figures written to", C.FIGS)
