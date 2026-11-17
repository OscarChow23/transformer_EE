#!/usr/bin/env python
"""Per-sample diagnostics of a finished TransformerEE run that used a `split_column` file.

    python -I pcnumu/eval/eval_by_sample.py --config RUN_CONFIG.json [--model-dir DIR] [--out-dir DIR]
    python -I pcnumu/eval/eval_by_sample.py --config CONFIG --flattening-only OUT_DIR   # before training
    python -I pcnumu/eval/eval_by_sample.py --config CONFIG --conditional-maps-only --out-dir DIR

Writes to OUT_DIR (default <model_dir>/eval_by_sample/):
  summary.csv / summary.md      bias, resolution, tail fractions per sample and combined: all E, the
                                0.5-6 GeV headline window, and trueE 6-10 / 10-20 / >20 GeV
  relerr_<target>.png           relative-error distributions, samples 1/2/3 + all
  relerr_<target>_above6.png    the same for trueE > 6 GeV
  vs_true_<target>.png          bias and resolution vs true value
  vs_true_<target>_wide.png     the same to 50 GeV, variable bins, log x, 6 GeV marked
  summary_<target>.png          bias and resolution per sample, 0.5-6 GeV trueE window
  flattening_<target0>.png      training-row spectrum before/after the run's flattening weights
Metric definitions: pcnumu/core/metrics.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib.pyplot as plt  # noqa: E402

from pcnumu.core import data, plots, report, style  # noqa: E402
from pcnumu.core.metrics import (MIN_PER_BIN, SAMPLES, WINDOW, in_headline_window, relative_error,  # noqa: E402
                                 sample_mask, stats)

WIN_LABEL = f"trueE {WINDOW[0]}-{WINDOW[1]} GeV"
# Supplementary trueE ranges above the headline window: table rows and the *_wide / *_above6 plots.
HIGH_E = [(6.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
WIDE_EDGES = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12.5, 15, 20, 30, 50])


def plot_flattening(cfg, code_dir, out):
    """True-energy spectrum of the training rows before and after the run's flattening weights,
    with the weighter rebuilt exactly as training does (transformer_ee create_weighter)."""
    sys.path.insert(0, code_dir)
    from transformer_ee.utils.weights import create_weighter  # pylint: disable=C0415

    t0 = cfg["target"][0]
    train = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
             .filter(pl.col(cfg["split_column"]) == "train")
             .select(["sample", t0])
             .collect())
    y = train[t0].cast(pl.Float64).to_numpy()
    smp = train["sample"].to_numpy().astype(str)
    wt = create_weighter(cfg, y, sample=smp)
    w = wt.getweight(y, smp) if getattr(wt, "needs_sample", False) else wt.getweight(y)
    edges = np.linspace(0, 10, 101)
    fig, (ax1, ax2) = style.figure(nrows=2, size=(6.4, 3.4), sharex=True, gridspec_kw=dict(hspace=0))
    for k, _ in SAMPLES:
        m = sample_mask(smp, k)
        st = style.SAMPLE_STYLE[k]
        kw = dict(histtype="step", lw=1.6, color=st["color"], ls=st["ls"], label=st["label"])
        ax1.hist(y[m], bins=edges, **kw)
        ax2.hist(y[m], bins=edges, weights=w[m], **kw)
    for ax in (ax1, ax2):  # headroom so the in-frame tags and legend sit above the curves
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    style.labels(ax1, None, "Training events / 0.1 GeV")
    style.labels(ax2, f"True {t0} (GeV), shown 0–10", "Weighted events / 0.1 GeV")
    shown = {k: v for k, v in cfg.get("weight", {}).get("kwargs", {}).items() if k != "per_sample"}
    style.tag(ax1, "Before flattening")
    style.tag(ax2, f"After {cfg.get('weight', {}).get('name', 'no weights')}{' ' + str(shown) if shown else ''}\n"
                   f"mean event weight {w.mean():.2f}")
    ax1.legend(loc="upper right", fontsize=9)
    style.save(fig, out, f"flattening_{t0}", formats=("png",))
    print(f"wrote {os.path.join(out, f'flattening_{t0}.png')}")


def conditional_maps(true, pred, samples, out):
    """Unweighted held-out maps; denominator includes vertical overflow in each slice."""
    xedges = np.linspace(0, 10, 41)
    specs = [("pred_vs_true_trueE", pred, np.linspace(0, 10, 81), "Predicted neutrino energy (GeV)"),
             ("relerr_vs_true_trueE", (pred - true) / true, np.linspace(-1, 1, 101), "(pred − true) / true")]
    records = []
    for sample in ("3", "1", "2"):
        mask = samples == sample
        for name, y, yedges, ylabel in specs:
            x, v = true[mask], y[mask]
            valid = np.isfinite(x) & np.isfinite(v) & (x > 0)
            x, v = x[valid], v[valid]
            # np.histogram includes the final edge; quantile assignment must do the same.
            idx = np.searchsorted(xedges, x, side="right") - 1
            idx[x == xedges[-1]] = len(xedges) - 2
            h, _, _ = np.histogram2d(x, v, bins=(xedges, yedges))
            centers = (xedges[1:] + xedges[:-1]) / 2
            quantiles = np.full((len(centers), 3), np.nan)
            counts = np.zeros(len(centers), dtype=int)
            for i in range(len(centers)):
                vi = v[idx == i]
                counts[i] = vi.size
                if vi.size >= MIN_PER_BIN:
                    quantiles[i] = np.quantile(vi, [0.16, 0.5, 0.84])
                records.append(dict(sample=sample, plot=name, true_lo=xedges[i], true_hi=xedges[i + 1],
                                    n=int(vi.size), visible=int(h[i].sum()), underflow=int((vi < yedges[0]).sum()),
                                    overflow=int((vi > yedges[-1]).sum()),
                                    q16=quantiles[i, 0], median=quantiles[i, 1], q84=quantiles[i, 2]))
            probability = np.divide(h, counts[:, None], out=np.zeros_like(h), where=counts[:, None] > 0)
            assert np.all(h.sum(axis=1) <= counts)
            fig, ax = style.figure(size=(7.6, 5.2))
            image = ax.pcolormesh(xedges, yedges, np.ma.masked_where(probability.T == 0, probability.T),
                                  vmin=0, shading="flat")
            fig.colorbar(image, ax=ax, label="Fraction of the true-energy slice")
            ax.plot(centers, quantiles[:, 1], color=style.BARS[1], lw=1.8, label="Median")
            for j in (0, 2):
                ax.plot(centers, quantiles[:, j], color=style.BARS[1], ls="--", lw=1.4,
                        label="16th / 84th percentile" if j == 0 else None)
            if name.startswith("pred_"):
                ax.plot([0, 10], [0, 10], color="black", lw=1, ls=":", label="pred = true")
            else:
                ax.axhline(0, color="black", lw=1, ls=":", label="Zero error")
            ax.set(xlim=(0, 10), ylim=(yedges[0], yedges[-1]))
            style.labels(ax, "True neutrino energy (GeV)", ylabel)
            excluded = int(mask.sum()) - len(x)
            outside = int(((x < 0) | (x > 10)).sum())
            vertical = int(((x >= 0) & (x <= 10) & ((v < yedges[0]) | (v > yedges[-1]))).sum())
            style.tag(ax, f"S{sample}, test set", loc="lower left")
            ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
            caption = (f"N={int(mask.sum()):,}; invalid={excluded:,}; trueE outside view={outside:,}; "
                       f"y outside view={vertical:,}. Quantiles use all y; require N >= {MIN_PER_BIN}.")
            style.save(fig, out, f"sample{sample}_{name}", caption=caption, formats=("png", "pdf"))
            np.savez_compressed(os.path.join(out, f"sample{sample}_{name}_counts.npz"),
                                counts=h, slice_counts=counts, conditional_probability=probability,
                                xedges=xedges, yedges=yedges, quantiles=quantiles,
                                n_total=int(mask.sum()), n_invalid=excluded, n_x_outside=outside, n_y_outside=vertical)
    with open(os.path.join(out, "slice_statistics.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summary_rows(smp, true, pred, targets):
    in_win = in_headline_window(true[targets[0]])
    t0 = true[targets[0]]
    rows = []
    for t in targets:
        r = relative_error(pred[t], true[t])
        for k, _ in SAMPLES:
            m = sample_mask(smp, k)
            wins = [("all E", np.ones_like(m)), (WIN_LABEL, in_win)]
            wins += [(f"trueE {lo:g}-{hi:g} GeV" if np.isfinite(hi) else f"trueE > {lo:g} GeV",
                      (t0 > lo) & (t0 <= hi)) for lo, hi in HIGH_E]
            for win, wm in wins:
                rows.append(dict(target=t, sample=k, window=win, n_outside_window=int(np.sum(m & ~in_win)),
                                 **stats(r[m & wm])))
    return rows


def plot_target(t, targets, smp, true, pred, rows, out):
    r = relative_error(pred[t], true[t])
    t0 = true[targets[0]]
    in_win, above = in_headline_window(t0), t0 > WINDOW[1]
    masks = {k: sample_mask(smp, k) for k, _ in SAMPLES}
    st = style.SAMPLE_STYLE
    xlabel = f"{t}: (pred − true) / true"

    fig = plots.relerr_hist([(st[k]["label"], st[k], r[masks[k] & in_win]) for k, _ in SAMPLES], xlabel,
                            tag=f"trueE {WINDOW[0]:g}–{WINDOW[1]:g} GeV")
    style.save(fig, out, f"relerr_{t}", formats=("png",))

    hi = 6.0 if t == targets[0] else 5.0
    fig, _ = plots.bias_resolution_panels([(st[k]["label"], st[k], true[t][masks[k]], r[masks[k]]) for k, _ in SAMPLES],
                                          np.arange(0.0, hi + 1e-9, 0.25), f"True {t} (GeV)", tag=t)
    style.save(fig, out, f"vs_true_{t}", formats=("png",))

    fig, (ax1, ax2) = plots.bias_resolution_panels(
        [(st[k]["label"], st[k], true[t][masks[k]], r[masks[k]]) for k, _ in SAMPLES], WIDE_EDGES,
        f"True {t} (GeV), log scale", centers=np.sqrt(WIDE_EDGES[1:] * WIDE_EDGES[:-1]), logx=True, tag=t)
    for ax in (ax1, ax2):
        ax.axvline(WINDOW[1], color=style.MUTED, lw=1, ls=":")
    ax2.set_xticks([0.5, 1, 2, 3, 6, 10, 20, 50], ["0.5", "1", "2", "3", "6", "10", "20", "50"])
    style.save(fig, out, f"vs_true_{t}_wide", formats=("png",))

    series = [(f"{st[k]['label']} (n={(masks[k] & above).sum()})", st[k], r[masks[k] & above])
              for k, _ in SAMPLES if (masks[k] & above).sum() >= MIN_PER_BIN]
    fig = plots.relerr_hist(series, xlabel, tag=f"trueE > {WINDOW[1]:g} GeV")
    style.save(fig, out, f"relerr_{t}_above6", formats=("png",))

    sub = [x for x in rows if x["target"] == t and x["window"] == WIN_LABEL]
    fig, (ax1, ax2) = style.figure(ncols=2, size=(4.2, 3.8), gridspec_kw=dict(wspace=0.45))
    for i, (k, _) in enumerate(SAMPLES):
        x = next(y for y in sub if y["sample"] == k)
        ax1.plot(i, x["bias"] * 100, marker=st[k]["marker"], color=st[k]["color"], ms=9, ls="none")
        ax2.plot(i, x["res"] * 100, marker=st[k]["marker"], color=st[k]["color"], ms=9, ls="none")
        ax1.annotate(f"{x['bias']:+.1%}", (i, x["bias"] * 100), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=10)
        ax2.annotate(f"{x['res']:.1%}", (i, x["res"] * 100), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=10)
    ax1.axhline(0, color=style.MUTED, lw=1, ls=":")
    for ax, yl in ((ax1, "Bias [%]"), (ax2, "Resolution [%]")):
        ax.set_xticks(range(len(SAMPLES)), ["S1", "S2", "S3", "All"])
        ax.set_xlim(-0.5, len(SAMPLES) - 0.5)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_ylabel(yl, loc="top")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)
    lo, hi1 = ax1.get_ylim()
    ax1.set_ylim(lo - 0.15 * (hi1 - lo), hi1 + 0.15 * (hi1 - lo))
    style.tag(ax1, f"{t}, trueE {WINDOW[0]:g}–{WINDOW[1]:g} GeV")
    style.save(fig, out, f"summary_{t}", formats=("png",))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the run's JSON config")
    p.add_argument("--model-dir", help="default: newest save_path/model_*/ with a result.csv")
    p.add_argument("--code-dir", default=data.REPO, help="transformer_EE checkout (default: this one)")
    p.add_argument("--flattening-only", metavar="OUT_DIR",
                   help="only make the before/after flattening plot into OUT_DIR (no trained model needed)")
    p.add_argument("--out-dir", help="fresh output directory; refuses an existing directory")
    p.add_argument("--conditional-maps-only", action="store_true",
                   help="only neutrino-energy conditional maps for samples 1/2/3, after alignment checks")
    a = p.parse_args()

    cfg = json.load(open(a.config))
    if not cfg.get("split_column"):
        sys.exit("config has no split_column; per-sample matching needs the split file")
    targets = cfg["target"]
    style.use()
    if a.flattening_only:
        os.makedirs(a.flattening_only, exist_ok=True)
        plot_flattening(cfg, a.code_dir, a.flattening_only)
        return
    model_dir = a.model_dir or data.find_model_dir(cfg["save_path"])
    out = a.out_dir or os.path.join(model_dir, "eval_by_sample")
    if a.conditional_maps_only and not a.out_dir:
        p.error("--conditional-maps-only requires a fresh --out-dir")
    if a.out_dir and os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")

    smp, true, pred, _ = data.load_test_predictions(cfg, model_dir)
    print(f"aligned {smp.size} test predictions with {cfg['data_path']}")
    os.makedirs(out, exist_ok=not bool(a.out_dir))
    if a.conditional_maps_only:
        conditional_maps(true["trueE"], pred["trueE"], smp, out)
        paths = [__file__, a.config, os.path.join(model_dir, "input.json"),
                 os.path.join(model_dir, "result.csv"), os.path.join(model_dir, "best_model.zip")]
        provenance = dict(model_dir=model_dir, data_path=cfg["data_path"], split_column=cfg["split_column"],
                          category_counts={k: int((smp == k).sum()) for k in ("1", "2", "3")},
                          aligned_targets=targets, hashes={os.path.abspath(p): data.sha256_file(p) for p in paths},
                          code=data.code_version(), config=cfg,
                          checkpoint_metadata=json.load(open(os.path.join(model_dir, "checkpoint_metadata.json"))))
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)
        print(f"wrote conditional maps to {out}")
        return
    plot_flattening(cfg, a.code_dir, out)

    rows = summary_rows(smp, true, pred, targets)
    pl.DataFrame(rows).write_csv(os.path.join(out, "summary.csv"))
    in_win = in_headline_window(true[targets[0]])
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"# Per-sample evaluation\n\nmodel dir: `{model_dir}`  \ndata: `{cfg['data_path']}`  \n"
                f"r = (pred - true)/true; bias = median(r); resolution = (q84 - q16)/2\n\n")
        f.write(report.stats_table(rows, "sample"))
        f.write("\nEvents outside the trueE window per sample: " +
                ", ".join(f"{k}: {int(np.sum(sample_mask(smp, k) & ~in_win))}" for k, _ in SAMPLES) + "\n")
    for t in targets:
        plot_target(t, targets, smp, true, pred, rows, out)
    plt.close("all")
    print(f"wrote {out}")
    print(open(os.path.join(out, "summary.md")).read())


if __name__ == "__main__":
    main()
