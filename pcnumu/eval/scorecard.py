#!/usr/bin/env python
"""Scorecard for one finished TransformerEE run: headline metrics, peak pull and diagnostic plots.

    python -I pcnumu/eval/scorecard.py RUN_DIR [--model-dir DIR] [--out-dir DIR]
    python -I pcnumu/eval/scorecard.py RUN_DIR --peak-only     # (re)write peak_metrics.json only

RUN_DIR is the run folder (the config's save_path: train.log, model_*/). Inputs are the model
folder's result.csv, input.json and checkpoint_metadata.json, the run's train.log, and the test
rows of the run's data file. Nothing is re-inferred. Writes into OUT_DIR (default
RUN_DIR/scorecard; refuses an existing directory):
  scorecard.json / .sha256 / .md   headline metrics per target and sample; identical bytes for identical inputs
  peak_metrics.json                peak pull, pull slope and bias curve per sample (metrics.peak_measures)
  loss_curves.png, loss_history.csv
  errmap_<target>_sample<k>.png    error map (eval_model.C view_0to6 family) with ±5/±10% bands
  errmap_stats.csv                 per-bin counts and 16/50/84th percentiles behind the error maps
  ellipse_sample<k>.png            joint trueE/trueLepE error plane with the frozen ellipse
Metric definitions: pcnumu/core/metrics.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from pcnumu.core import data, report, style  # noqa: E402
from pcnumu.core.metrics import (BANDS, ERRMAP, PEAK_DEFINITIONS, PEAK_HIGH, PEAK_LOW, PEAK_VERSION,  # noqa: E402
                                 SAMPLES, WINDOW, ellipse_axes, ellipse_fraction, errmap_bins,
                                 in_headline_window, peak_measures, relative_error, sample_mask, stats)

SCHEMA = "pcnumu-scorecard/2"  # 2: frozen ellipse, unbinned headline fraction
AXIS_LABEL = {"trueE": "True neutrino energy (GeV)", "trueLepE": "True muon energy (GeV)"}


# ---------- numbers ----------
def build_scorecard(cfg, model_dir, run_dir, smp, true, pred, fingerprint, losses, n_sessions):
    targets = cfg["target"]
    t0 = targets[0]
    in_win = in_headline_window(true[t0])
    r = {t: relative_error(pred[t], true[t]) for t in targets}

    metrics = {}
    for t in targets:
        metrics[t] = {}
        for k, _ in SAMPLES:
            m = sample_mask(smp, k)
            s = stats(r[t][m & in_win])
            metrics[t][k] = dict(n=s["n"], n_outside_window=int(np.sum(m & ~in_win)),
                                 bias=float(s["bias"]), resolution=float(s["res"]),
                                 band_05=s["f5"], band_10=s["f10"], band_15=s["f15"])
    axes = ellipse_axes(fingerprint)
    ellipse = {}
    if len(targets) >= 2:
        for k, _ in SAMPLES:
            m = sample_mask(smp, k) & in_win
            e = ellipse_fraction(100 * r[targets[0]][m], 100 * r[targets[1]][m], axes["a_percent"], axes["b_percent"])
            # Headline = unbinned (exact per event); binned = eval_model.C port, kept as a cross-check.
            ellipse[k] = dict(fraction=e["unbinned_fraction"], binned_fraction=e["fraction"],
                              binned_n_out_of_range=e["n_out_of_range"],
                              binned_x_range=e["x_range"], binned_y_range=e["y_range"])

    meta = json.load(open(os.path.join(model_dir, "checkpoint_metadata.json")))
    best = meta["best_model"]
    files = {name: os.path.join(model_dir, name) for name in
             ("result.csv", "input.json", "checkpoint_metadata.json", best.get("filename", "best_model.zip"))}
    return {
        "schema": SCHEMA,
        "run_dir": os.path.abspath(run_dir),
        "model_dir": os.path.abspath(model_dir),
        "model_phys_name": cfg.get("model_phys_name"),
        "data_path": cfg["data_path"],
        "split_column": cfg["split_column"],
        "test_set": dict(n=int(smp.size), counts={k: int(np.sum(smp == k)) for k in ("1", "2", "3")},
                         fingerprint_sha256=fingerprint,
                         fingerprint_of=f"test rows' {'/'.join(data.ID_COLUMNS)}/sample, file order, CSV text"),
        "definitions": dict(
            relative_error="r = (pred - true) / true", bias="median(r)", resolution="(q84 - q16) / 2 of r",
            tolerance_bands=[f"|r| < {b:g}" for b in BANDS],
            window=f"true {t0} in [{WINDOW[0]}, {WINDOW[1]}] GeV, applied to every target",
            ellipse=(f"fraction = share of window events with (x/a)^2 + (y/b)^2 <= 1, x = 100 r_{targets[0]}, "
                     f"y = 100 r_{targets[1] if len(targets) > 1 else '-'} (unbinned); binned_fraction = "
                     "fractionInsideEllipse port (graph_eval/eval_model.C), 200x200 bins, bin centres, "
                     "in-range denominator"),
            source="pcnumu/core/metrics.py stats()"),
        "metrics": metrics,
        "ellipse": dict(axes, by_sample=ellipse),
        "training": dict(best_epoch=best["epoch"], best_validation_loss=best["validation_loss"],
                         epoch_numbering="zero-based, as in train.log and checkpoint_metadata.json",
                         epochs_run=len(losses), log_sessions=n_sessions,
                         loss_history="train.log per-epoch lines, last session, rounded to 4 decimals",
                         early_stopping=meta.get("early_stopping")),
        "provenance": dict(hashes={name: data.sha256_file(p) for name, p in files.items() if os.path.exists(p)},
                           scorecard_script_sha256=data.sha256_file(os.path.abspath(__file__)),
                           code_sha256=data.code_hashes(os.path.abspath(__file__)),
                           code=data.code_version()),
    }


def build_peak(smp, true, pred, fingerprint, model_dir):
    out = {"version": PEAK_VERSION, "target": "trueE", "definitions": PEAK_DEFINITIONS,
           "low_bin_gev": list(PEAK_LOW), "high_bin_gev": list(PEAK_HIGH), "test_fingerprint": fingerprint,
           "model_dir": model_dir, "by_sample": {}}
    for s in ["1", "2", "3"]:
        m = smp == s
        out["by_sample"][s] = peak_measures(true["trueE"][m], pred["trueE"][m])
    return out


def render_md(sc, targets):
    L = [f"# Scorecard: {sc['model_phys_name']}", "",
         f"run: `{sc['run_dir']}`  ", f"model: `{sc['model_dir']}`  ",
         f"test set: {sc['test_set']['n']} events (S1 {sc['test_set']['counts']['1']}, "
         f"S2 {sc['test_set']['counts']['2']}, S3 {sc['test_set']['counts']['3']}), "
         f"fingerprint `{sc['test_set']['fingerprint_sha256'][:16]}`  ",
         f"window: {sc['definitions']['window']}; bias = median r, resolution = (q84 - q16)/2", ""]
    for t in targets:
        L += [f"## {t}", "", "| sample | N | bias | resolution | within ±5% | ±10% | ±15% |",
              "|---|---|---|---|---|---|---|"]
        for k, _ in SAMPLES:
            m = sc["metrics"][t][k]
            L.append(f"| {k} | {m['n']} | {m['bias']:+.2%} | {m['resolution']:.2%} | "
                     f"{m['band_05']:.1%} | {m['band_10']:.1%} | {m['band_15']:.1%} |")
        L.append("")
    e = sc["ellipse"]
    if e["by_sample"]:
        state = (f"frozen as {e['frozen_as']}" if not e["provisional"]
                 else "PROVISIONAL: test set not in ellipse_frozen.json")
        L += [f"## Ellipse fraction (a = {e['a_percent']:g}%, b = {e['b_percent']:g}%, {state})", "",
              "| sample | fraction (unbinned) | binned cross-check (eval_model.C) |",
              "|---|---|---|"]
        for k, _ in SAMPLES:
            x = e["by_sample"][k]
            L.append(f"| {k} | {x['fraction']:.1%} | {x['binned_fraction']:.1%} |")
        L.append("")
    tr = sc["training"]
    L += ["## Training", "", f"best validation loss {tr['best_validation_loss']:.6f} at epoch {tr['best_epoch']} "
          f"(zero-based) of {tr['epochs_run']} run; log sessions: {tr['log_sessions']}", "",
          f"code: `{sc['provenance']['code']['commit']}` dirty={sc['provenance']['code']['dirty']}", ""]
    return "\n".join(L)


# ---------- plots ----------
def plot_losses(losses, best_epoch, best_loss, out):
    with open(os.path.join(out, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        w.writerows(losses)
    if not losses:
        return
    ep, tr, va = (np.array(c) for c in zip(*losses))
    fig, ax = style.figure()
    ax.plot(ep, tr, **{**style.SERIES[1], "ms": 4}, label="Training (epoch mean, dropout on)")
    ax.plot(ep, va, **{**style.SERIES[0], "ms": 4}, label="Validation")
    # Best epoch from checkpoint_metadata.json: the log's 4-decimal losses can tie.
    ax.axvline(best_epoch, color=style.MUTED, lw=1, ls=":")
    style.labels(ax, "Epoch (zero-based)", "Loss")
    style.tag(ax, f"Best checkpoint: epoch {best_epoch}\nvalidation loss {best_loss:.6f}", loc="lower left")
    ax.legend(loc="upper right")
    style.save(fig, out, "loss_curves", formats=("png",))


def plot_errmaps(smp, true, pred, targets, out):
    nb, lo, hi = ERRMAP["bins"], ERRMAP["xmin"], ERRMAP["xmax"]
    width = (hi - lo) / nb
    centers = lo + (np.arange(nb) + 0.5) * width
    rows = []
    for t in targets:
        r = relative_error(pred[t], true[t])
        for k, _ in SAMPLES:
            m = sample_mask(smp, k)
            b, y = errmap_bins(true[t][m], r[m])
            q = np.full((nb, 3), np.nan)
            n = np.bincount(b, minlength=nb)
            for i in range(nb):
                if n[i]:
                    q[i] = np.quantile(y[b == i], [0.16, 0.5, 0.84])  # linear, as quantile_range()
                rows.append(dict(target=t, sample=k, true_lo=round(lo + i * width, 6),
                                 true_hi=round(lo + (i + 1) * width, 6), n=int(n[i]),
                                 q16=q[i, 0], median=q[i, 1], q84=q[i, 2]))
            fig, ax = style.figure(size=(7.2, 5.4))
            h, xe, ye = np.histogram2d(centers[b], y, bins=(np.linspace(lo, hi, nb + 1), np.linspace(-1, 1, 201)))
            if h.sum():
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(h.T == 0, h.T), norm=LogNorm(vmin=1, vmax=max(h.max(), 1)),
                                   shading="flat")
                fig.colorbar(im, ax=ax, label="Events")
            # Bands are nested: the ±5% span sits on top of the ±10% one, so it reads darker.
            ax.axhspan(-0.10, 0.10, color=style.BARS[1], alpha=0.10, lw=0, label="±10% band")
            ax.axhspan(-0.05, 0.05, color=style.BARS[1], alpha=0.20, lw=0, label="±5% band")
            ax.plot(centers, q[:, 1], color="black", lw=1.8, label="Median")
            ax.plot(centers, q[:, 0], color=style.BARS[1], lw=1.6, ls="--", label="16th–84th percentile")
            ax.plot(centers, q[:, 2], color=style.BARS[1], lw=1.6, ls="--")
            ax.axhline(0, color="black", lw=0.8)
            ax.set(xlim=(lo, hi), ylim=(-1, 1))
            style.labels(ax, AXIS_LABEL.get(t, f"True {t} (GeV)"), f"{t}: (reco − true) / true")
            style.tag(ax, f"{'All samples' if k == 'all' else 'S' + k}, test set", loc="lower left")
            ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
            style.save(fig, out, f"errmap_{t}_sample{k}", formats=("png",))
    with open(os.path.join(out, "errmap_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_ellipses(smp, true, pred, targets, ellipse, out, view=60.0):
    """Joint error plane per sample with the ellipse drawn; display binning 1% over ±view."""
    if len(targets) < 2 or not ellipse["by_sample"]:
        return
    t0, t1 = targets[0], targets[1]
    in_win = in_headline_window(true[t0])
    x_all = 100 * relative_error(pred[t0], true[t0])
    y_all = 100 * relative_error(pred[t1], true[t1])
    edges = np.arange(-view, view + 1e-9, 1.0)
    a, b = ellipse["a_percent"], ellipse["b_percent"]
    for k, _ in SAMPLES:
        m = sample_mask(smp, k) & in_win
        h, _, _ = np.histogram2d(x_all[m], y_all[m], bins=(edges, edges))
        fig, ax = style.figure(size=(6.6, 5.8))
        if h.sum():
            im = ax.pcolormesh(edges, edges, np.ma.masked_where(h.T == 0, h.T),
                               norm=LogNorm(vmin=1, vmax=max(h.max(), 1)), shading="flat")
            fig.colorbar(im, ax=ax, label="Events / (1% × 1%)", shrink=0.85)
        frac = ellipse["by_sample"][k]["fraction"]
        ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, fill=False, color=style.BARS[1], lw=2.2,
                             label=f"±{a:g}% × ±{b:g}%{' (provisional)' if ellipse['provisional'] else ''}: "
                                   f"{frac:.1%} inside"))
        ax.set(xlim=(-view, view), ylim=(-view, view), aspect="equal")
        style.labels(ax, f"{t0} error [%]", f"{t1} error [%]")
        style.tag(ax, f"{'All samples' if k == 'all' else 'S' + k}, true {t0} {WINDOW[0]:g}–{WINDOW[1]:g} GeV")
        ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
        style.save(fig, out, f"ellipse_sample{k}", formats=("png",))


# ---------- main ----------
def write_peak_only(run_dir):
    """Old peak_metrics.py behaviour: RUN_DIR/scorecard/peak_metrics.json, rewritten only if it changes."""
    card = data.scorecard_of(run_dir)
    model_dir = card["model_dir"]
    cfg = json.load(open(os.path.join(model_dir, "input.json")))
    smp, true, pred, fp = data.load_test_predictions(cfg, model_dir)
    if fp != card["test_set"]["fingerprint_sha256"]:
        sys.exit("test-set fingerprint differs from the scorecard's")
    peak = build_peak(smp, true, pred, fp, model_dir)
    path = os.path.join(run_dir, "scorecard", "peak_metrics.json")
    text = json.dumps(peak, indent=1, sort_keys=True) + "\n"
    if not (os.path.exists(path) and open(path).read() == text):
        open(path, "w").write(text)
    for s, v in peak["by_sample"].items():
        print(f"S{s}: peak_pull {v['peak_pull']:+.2f} pts, pull_slope {v['pull_slope']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run folder (the config's save_path)")
    p.add_argument("--model-dir", help="default: newest RUN_DIR/model_*/ with a result.csv")
    p.add_argument("--out-dir", help="new output directory (default RUN_DIR/scorecard)")
    p.add_argument("--peak-only", action="store_true", help="only (re)write RUN_DIR/scorecard/peak_metrics.json")
    a = p.parse_args()
    if a.peak_only:
        write_peak_only(a.run_dir)
        return

    model_dir = a.model_dir or data.find_model_dir(a.run_dir)
    out = a.out_dir or os.path.join(a.run_dir, "scorecard")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")
    ancestor = os.path.dirname(os.path.abspath(out))
    while not os.path.isdir(ancestor):
        ancestor = os.path.dirname(ancestor)
    if not os.access(ancestor, os.W_OK):
        sys.exit(f"{ancestor} is not writable (frozen run folder?); pass --out-dir")
    cfg = json.load(open(os.path.join(model_dir, "input.json")))

    smp, true, pred, fingerprint = data.load_test_predictions(cfg, model_dir)
    print(f"aligned {smp.size} test predictions with {cfg['data_path']}")
    log = os.path.join(a.run_dir, "train.log")
    losses, n_sessions = data.parse_losses(log) if os.path.exists(log) else ([], 0)
    if n_sessions > 1:
        print(f"warning: {log} holds {n_sessions} training sessions; using the last one")

    sc = build_scorecard(cfg, model_dir, a.run_dir, smp, true, pred, fingerprint, losses, n_sessions)
    os.makedirs(out)
    digest = report.write_json(os.path.join(out, "scorecard.json"), sc, sidecar=True)
    with open(os.path.join(out, "scorecard.md"), "w") as f:
        f.write(render_md(sc, cfg["target"]))
    with open(os.path.join(out, "peak_metrics.json"), "w") as f:
        f.write(json.dumps(build_peak(smp, true, pred, fingerprint, model_dir), indent=1, sort_keys=True) + "\n")
    style.use()
    best = json.load(open(os.path.join(model_dir, "checkpoint_metadata.json")))["best_model"]
    plot_losses(losses, best["epoch"], best["validation_loss"], out)
    plot_errmaps(smp, true, pred, cfg["target"], out)
    plot_ellipses(smp, true, pred, cfg["target"], sc["ellipse"], out)
    print(f"wrote {out} (scorecard sha256 {digest})")
    print(render_md(sc, cfg["target"]))


if __name__ == "__main__":
    main()
