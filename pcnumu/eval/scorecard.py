#!/usr/bin/env python
"""Scorecard and diagnostics for one finished TransformerEE run (experiment loop, issue #3).

    python scorecard.py RUN_DIR [--model-dir DIR] [--out-dir DIR]

RUN_DIR is the run folder (the config's save_path: train.log, model_*/). Inputs are the
model directory's result.csv, input.json and checkpoint_metadata.json, the run's train.log,
and the test rows (split column == "test") of the run's data file. Nothing is re-inferred.

Writes into OUT_DIR (default RUN_DIR/scorecard; refuses an existing directory):
  scorecard.json      machine-readable scorecard; identical bytes for identical inputs
  scorecard.sha256    sha256 of scorecard.json
  scorecard.md        human-readable rendering
  loss_curves.png     per-epoch train/valid loss from the last training session in train.log
  loss_history.csv
  errmap_<target>_sample<k>.png   error map, eval_model.C view_0to6 family, ±5/±10% bands
  errmap_stats.csv    per-bin counts and 16/50/84th percentiles behind the error maps
  ellipse_sample<k>.png           joint trueE/trueLepE error plane with the ellipse drawn

Metric definitions are those of eval_by_sample.py (imported, not copied):
r = (pred - true) / true; bias = median(r); resolution = (q84 - q16) / 2; tolerance-band
fraction = share of events with |r| < band; window = trueE in [0.5, 6] GeV for both targets.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_by_sample import SAMPLES, STYLE, WINDOW, find_model_dir, stats, style_axes  # noqa: E402

SCHEMA = "pcnumu-scorecard/2"  # 2: frozen ellipse, unbinned headline fraction
BANDS = (0.05, 0.10, 0.15)
ID_COLUMNS = ["run", "subrun", "cycle", "evt", "subevt"]
# Ellipse in the (trueE error %, trueLepE error %) plane, centred at (0, 0). The axes are
# frozen per test set in ellipse_frozen.json (issue #4); other test sets are flagged provisional.
ELLIPSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ellipse_frozen.json")
ERRMAP = dict(bins=60, xmin=0.0, xmax=6.0)  # eval_model.C graph_resolution_stat "view_0to6_"
GIT = "/data/chow/tools/git-env/bin/git"
AXIS_LABEL = {"trueE": "true neutrino energy (trueE)", "trueLepE": "true muon energy (trueLepE)"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version():
    """Commit and dirty flag of the checkout this script runs from (None if not a checkout)."""
    here = os.path.dirname(os.path.abspath(__file__))
    git = GIT if os.path.exists(GIT) else "git"
    try:
        commit = subprocess.run([git, "-C", here, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run([git, "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


# ---------- inputs ----------
def load_test_predictions(cfg, model_dir):
    """Predictions aligned with the data file's test rows; same checks as eval_by_sample.py."""
    sc = cfg.get("split_column")
    if not sc:
        sys.exit("config has no split_column; the frozen test set needs the split file")
    targets = cfg["target"]
    res = pl.read_csv(os.path.join(model_dir, "result.csv"))
    test = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(sc) == "test")
            .select(ID_COLUMNS + ["sample"] + targets)
            .collect())
    if test.height != res.height:
        sys.exit(f"row mismatch: {res.height} predictions vs {test.height} test rows in {cfg['data_path']}")
    smp = test["sample"].to_numpy().astype(str)
    if not set(smp) <= {"1", "2", "3"}:
        sys.exit("unexpected sample labels")
    true = {t: test[t].cast(pl.Float64).to_numpy() for t in targets}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in targets}
    for t in targets:  # result.csv stores float32 truths: they must match the file row by row
        rt = res[f"true_{t}"].to_numpy().astype(np.float64)
        bad = int(np.sum(~np.isclose(rt, true[t], rtol=1e-5, atol=1e-6)))
        if bad:
            sys.exit(f"{bad} rows where result.csv true_{t} != test-row {t}: predictions are not aligned")
        if not np.all(np.isfinite(true[t]) & (true[t] > 0)):
            sys.exit(f"nonpositive or nonfinite {t}; cannot compute relative error")
    # Identity of the test set: event IDs and sample labels of the test rows, in file order.
    fingerprint = hashlib.sha256(test.select(ID_COLUMNS + ["sample"]).write_csv().encode()).hexdigest()
    return smp, true, pred, fingerprint


def parse_losses(log_path):
    """Per-epoch (epoch, train_loss, valid_loss) from the LAST training session in train.log.

    train.log is opened in append mode, so a re-launch into the same folder appends a second
    session (r04 appended to r03's log). Sessions start at a "Start Time:" line.
    """
    pat = re.compile(r"^Epoch: (\d+), train_loss: (\S+), valid_loss: (\S+)\s*$")
    sessions = [[]]
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("Start Time:"):
                sessions.append([])
            m = pat.match(line)
            if m:
                sessions[-1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    n_sessions = sum(1 for s in sessions if s)
    return (sessions[-1] if sessions[-1] else []), n_sessions


# ---------- ellipse: port of graph_eval/eval_model.C ----------
def _root_axis(v, nbins):
    """Range as eval_model.C builds h2_energy_theta's x axis: data min/max, clamp_res_range
    to [-200, 200], then pad 5% of max(1, width)."""
    lo, hi = max(float(v.min()), -200.0), min(float(v.max()), 200.0)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * max(1.0, abs(hi - lo))
    return lo - pad, hi + pad, nbins


def _root_findbin(v, lo, hi, nbins):
    """TAxis::FindBin for a fixed-width axis: 0 underflow, 1..nbins, nbins+1 overflow."""
    b = 1 + np.floor(nbins * (v - lo) / (hi - lo)).astype(np.int64)
    b[v < lo] = 0
    b[v >= hi] = nbins + 1
    return b


def ellipse_fraction(x, y, a, b, nbins=200):
    """fractionInsideEllipse(h2, 0, 0, a, b, 0) on a 200x200 TH2D filled with (x, y).

    Bin centres are tested against the ellipse; the denominator is the histogram's in-range
    content, so entries beyond about ±210% on either axis are not counted. The y axis gets the
    same range rule as x (eval_model.C's y axis was an angle with a fixed ±180 range).
    """
    xa, ya = _root_axis(x, nbins), _root_axis(y, nbins)
    bx, by = _root_findbin(x, *xa), _root_findbin(y, *ya)
    ok = (bx >= 1) & (bx <= nbins) & (by >= 1) & (by <= nbins)
    cx = xa[0] + (bx[ok] - 0.5) * (xa[1] - xa[0]) / nbins
    cy = ya[0] + (by[ok] - 0.5) * (ya[1] - ya[0]) / nbins
    inside = int(np.sum(cx * cx / (a * a) + cy * cy / (b * b) <= 1.0))
    total = int(ok.sum())
    return dict(fraction=inside / total if total else None, n_in_histogram=total,
                n_out_of_range=int(x.size - total), x_range=[xa[0], xa[1]], y_range=[ya[0], ya[1]],
                unbinned_fraction=float(np.mean(x * x / (a * a) + y * y / (b * b) <= 1.0)) if x.size else None)


def ellipse_axes(fingerprint):
    """Frozen axes for this test set, or the provisional fallback if it is not listed."""
    spec = json.load(open(ELLIPSE_FILE))
    for name, e in spec["frozen"].items():
        for horn, fp in e["test_sets"].items():
            if fp == fingerprint:
                return dict(a_percent=e["a_percent"], b_percent=e["b_percent"], provisional=False,
                            frozen_as=f"{name}/{horn}", definition=e["definition"])
    fb = spec["provisional_fallback"]
    return dict(a_percent=fb["a_percent"], b_percent=fb["b_percent"], provisional=True,
                frozen_as=None, definition="unbinned")


# ---------- scorecard ----------
def build_scorecard(cfg, model_dir, run_dir, smp, true, pred, fingerprint, losses, n_sessions):
    targets = cfg["target"]
    t0 = targets[0]
    in_win = (true[t0] >= WINDOW[0]) & (true[t0] <= WINDOW[1])
    masks = {k: (smp == k) if k != "all" else np.ones(smp.size, bool) for k, _ in SAMPLES}
    r = {t: (pred[t] - true[t]) / true[t] for t in targets}

    metrics = {}
    for t in targets:
        metrics[t] = {}
        for k, _ in SAMPLES:
            s = stats(r[t][masks[k] & in_win])
            metrics[t][k] = dict(n=s["n"], n_outside_window=int(np.sum(masks[k] & ~in_win)),
                                 bias=float(s["bias"]), resolution=float(s["res"]),
                                 band_05=s["f5"], band_10=s["f10"], band_15=s["f15"])
    axes = ellipse_axes(fingerprint)
    ellipse = {}
    if len(targets) >= 2:
        for k, _ in SAMPLES:
            m = masks[k] & in_win
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
                         fingerprint_of=f"test rows' {'/'.join(ID_COLUMNS)}/sample, file order, CSV text"),
        "definitions": dict(
            relative_error="r = (pred - true) / true", bias="median(r)", resolution="(q84 - q16) / 2 of r",
            tolerance_bands=[f"|r| < {b:g}" for b in BANDS],
            window=f"true {t0} in [{WINDOW[0]}, {WINDOW[1]}] GeV, applied to every target",
            ellipse=(f"fraction = share of window events with (x/a)^2 + (y/b)^2 <= 1, x = 100 r_{targets[0]}, "
                     f"y = 100 r_{targets[1] if len(targets) > 1 else '-'} (unbinned); binned_fraction = "
                     "fractionInsideEllipse port (graph_eval/eval_model.C), 200x200 bins, bin centres, "
                     "in-range denominator"),
            source="pcnumu/eval/eval_by_sample.py stats()"),
        "metrics": metrics,
        "ellipse": dict(axes, by_sample=ellipse),
        "training": dict(best_epoch=best["epoch"], best_validation_loss=best["validation_loss"],
                         epoch_numbering="zero-based, as in train.log and checkpoint_metadata.json",
                         epochs_run=len(losses), log_sessions=n_sessions,
                         loss_history="train.log per-epoch lines, last session, rounded to 4 decimals",
                         early_stopping=meta.get("early_stopping")),
        "provenance": dict(hashes={name: sha256_file(p) for name, p in files.items() if os.path.exists(p)},
                           scorecard_script_sha256=sha256_file(os.path.abspath(__file__)),
                           code=code_version()),
    }


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


# ---------- diagnostics ----------
def plot_losses(losses, best_epoch, best_loss, out):
    with open(os.path.join(out, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        w.writerows(losses)
    if not losses:
        return
    ep, tr, va = (np.array(c) for c in zip(*losses))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, tr, color=STYLE["1"]["color"], marker="o", ms=4, lw=2, label="training (epoch mean, dropout on)")
    ax.plot(ep, va, color=STYLE["all"]["color"], marker="D", ms=4, lw=2, ls="--", label="validation")
    # Best epoch from checkpoint_metadata.json: the log's 4-decimal losses can tie.
    ax.axvline(best_epoch, color="#6b6b6b", lw=1, ls=":")
    ax.annotate(f"best checkpoint: epoch {best_epoch}\nvalidation loss {best_loss:.6f}", (best_epoch, best_loss),
                textcoords="offset points", xytext=(-8, 28), ha="right", fontsize=8)
    ax.set(xlabel="epoch (zero-based)", ylabel="loss", title="Loss per epoch (train.log, last session)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss_curves.png"), dpi=150)
    plt.close(fig)


def errmap_bins(x, y):
    """graph_resolution_stat binning: bin = int((x - xmin) / width) for xmin <= x <= xmax,
    kept if 0 <= bin < bins (so x == xmax is dropped in a fixed-range view)."""
    nb, lo, hi = ERRMAP["bins"], ERRMAP["xmin"], ERRMAP["xmax"]
    keep = (x >= lo) & (x <= hi)
    b = ((x[keep] - lo) / ((hi - lo) / nb)).astype(np.int64)
    ok = (b >= 0) & (b < nb)
    return b[ok], y[keep][ok]


def plot_errmaps(smp, true, pred, targets, out):
    nb, lo, hi = ERRMAP["bins"], ERRMAP["xmin"], ERRMAP["xmax"]
    width = (hi - lo) / nb
    centers = lo + (np.arange(nb) + 0.5) * width
    rows = []
    for t in targets:
        r = (pred[t] - true[t]) / true[t]
        for k, label in SAMPLES:
            m = (smp == k) if k != "all" else np.ones(smp.size, bool)
            b, y = errmap_bins(true[t][m], r[m])
            q = np.full((nb, 3), np.nan)
            n = np.bincount(b, minlength=nb)
            for i in range(nb):
                if n[i]:
                    q[i] = np.quantile(y[b == i], [0.16, 0.5, 0.84])  # linear, as quantile_range()
                rows.append(dict(target=t, sample=k, true_lo=round(lo + i * width, 6),
                                 true_hi=round(lo + (i + 1) * width, 6), n=int(n[i]),
                                 q16=q[i, 0], median=q[i, 1], q84=q[i, 2]))
            fig, ax = plt.subplots(figsize=(7.5, 5.6))
            h, xe, ye = np.histogram2d(centers[b], y, bins=(np.linspace(lo, hi, nb + 1), np.linspace(-1, 1, 201)))
            if h.sum():
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(h.T == 0, h.T), cmap="viridis",
                                   norm=LogNorm(vmin=1, vmax=max(h.max(), 1)), shading="flat")
                fig.colorbar(im, ax=ax, label="events")
            # Bands are nested: the ±5% span sits on top of the ±10% one, so it reads darker.
            ax.axhspan(-0.10, 0.10, color="#e66101", alpha=0.12, lw=0, label="±10% tolerance band")
            ax.axhspan(-0.05, 0.05, color="#e66101", alpha=0.22, lw=0, label="±5% tolerance band")
            ax.plot(centers, q[:, 1], color="black", lw=2, label="median")
            ax.plot(centers, q[:, 0], color="#c0392b", lw=2, ls="--",
                    label="middle 68% of events\n(16th–84th percentile)")
            ax.plot(centers, q[:, 2], color="#c0392b", lw=2, ls="--")
            ax.axhline(0, color="black", lw=0.8)
            ax.set(xlim=(lo, hi), ylim=(-1, 1), xlabel=f"{AXIS_LABEL.get(t, 'true ' + t)} [GeV]",
                   ylabel=f"{t} error  (reco − true) / true")
            ax.set_title(f"{label}, test set", fontsize=11)
            ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"errmap_{t}_sample{k}.png"), dpi=150)
            plt.close(fig)
    with open(os.path.join(out, "errmap_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_ellipses(smp, true, pred, targets, ellipse, out, view=60.0):
    """Joint error plane (trueE error %, trueLepE error %) per sample with the ellipse drawn.

    Display binning is 1% over ±view; the fraction quoted is the scorecard's value.
    """
    from matplotlib.patches import Ellipse  # pylint: disable=C0415

    if len(targets) < 2 or not ellipse["by_sample"]:
        return
    t0, t1 = targets[0], targets[1]
    in_win = (true[t0] >= WINDOW[0]) & (true[t0] <= WINDOW[1])
    x_all = 100 * (pred[t0] - true[t0]) / true[t0]
    y_all = 100 * (pred[t1] - true[t1]) / true[t1]
    edges = np.arange(-view, view + 1e-9, 1.0)
    a, b = ellipse["a_percent"], ellipse["b_percent"]
    for k, label in SAMPLES:
        m = ((smp == k) if k != "all" else np.ones(smp.size, bool)) & in_win
        h, _, _ = np.histogram2d(x_all[m], y_all[m], bins=(edges, edges))
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        if h.sum():
            im = ax.pcolormesh(edges, edges, np.ma.masked_where(h.T == 0, h.T), cmap="viridis",
                               norm=LogNorm(vmin=1, vmax=max(h.max(), 1)), shading="flat")
            fig.colorbar(im, ax=ax, label="events / (1% × 1%)", shrink=0.85)
        frac = ellipse["by_sample"][k]["fraction"]
        ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, fill=False, color="#e66101", lw=2.5,
                             label=f"ellipse ±{a:g}% × ±{b:g}%{' (provisional)' if ellipse['provisional'] else ''}"
                                   f"\n{frac:.1%} of events inside"))
        ax.axhline(0, color="white", lw=0.6, alpha=0.6)
        ax.axvline(0, color="white", lw=0.6, alpha=0.6)
        ax.set(xlim=(-view, view), ylim=(-view, view), aspect="equal",
               xlabel=f"{t0} error  (reco − true) / true  [%]", ylabel=f"{t1} error  (reco − true) / true  [%]")
        ax.set_title(f"{label}, test set, true {t0} {WINDOW[0]:g}–{WINDOW[1]:g} GeV", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"ellipse_sample{k}.png"), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run folder (the config's save_path)")
    p.add_argument("--model-dir", help="default: newest RUN_DIR/model_*/ with a result.csv")
    p.add_argument("--out-dir", help="new output directory (default RUN_DIR/scorecard)")
    a = p.parse_args()

    model_dir = a.model_dir or find_model_dir(a.run_dir)
    out = a.out_dir or os.path.join(a.run_dir, "scorecard")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")
    ancestor = os.path.dirname(os.path.abspath(out))
    while not os.path.isdir(ancestor):
        ancestor = os.path.dirname(ancestor)
    if not os.access(ancestor, os.W_OK):
        sys.exit(f"{ancestor} is not writable (frozen run folder?); pass --out-dir")
    cfg = json.load(open(os.path.join(model_dir, "input.json")))

    smp, true, pred, fingerprint = load_test_predictions(cfg, model_dir)
    print(f"aligned {smp.size} test predictions with {cfg['data_path']}")
    log = os.path.join(a.run_dir, "train.log")
    losses, n_sessions = parse_losses(log) if os.path.exists(log) else ([], 0)
    if n_sessions > 1:
        print(f"warning: {log} holds {n_sessions} training sessions; using the last one")

    sc = build_scorecard(cfg, model_dir, a.run_dir, smp, true, pred, fingerprint, losses, n_sessions)
    os.makedirs(out)
    text = json.dumps(sc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(os.path.join(out, "scorecard.json"), "w") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    with open(os.path.join(out, "scorecard.sha256"), "w") as f:
        f.write(f"{digest}  scorecard.json\n")
    with open(os.path.join(out, "scorecard.md"), "w") as f:
        f.write(render_md(sc, cfg["target"]))
    best = json.load(open(os.path.join(model_dir, "checkpoint_metadata.json")))["best_model"]
    plot_losses(losses, best["epoch"], best["validation_loss"], out)
    plot_errmaps(smp, true, pred, cfg["target"], out)
    plot_ellipses(smp, true, pred, cfg["target"], sc["ellipse"], out)
    print(f"wrote {out} (scorecard sha256 {digest})")
    print(render_md(sc, cfg["target"]))


if __name__ == "__main__":
    main()
