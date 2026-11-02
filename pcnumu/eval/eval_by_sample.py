#!/usr/bin/env python
"""Per-sample evaluation of a finished TransformerEE run that used a `split_column` file.

Matches result.csv (test-set predictions, written in test-row file order) to the test rows
of the run's data file, checks the match row by row on the true energies, and writes to
<model_dir>/eval_by_sample/:
  summary.csv / summary.md      bias, resolution, tail fractions per sample and combined
  relerr_<target>.png           relative-error distributions, samples 1/2/3 + all
  vs_true_<target>.png          bias and resolution vs true value (two panels, shared x)
  summary_<target>.png          bias and resolution per sample, 0.5-6 GeV trueE window
  flattening_<target0>.png      training-row spectrum before/after the run's flattening weights

  python eval_by_sample.py --config RUN_CONFIG.json [--model-dir DIR]
Relative error r = (pred - true) / true. Bias = median(r). Resolution = (q84 - q16) / 2 of r.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SAMPLES = [("1", "Sample 1 (contained)"), ("2", "Sample 2 (peripheral, µ contained)"),
           ("3", "Sample 3 (peripheral, µ uncontained)"), ("all", "All samples")]
# Categorical slots 1-3 of the dataviz reference palette; "all" in ink. Markers/linestyles
# give each series a second, non-colour encoding.
STYLE = {"1": dict(color="#2a78d6", marker="o", ls="-"), "2": dict(color="#eb6834", marker="s", ls="-"),
         "3": dict(color="#1baf7a", marker="^", ls="-"), "all": dict(color="#2b2b2b", marker="D", ls="--")}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e3df"
WINDOW = (0.5, 6.0)  # GeV, applied on trueE
MIN_PER_BIN = 50


def find_model_dir(save_path):
    dirs = [d for d in glob.glob(os.path.join(save_path, "model_*")) if os.path.exists(os.path.join(d, "result.csv"))]
    if not dirs:
        sys.exit(f"no model_*/result.csv under {save_path}")
    return max(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "result.csv")))


def stats(r):
    if r.size == 0:
        return dict(n=0, bias=np.nan, res=np.nan, rms=np.nan, f5=np.nan, f10=np.nan, f15=np.nan)
    q16, q50, q84 = np.quantile(r, [0.16, 0.5, 0.84])
    a = np.abs(r)
    return dict(n=int(r.size), bias=q50, res=(q84 - q16) / 2, rms=float(np.sqrt(np.mean(r ** 2))),
                f5=float(np.mean(a < 0.05)), f10=float(np.mean(a < 0.10)), f15=float(np.mean(a < 0.15)))


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_flattening(cfg, code_dir, out):
    """True-energy spectrum of the training rows before and after the run's flattening weights.

    Rebuilds the weighter exactly as training does: transformer_ee's FlatSpectraWeights with
    the config's weight kwargs, on target[0] of the training split.
    """
    sys.path.insert(0, code_dir)
    from transformer_ee.utils.weights import create_weighter  # pylint: disable=C0415

    t0 = cfg["target"][0]
    train = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
             .filter(pl.col(cfg["split_column"]) == "train")
             .select(["sample", t0])
             .collect())
    y = train[t0].cast(pl.Float64).to_numpy()
    smp = train["sample"].to_numpy().astype(str)
    w = create_weighter(cfg, y).getweight(y)
    edges = np.linspace(0, 10, 101)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for k, label in SAMPLES:
        m = (smp == k) if k != "all" else np.ones(smp.size, bool)
        st = STYLE[k]
        kw = dict(histtype="step", lw=2, color=st["color"], ls=st["ls"], label=label)
        ax1.hist(y[m], bins=edges, **kw)
        ax2.hist(y[m], bins=edges, weights=w[m], **kw)
    ax1.set_ylabel("training events / 0.1 GeV")
    ax2.set_ylabel("weighted events / 0.1 GeV")
    ax2.set_xlabel(f"true {t0} [GeV] (shown 0–10; weights use the full range)")
    for ax in (ax1, ax2):
        style_axes(ax)
    kwargs = cfg.get("weight", {}).get("kwargs", {})
    ax1.set_title("before flattening", fontsize=10)
    ax2.set_title(f"after {cfg.get('weight', {}).get('name', 'no weights')} {kwargs}  "
                  f"(mean event weight {w.mean():.2f})", fontsize=10)
    handles, labels = ax1.get_legend_handles_labels()
    fig.suptitle(f"{t0} spectrum of the training rows, before and after flattening", color=INK)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(out, f"flattening_{t0}.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.join(out, f'flattening_{t0}.png')}")



def conditional_maps(true, pred, samples, out):
    """Unweighted held-out maps; denominator includes vertical overflow in each slice."""
    import csv
    xedges = np.linspace(0, 10, 41)
    specs = [("pred_vs_true_trueE", pred, np.linspace(0, 10, 81), "predicted neutrino energy [GeV]"),
             ("relerr_vs_true_trueE", (pred - true) / true, np.linspace(-1, 1, 101),
              "relative error (pred - true) / true")]
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
            for i, center in enumerate(centers):
                vi = v[idx == i]
                counts[i] = vi.size
                if vi.size >= MIN_PER_BIN:
                    quantiles[i] = np.quantile(vi, [0.16, 0.5, 0.84])
                records.append(dict(sample=sample, plot=name, true_lo=xedges[i], true_hi=xedges[i+1],
                    n=int(vi.size), visible=int(h[i].sum()), underflow=int((vi < yedges[0]).sum()),
                    overflow=int((vi > yedges[-1]).sum()),
                    q16=quantiles[i, 0], median=quantiles[i, 1], q84=quantiles[i, 2]))
            probability = np.divide(h, counts[:, None], out=np.zeros_like(h), where=counts[:, None] > 0)
            assert np.all(h.sum(axis=1) <= counts)
            fig, ax = plt.subplots(figsize=(8, 5.3))
            image = ax.pcolormesh(xedges, yedges, np.ma.masked_where(probability.T == 0, probability.T),
                                  cmap="viridis", vmin=0, shading="flat")
            fig.colorbar(image, ax=ax, label="fraction of all events in each true-energy slice / y bin")
            ax.plot(centers, quantiles[:, 1], color="#e66101", lw=2, label="median")
            for j in (0, 2):
                ax.plot(centers, quantiles[:, j], color="#e66101", ls="--", lw=1.5,
                        label="q16 / q84" if j == 0 else None)
            if name.startswith("pred_"):
                ax.plot([0, 10], [0, 10], color="black", lw=1, ls=":", label="pred = true")
            else:
                ax.axhline(0, color="black", lw=1, ls=":", label="zero error")
            label = next(label for k, label in SAMPLES if k == sample)
            ax.set(xlabel="true neutrino energy [GeV]", ylabel=ylabel, title=label, xlim=(0, 10), ylim=(yedges[0], yedges[-1]))
            ax.legend(fontsize=8, loc="upper left")
            excluded = int(mask.sum()) - len(x)
            outside = int(((x < 0) | (x > 10)).sum())
            vertical = int(((x >= 0) & (x <= 10) & ((v < yedges[0]) | (v > yedges[-1]))).sum())
            fig.text(0.08, 0.015, f"N={int(mask.sum()):,}; invalid={excluded:,}; trueE outside view={outside:,}; "
                     f"y outside view={vertical:,}. Quantiles use all y; require N >= {MIN_PER_BIN}.", fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, 1))
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(out, f"sample{sample}_{name}.{ext}"), dpi=170)
            plt.close(fig)
            np.savez_compressed(os.path.join(out, f"sample{sample}_{name}_counts.npz"),
                counts=h, slice_counts=counts, conditional_probability=probability,
                xedges=xedges, yedges=yedges, quantiles=quantiles,
                n_total=int(mask.sum()), n_invalid=excluded, n_x_outside=outside, n_y_outside=vertical)
    with open(os.path.join(out, "slice_statistics.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the run's JSON config")
    p.add_argument("--model-dir", help="default: newest save_path/model_*/ with a result.csv")
    p.add_argument("--code-dir", default="/data/nova/chow/transformerEE/code", help="transformer_EE checkout")
    p.add_argument("--flattening-only", metavar="OUT_DIR",
                   help="only make the before/after flattening plot into OUT_DIR (no trained model needed)")
    p.add_argument("--out-dir", help="fresh output directory; refuses an existing directory")
    p.add_argument("--conditional-maps-only", action="store_true",
                   help="only neutrino-energy conditional maps for samples 1/2/3, after alignment checks")
    a = p.parse_args()

    cfg = json.load(open(a.config))
    sc = cfg.get("split_column")
    if not sc:
        sys.exit("config has no split_column; per-sample matching needs the split file")
    targets = cfg["target"]
    if a.flattening_only:
        os.makedirs(a.flattening_only, exist_ok=True)
        plot_flattening(cfg, a.code_dir, a.flattening_only)
        return
    model_dir = a.model_dir or find_model_dir(cfg["save_path"])
    out = a.out_dir or os.path.join(model_dir, "eval_by_sample")
    if a.conditional_maps_only and not a.out_dir:
        p.error("--conditional-maps-only requires a fresh --out-dir")
    if a.out_dir and os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")

    res = pl.read_csv(os.path.join(model_dir, "result.csv"))
    test = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(sc) == "test")
            .select(["sample"] + targets)
            .collect())
    if test.height != res.height:
        sys.exit(f"row mismatch: {res.height} predictions vs {test.height} test rows in {cfg['data_path']}")
    smp = test["sample"].to_numpy().astype(str)
    true = {t: test[t].cast(pl.Float64).to_numpy() for t in targets}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in targets}
    for t in targets:  # result.csv stores float32 truths: they must match the file row by row
        rt = res[f"true_{t}"].to_numpy().astype(np.float64)
        bad = int(np.sum(~np.isclose(rt, true[t], rtol=1e-5, atol=1e-6)))
        if bad:
            sys.exit(f"{bad} rows where result.csv true_{t} != test-row {t}: predictions are not aligned")
    print(f"aligned {res.height} test predictions with {cfg['data_path']}")
    if not set(smp) <= {"1", "2", "3"}:
        sys.exit("unexpected sample labels")
    os.makedirs(out, exist_ok=not bool(a.out_dir))
    if a.conditional_maps_only:
        if not np.all(np.isfinite(true["trueE"]) & (true["trueE"] > 0)):
            sys.exit("nonpositive or nonfinite trueE; cannot compute relative error")
        conditional_maps(true["trueE"], pred["trueE"], smp, out)
        def digest(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        paths = [__file__, a.config, os.path.join(model_dir, "input.json"),
                 os.path.join(model_dir, "result.csv"), os.path.join(model_dir, "best_model.zip")]
        provenance = dict(model_dir=model_dir, data_path=cfg["data_path"],
                          split_column=sc, category_counts={k: int((smp == k).sum()) for k in ("1", "2", "3")},
                          aligned_targets=targets, hashes={os.path.abspath(p): digest(p) for p in paths},
                          config=cfg, checkpoint_metadata=json.load(open(os.path.join(model_dir, "checkpoint_metadata.json"))))
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)
        print(f"wrote conditional maps to {out}")
        return
    plot_flattening(cfg, a.code_dir, out)

    in_win = (true[targets[0]] >= WINDOW[0]) & (true[targets[0]] <= WINDOW[1])
    masks = {k: (smp == k) if k != "all" else np.ones(smp.size, bool) for k, _ in SAMPLES}

    rows = []
    for t in targets:
        r = (pred[t] - true[t]) / true[t]
        for k, label in SAMPLES:
            m = masks[k]
            for win, wm in (("all E", np.ones_like(m)), (f"trueE {WINDOW[0]}-{WINDOW[1]} GeV", in_win)):
                s = stats(r[m & wm])
                rows.append(dict(target=t, sample=k, window=win, n_outside_window=int(np.sum(m & ~in_win)), **s))
    summary = pl.DataFrame(rows)
    summary.write_csv(os.path.join(out, "summary.csv"))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"# Per-sample evaluation\n\nmodel dir: `{model_dir}`  \ndata: `{cfg['data_path']}`  \n"
                f"r = (pred - true)/true; bias = median(r); resolution = (q84 - q16)/2\n\n")
        f.write("| target | sample | window | N | bias | resolution | RMS | abs(r)<5% | <10% | <15% |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for x in rows:
            f.write(f"| {x['target']} | {x['sample']} | {x['window']} | {x['n']} | {x['bias']:+.2%} | {x['res']:.2%} | "
                    f"{x['rms']:.2%} | {x['f5']:.1%} | {x['f10']:.1%} | {x['f15']:.1%} |\n")
        f.write("\nEvents outside the trueE window per sample: " +
                ", ".join(f"{k}: {int(np.sum(masks[k] & ~in_win))}" for k, _ in SAMPLES) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK})
    for t in targets:
        r = (pred[t] - true[t]) / true[t]
        unit = "GeV"

        # 1) relative-error distributions
        fig, ax = plt.subplots(figsize=(7, 4.5))
        edges = np.linspace(-1, 1, 101)
        for k, label in SAMPLES:
            st = STYLE[k]
            ax.hist(r[masks[k] & in_win], bins=edges, density=True, histtype="step", lw=2, color=st["color"],
                    ls=st["ls"], label=f"{label}")
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(f"(pred − true) / true, {t}")
        ax.set_ylabel("density (each curve normalised to 1)")
        ax.set_title(f"{t}: relative error, trueE {WINDOW[0]}–{WINDOW[1]} GeV")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"relerr_{t}.png"), dpi=150)
        plt.close(fig)

        # 2) bias and resolution vs true value (two panels, one y-scale each)
        hi = 6.0 if t == targets[0] else 5.0
        bins = np.arange(0.0, hi + 1e-9, 0.25)
        centers = 0.5 * (bins[1:] + bins[:-1])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
        for k, label in SAMPLES:
            st = STYLE[k]
            idx = np.digitize(true[t], bins) - 1
            b, s_ = [], []
            for i in range(len(centers)):
                ri = r[masks[k] & (idx == i)]
                if ri.size >= MIN_PER_BIN:
                    q16, q50, q84 = np.quantile(ri, [0.16, 0.5, 0.84])
                    b.append(q50); s_.append((q84 - q16) / 2)
                else:
                    b.append(np.nan); s_.append(np.nan)
            kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], lw=2, ms=5, label=label)
            ax1.plot(centers, np.array(b) * 100, **kw)
            ax2.plot(centers, np.array(s_) * 100, **kw)
        ax1.axhline(0, color=MUTED, lw=1)
        ax1.set_ylabel("bias = median r  [%]")
        ax2.set_ylabel("resolution = (q84 − q16)/2  [%]")
        ax2.set_xlabel(f"true {t} [{unit}]")
        for ax in (ax1, ax2):
            style_axes(ax)
        handles, labels = ax1.get_legend_handles_labels()
        fig.suptitle(f"{t}: bias and resolution vs true value (bins with ≥{MIN_PER_BIN} events)", color=INK)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(os.path.join(out, f"vs_true_{t}.png"), dpi=150)
        plt.close(fig)

        # 3) per-sample summary in the window (two panels)
        sub = [x for x in rows if x["target"] == t and x["window"].startswith("trueE")]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
        xs = np.arange(len(SAMPLES))
        for i, (k, label) in enumerate(SAMPLES):
            x = next(y for y in sub if y["sample"] == k)
            st = STYLE[k]
            ax1.plot(i, x["bias"] * 100, marker=st["marker"], color=st["color"], ms=9, ls="none")
            ax2.plot(i, x["res"] * 100, marker=st["marker"], color=st["color"], ms=9, ls="none")
            ax1.annotate(f"{x['bias']:+.1%}", (i, x["bias"] * 100), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color=INK)
            ax2.annotate(f"{x['res']:.1%}", (i, x["res"] * 100), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color=INK)
        ax1.axhline(0, color=MUTED, lw=1)
        for ax, yl in ((ax1, "bias [%]"), (ax2, "resolution [%]")):
            ax.set_xticks(xs, ["S1", "S2", "S3", "All"])
            ax.set_ylabel(yl)
            ax.set_xlim(-0.5, len(SAMPLES) - 0.5)
            style_axes(ax)
        lo, hi2 = ax2.get_ylim()
        ax2.set_ylim(0, hi2 * 1.15)
        lo, hi1 = ax1.get_ylim()
        ax1.set_ylim(lo - 0.15 * (hi1 - lo), hi1 + 0.15 * (hi1 - lo))
        fig.suptitle(f"{t}: per-sample bias and resolution, trueE {WINDOW[0]}–{WINDOW[1]} GeV", color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"summary_{t}.png"), dpi=150)
        plt.close(fig)

    print(f"wrote {out}")
    print(open(os.path.join(out, "summary.md")).read())


if __name__ == "__main__":
    main()
