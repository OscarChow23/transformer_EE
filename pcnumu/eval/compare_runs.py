#!/usr/bin/env python
"""Compare finished TransformerEE runs on their own test predictions, with eval_by_sample's metrics.

Each --run is LABEL|RESULT_CSV (all rows of an older run's result.csv) or LABEL|CONFIG|SAMPLE
(a split_column run's test rows of one sample, matched to result.csv as eval_by_sample does).
The runs need not share test events: this is a like-for-like metric comparison, not a paired one.
Writes to a fresh --out-dir:
  summary.csv / summary.md      bias, resolution, tail fractions per run, all E and trueE window
  relerr_<target>.png           relative-error distributions, one curve per run
  vs_true_<target>.png          bias and resolution vs true value (two panels, shared x)

  python compare_runs.py --out-dir DIR --run "r04|.../result.csv" --run "r05 S3|r05.json|3"
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_by_sample import INK, MIN_PER_BIN, MUTED, WINDOW, find_model_dir, stats, style_axes  # noqa: E402

TARGETS = ["trueE", "trueLepE"]
# Categorical slots 1-2 of the dataviz reference palette, plus marker/linestyle per run.
STYLES = [dict(color="#2a78d6", marker="o", ls="-"), dict(color="#eb6834", marker="s", ls="--"),
          dict(color="#1baf7a", marker="^", ls="-.")]


def load_run(spec):
    parts = spec.split("|")
    if len(parts) == 2:
        label, path = parts
        res = pl.read_csv(path)
        true = {t: res[f"true_{t}"].to_numpy().astype(np.float64) for t in TARGETS}
        pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in TARGETS}
        return label, path, true, pred
    label, config, sample = parts
    import json
    cfg = json.load(open(config))
    model_dir = find_model_dir(cfg["save_path"])
    res = pl.read_csv(os.path.join(model_dir, "result.csv"))
    test = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(cfg["split_column"]) == "test")
            .select(["sample"] + TARGETS)
            .collect())
    if test.height != res.height:
        sys.exit(f"{label}: {res.height} predictions vs {test.height} test rows")
    for t in TARGETS:
        bad = int(np.sum(~np.isclose(res[f"true_{t}"].to_numpy().astype(np.float64),
                                     test[t].cast(pl.Float64).to_numpy(), rtol=1e-5, atol=1e-6)))
        if bad:
            sys.exit(f"{label}: {bad} rows where result.csv true_{t} != test-row {t}")
    m = test["sample"].to_numpy().astype(str) == sample
    true = {t: test[t].cast(pl.Float64).to_numpy()[m] for t in TARGETS}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64)[m] for t in TARGETS}
    return label, os.path.join(model_dir, "result.csv") + f" (sample {sample})", true, pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="LABEL|RESULT_CSV or LABEL|CONFIG|SAMPLE")
    p.add_argument("--out-dir", required=True, help="fresh output directory; refuses an existing directory")
    a = p.parse_args()
    if os.path.exists(a.out_dir):
        sys.exit(f"refusing to overwrite {a.out_dir}")
    runs = [load_run(s) for s in a.run]
    os.makedirs(a.out_dir)

    rows = []
    for t in TARGETS:
        for label, src, true, pred in runs:
            r = (pred[t] - true[t]) / true[t]
            in_win = (true["trueE"] >= WINDOW[0]) & (true["trueE"] <= WINDOW[1])
            for win, wm in (("all E", np.ones(r.size, bool)), (f"trueE {WINDOW[0]}-{WINDOW[1]} GeV", in_win)):
                rows.append(dict(target=t, run=label, window=win, **stats(r[wm])))
    pl.DataFrame(rows).write_csv(os.path.join(a.out_dir, "summary.csv"))
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write("# Run comparison\n\n" + "".join(f"- {label}: `{src}`\n" for label, src, _, _ in runs) +
                "\nr = (pred - true)/true; bias = median(r); resolution = (q84 - q16)/2. "
                "Each run on its own test events (not paired).\n\n")
        f.write("| target | run | window | N | bias | resolution | RMS | abs(r)<5% | <10% | <15% |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for x in rows:
            f.write(f"| {x['target']} | {x['run']} | {x['window']} | {x['n']} | {x['bias']:+.2%} | {x['res']:.2%} | "
                    f"{x['rms']:.2%} | {x['f5']:.1%} | {x['f10']:.1%} | {x['f15']:.1%} |\n")

    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK})
    for t in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        edges = np.linspace(-1, 1, 101)
        for (label, _, true, pred), st in zip(runs, STYLES):
            r = (pred[t] - true[t]) / true[t]
            in_win = (true["trueE"] >= WINDOW[0]) & (true["trueE"] <= WINDOW[1])
            ax.hist(r[in_win], bins=edges, density=True, histtype="step", lw=2, color=st["color"], ls=st["ls"], label=label)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(f"(pred − true) / true, {t}")
        ax.set_ylabel("density (each curve normalised to 1)")
        ax.set_title(f"{t}: relative error, trueE {WINDOW[0]}–{WINDOW[1]} GeV")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"relerr_{t}.png"), dpi=150)
        plt.close(fig)

        hi = 6.0 if t == "trueE" else 5.0
        bins = np.arange(0.0, hi + 1e-9, 0.25)
        centers = 0.5 * (bins[1:] + bins[:-1])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
        for (label, _, true, pred), st in zip(runs, STYLES):
            r = (pred[t] - true[t]) / true[t]
            idx = np.digitize(true[t], bins) - 1
            b, s_ = [], []
            for i in range(len(centers)):
                ri = r[idx == i]
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
        ax2.set_xlabel(f"true {t} [GeV]")
        for ax in (ax1, ax2):
            style_axes(ax)
        handles, labels = ax1.get_legend_handles_labels()
        fig.suptitle(f"{t}: bias and resolution vs true value (bins with ≥{MIN_PER_BIN} events)", color=INK)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(runs), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(os.path.join(a.out_dir, f"vs_true_{t}.png"), dpi=150)
        plt.close(fig)

    print(f"wrote {a.out_dir}")
    print(open(os.path.join(a.out_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
