#!/usr/bin/env python
"""Ellipse-fraction calibration scan for one or more runs (issue #4).

    python calibrate_ellipse.py OUT_DIR RUN_DIR [RUN_DIR ...] [--sizes 10:41:1]

For each run and sample, scans circular ellipses (a = b = s %) in the (trueE error %,
trueLepE error %) plane, test set, trueE 0.5-6 GeV, and records the fraction inside with
both definitions the scorecard knows: binned (eval_model.C fractionInsideEllipse port) and
unbinned (each event tested exactly). Writes scan.csv and scan.png into a new OUT_DIR.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_by_sample import SAMPLES, STYLE, WINDOW, find_model_dir, style_axes  # noqa: E402
from scorecard import ellipse_fraction, load_test_predictions  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--sizes", default="10:41:1", help="start:stop:step of the semi-axis in percent")
    a = p.parse_args()
    if os.path.exists(a.out_dir):
        sys.exit(f"refusing to overwrite {a.out_dir}")
    start, stop, step = (float(x) for x in a.sizes.split(":"))
    sizes = np.arange(start, stop, step)

    rows = []
    for run_dir in a.run_dirs:
        model_dir = find_model_dir(run_dir)
        cfg = json.load(open(os.path.join(model_dir, "input.json")))
        smp, true, pred, fingerprint = load_test_predictions(cfg, model_dir)
        t0, t1 = cfg["target"][:2]
        in_win = (true[t0] >= WINDOW[0]) & (true[t0] <= WINDOW[1])
        x = 100 * (pred[t0] - true[t0]) / true[t0]
        y = 100 * (pred[t1] - true[t1]) / true[t1]
        label = cfg.get("model_phys_name", os.path.basename(run_dir.rstrip("/")))
        for k, _ in SAMPLES:
            m = ((smp == k) if k != "all" else np.ones(smp.size, bool)) & in_win
            for s in sizes:
                e = ellipse_fraction(x[m], y[m], s, s)
                rows.append(dict(run=label, test_set=fingerprint[:16], sample=k, semi_axis=float(s),
                                 binned=e["fraction"], unbinned=e["unbinned_fraction"], n=int(m.sum())))
        print(f"scanned {label}")

    os.makedirs(a.out_dir)
    with open(os.path.join(a.out_dir, "scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    runs = list(dict.fromkeys(r["run"] for r in rows))
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4.8), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        ax.axhspan(0.60, 0.70, color="#e66101", alpha=0.15, lw=0, label="target for sample 3 (60–70%)")
        for k, label in SAMPLES:
            sel = [r for r in rows if r["run"] == run and r["sample"] == k]
            st = STYLE[k]
            ax.plot([r["semi_axis"] for r in sel], [r["unbinned"] for r in sel], color=st["color"],
                    ls=st["ls"], lw=2, label=label)
            ax.plot([r["semi_axis"] for r in sel], [r["binned"] for r in sel], color=st["color"],
                    ls=":", lw=1)
        ax.set(xlabel="circle radius a = b  [% error on each target]", title=run.replace("NOvA_PC_Numu_mprod6_1_fd_", ""))
        ax.title.set_fontsize(9)
        style_axes(ax)
    axes[0][0].set_ylabel("fraction of test events inside (solid: unbinned, dotted: binned)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "scan.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
