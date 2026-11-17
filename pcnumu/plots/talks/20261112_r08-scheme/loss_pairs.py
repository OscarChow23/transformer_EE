"""Training/validation loss per epoch: r05 vs r06 and r08 vs r09, FHC and RHC panels.
Source: each run's scorecard/loss_history.csv (parsed from train.log by scorecard.py); best epoch from scorecard.json.
usage: python -I loss_pairs.py OUT_DIR"""
import csv
import json
import sys

sys.path.insert(0, "/data/nova/chow/talk-plots_20261106/update_20261111/scripts")
import talk12 as t  # noqa: E402  (rootstyle rc + r05-r09 model styles)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

out = sys.argv[1]
RUNS = {"r05": "r05_v3data-allsamples_v3features", "r06": "r06_flat-alpha001",
        "r08": "r08_per-sample-flat", "r09": "r09_per-sample-flat-r05shares"}
BASE = "/data/nova/chow/pcnumu-runs"


def read(r, h):
    d = f"{BASE}/{RUNS[r]}/{h}/scorecard"
    rows = list(csv.DictReader(open(f"{d}/loss_history.csv")))
    best = json.load(open(f"{d}/scorecard.json"))["training"]["best_epoch"]
    ep = [int(x["epoch"]) for x in rows]
    return ep, [float(x["train_loss"]) for x in rows], [float(x["valid_loss"]) for x in rows], best


for pair, note in ((("r05", "r06"), "pooled weights: mean event weight ≈ 0.2 (r05), 0.1 (r06)"),
                   (("r08", "r09"), "per-sample weights: mean event weight 1 (both)")):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, h in zip(axes, ("fhc", "rhc")):
        for r in pair:
            ep, tr, va, best = read(r, h)
            st = t.rs.MODELS[r]
            ax.plot(ep, tr, color=st["color"], ls="--", lw=1.6, label=f"{r} train")
            ax.plot(ep, va, color=st["color"], ls="-", lw=2.2, label=f"{r} validation")
            ax.plot(best, va[ep.index(best)], marker="*", ms=14, color=st["color"], mec="white", mew=0.8, zorder=5)
        t.rs.labels(ax, "Epoch", "Weighted loss" if h == "fhc" else None)
        t.rs.tag(ax, t.rs.HORN[h], loc="upper right")
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].plot([], [], ls="none", marker="*", ms=14, color="#555555", label="best epoch")
    axes[0].legend(loc="center right", fontsize=11, bbox_to_anchor=(1.0, 0.6))
    fig.tight_layout()
    name = f"loss-{pair[0]}-vs-{pair[1]}"
    t.rs.save(fig, out, name,
              f"Training (dashed) and validation (solid) loss per epoch, {pair[0]} vs {pair[1]}, FHC left, RHC right; star = "
              f"best epoch (checkpoint used). Loss = weighted 0.5 MAPE(trueE) + 0.5 MAPE(trueLepE), validation weighted with the "
              f"training weight map. {note}. Absolute loss values depend on the weights, so they are only comparable within a "
              f"run (shape, convergence, overfitting), not between differently weighted runs; the unweighted test scorecard "
              f"compares models. Source: scorecard/loss_history.csv and scorecard.json of each run.")
    print("wrote", name)
