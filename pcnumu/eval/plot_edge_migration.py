"""Plot r05 migration into the 0.5-5 GeV reconstructed window, by true energy and sample (findings 2026-11-03).
Reads the CSV written by r05_edge_migration_fine.py (columns horn, sample, true_lo, true_hi, n_test,
leak_frac, share_of_window). Top row: fraction of events in each true-energy bin reconstructed inside the
window. Bottom row: those events as a share of everything of that sample reconstructed in the window.
Usage: python -I pcnumu/eval/plot_edge_migration.py --csv <r05_edge_migration_fine.csv> --out <png>
"""
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES = {1: ("Sample 1 (contained)", "#2a6fdb"),
           2: ("Sample 2 (peripheral, μ contained)", "#e8603c"),
           3: ("Sample 3 (peripheral, μ uncontained)", "#1a9e77")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=50, help="hide bins with fewer test events")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, horn in enumerate(["fhc", "rhc"]):
        for s, (name, col) in SAMPLES.items():
            r = [x for x in rows if x["horn"] == horn and int(x["sample"]) == s and int(x["n_test"]) >= a.min_n]
            lo = [float(x["true_lo"]) for x in r]
            hi = [float(x["true_hi"]) for x in r]
            mid = [(l + h) / 2 for l, h in zip(lo, hi)]
            half = [(h - l) / 2 for l, h in zip(lo, hi)]
            for i, key, scale in [(0, "leak_frac", 100), (1, "share_of_window", 100)]:
                y = [float(x[key]) * scale for x in r]
                ax[i, j].errorbar(mid, y, xerr=half, fmt="o", ms=5, lw=2, color=col, label=name)
        ax[0, j].set_title(f"{horn.upper()} · r05 test set")
        ax[0, j].set_ylabel("% of bin reconstructed in 0.5–5 GeV")
        ax[1, j].set_ylabel("% of the sample's window events")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("true neutrino energy [GeV]")
        for i in range(2):
            ax[i, j].grid(alpha=0.3)
            ax[i, j].axvline(5, color="0.5", ls=":", lw=1)
            ax[i, j].axvline(10, color="0.5", ls=":", lw=1)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Migration into the 0.5–5 GeV window by true energy (bins with ≥ {a.min_n} test events)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
