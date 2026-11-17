"""r05 test set: per sample and true-E bin, the leak into the 0.5-5 GeV window, the bin's share of all
events predicted in the window, and median pred/true. r05 = pooled alpha 0.05, so a guide only.
Run from the loop repo root with PYTHONPATH=. ; output -> r05_edge_migration_fine.{txt,csv}"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, "/data/nova/chow/transformerEE-pcnumu-loop/pcnumu/eval")
from scorecard import load_test_predictions
RUNS = {"fhc": "/data/nova/chow/transformerEE/runs/pc-numu/fhc/pcnumu_fhc_r05_v3data-allsamples_v3features_20261029/model_6db92fccf53484d786f0c76dbc5a2ed9",
        "rhc": "/data/nova/chow/transformerEE/runs/pc-numu/rhc/pcnumu_rhc_r05_v3data-allsamples_v3features_20261029/model_6d64aa3fca02cf0641e52337a4d80ca9"}
EDGES = [4, 5, 5.5, 6, 7, 8, 10, 12, 15]
W = (0.5, 5.0)
out = os.path.dirname(os.path.abspath(__file__))
rows = []
for h, md in RUNS.items():
    smp, true, pred, _ = load_test_predictions(json.load(open(md + "/input.json")), md)
    t, p = true["trueE"], pred["trueE"]
    inwin = (p >= W[0]) & (p < W[1])
    for k in ["1", "2", "3"]:
        m = smp == k
        nwin = (m & inwin).sum()
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            b = m & (t >= lo) & (t < hi)
            rows.append({"horn": h, "sample": k, "true_lo": lo, "true_hi": hi, "n_test": int(b.sum()),
                         "leak_frac": (b & inwin).sum() / max(b.sum(), 1),
                         "share_of_window": (b & inwin).sum() / nwin,
                         "median_pred_over_true": float(np.median(p[b] / t[b])) if b.sum() else np.nan})
with open(os.path.join(out, "r05_edge_migration_fine.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(out, "r05_edge_migration_fine.txt"), "w") as f:
    for h in RUNS:
        f.write(f"\n== {h}: leak into 0.5-5 GeV | share of the sample's window events | median pred/true | test events\n")
        f.write(f"{'true E':>9} " + " ".join(f"{'S'+k:>30}" for k in "123") + "\n")
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            cells = []
            for k in "123":
                r = next(r for r in rows if r["horn"] == h and r["sample"] == k and r["true_lo"] == lo)
                cells.append(f"{r['leak_frac']:6.1%} {r['share_of_window']:7.3%} {r['median_pred_over_true']:5.2f} {r['n_test']:7d}")
            f.write(f"{lo:>4g}-{hi:<4g} " + " ".join(f"{c:>30}" for c in cells) + "\n")
print(open(os.path.join(out, "r05_edge_migration_fine.txt")).read())
