import json, sys, numpy as np
sys.path.insert(0, "/data/nova/chow/transformerEE-pcnumu-loop/pcnumu/eval")  # run from the repo root with PYTHONPATH=.
from scorecard import load_test_predictions
runs = {"fhc": "/data/nova/chow/transformerEE/runs/pc-numu/fhc/pcnumu_fhc_r05_v3data-allsamples_v3features_20261029/model_6db92fccf53484d786f0c76dbc5a2ed9",
        "rhc": "/data/nova/chow/transformerEE/runs/pc-numu/rhc/pcnumu_rhc_r05_v3data-allsamples_v3features_20261029/model_6d64aa3fca02cf0641e52337a4d80ca9"}
edges = [0, 0.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 1e9]
for h, md in runs.items():
    cfg = json.load(open(md + "/input.json"))
    smp, true, pred, _ = load_test_predictions(cfg, md)
    t, p = true["trueE"], pred["trueE"]
    print(f"\n===== {h} r05 test set, trueE")
    for k in ["1", "2", "3", "all"]:
        m = (smp == k) if k != "all" else np.ones_like(t, bool)
        print(f"-- sample {k}: N={m.sum()}")
        print(f"{'true bin':>10} {'N':>7} {'med pred/true':>13} {'p16':>6} {'p84':>6} {'frac pred<5':>11} {'frac pred<6':>11}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = m & (t >= lo) & (t < hi)
            if s.sum() < 20: continue
            r = p[s] / t[s]
            print(f"{lo:>4g}-{hi if hi<1e8 else 'inf':<5} {s.sum():>7} {np.median(r):>13.3f} {np.percentile(r,16):>6.3f} {np.percentile(r,84):>6.3f} {np.mean(p[s]<5):>11.3f} {np.mean(p[s]<6):>11.3f}")
        win = m & (p >= 0.5) & (p < 5)
        print(f"   of events PREDICTED in 0.5-5 GeV (N={win.sum()}): true>6: {np.mean(t[win]>6):.4f}, true>10: {np.mean(t[win]>10):.4f}, true 5-6: {np.mean((t[win]>=5)&(t[win]<6)):.4f}")
