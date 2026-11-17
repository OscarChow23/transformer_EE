"""Every candidate measure on the real runs (r05-r09), with the r07 run-to-run spread and a bootstrap error.

    python -I real_runs.py        # writes real_metrics.json, real_curves.npz

Noise floor convention as analysis/noise-floor-r07/summary.md: sd (n-1) of r07, r07b, r07c, r07d;
a single-run difference is convincing at 2*sqrt(2)*sd = 2.8 sd.
'boot_se' is the event-sampling error of one run's number (100 bootstraps of the test events); the run-to-run
sd contains it plus the training-to-training part.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

NBOOT = 100


def one(args):
    horn, run, s = args
    sig = C.SIGMA[horn][s]
    d = C.load(horn)
    sel = (d["smp"] == s) & (d["t"] >= 0.5) & (d["t"] < 6.0)
    t, p = d["t"][sel], d[f"p_{run}"][sel]
    B = C.predicted_pull(horn, run, s, sig)
    w = C.weighter(horn, run)(t, np.full(t.size, int(s)))
    val = M.scalar_metrics(t, p, sig, B, w)
    # what the toy predicts for this run's training spectrum (no model involved), same window averages
    lo, hi = (t >= M.LOW[0]) & (t < M.LOW[1]), (t >= M.HIGH[0]) & (t < M.HIGH[1])
    mo = (t >= M.MODE[0]) & (t < M.MODE[1])
    val["pred_low"], val["pred_high"] = 100 * B(t[lo]).mean(), -100 * B(t[hi]).mean()
    val["pred_low_ref"] = val["pred_low"] - 100 * B(t[mo]).mean()
    val["pred_high_ref"] = val["pred_high"] + 100 * B(t[mo]).mean()
    val["res_low"] = 100 * np.diff(np.percentile(p[lo] / t[lo] - 1, [16, 84]))[0] / 2
    val["res_high"] = 100 * np.diff(np.percentile(p[hi] / t[hi] - 1, [16, 84]))[0] / 2
    rng = np.random.default_rng(20261111)
    boots = []
    for _ in range(NBOOT):
        j = rng.integers(0, t.size, t.size)
        boots.append(M.scalar_metrics(t[j], p[j], sig, B, w[j]))
    se = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in boots[0]}
    cur = M.curves(t, p, sig)
    bcur = np.array([100 * B(t[(t >= a) & (t < b)]).mean() for a, b in zip(M.EDGES[:-1], M.EDGES[1:])])
    cal = M.reco_calibration(t, p, w)[1]
    print(horn, run, s, f"v1 {val['v1_peak_pull']:+.2f}", flush=True)
    return (horn, run, s), {k: float(v) for k, v in val.items()}, se, cur, bcur, cal


if __name__ == "__main__":
    jobs = [(h, r, s) for h in C.HORNS for r in C.RUNS for s in C.SAMPLES]
    with mp.Pool(12) as pool:
        res = pool.map(one, jobs)
    out, curves = {}, {}
    for (h, r, s), val, se, cur, bcur, cal in res:
        out[f"{h}_{r}_S{s}"] = {"value": val, "boot_se": se}
        curves[f"{h}_{r}_S{s}"] = cur
        curves[f"{h}_{r}_S{s}_pred"] = bcur
        curves[f"{h}_{r}_S{s}_recocal"] = cal
    noise = {}
    for h in C.HORNS:
        for s in C.SAMPLES:
            vals = [out[f"{h}_{r}_S{s}"]["value"] for r in C.R07]
            noise[f"{h}_S{s}"] = {k: {"mean": float(np.mean([v[k] for v in vals])),
                                      "sd": float(np.std([v[k] for v in vals], ddof=1))} for k in vals[0]}
    json.dump({"runs": out, "r07_noise": noise}, open(os.path.join(C.HERE, "real_metrics.json"), "w"), indent=1)
    np.savez(os.path.join(C.HERE, "real_curves.npz"), **curves)
    print("written real_metrics.json, real_curves.npz")
