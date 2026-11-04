#!/usr/bin/env python
"""Has the model learned the training-spectrum peak? Per-sample peak measures for one finished run.

    python -I pcnumu/eval/peak_metrics.py RUN_DIR      # RUN_DIR has scorecard/scorecard.json

A model that learned the ~2 GeV training peak as a prior pulls uncertain events towards it:
events below the peak are overestimated, events above it underestimated. On trueE, per sample,
with r = pred/true - 1 on the frozen test rows (same rows and loader checks as scorecard.py):

  peak_pull  = 100 * (median r for true 1.0-1.5 GeV  -  median r for true 3.0-4.0 GeV)
               0 = no pull; positive = pulled towards the peak (points of relative error)
  pull_slope = slope of log(median pred) vs log(bin centre), 0.25 GeV bins over 1-5 GeV
               1 = no pull; below 1 = estimates squeezed towards the middle
  bias_curve = median r (%) in the same 0.25 GeV bins (supporting evidence, also plotted
               by compare_runs.py)

Adopted 2026-11-04 (findings note of that date). The pull slope also rises if the model
overestimates the tail, so read it together with peak_pull and the bias curve.
Writes RUN_DIR/scorecard/peak_metrics.json (rewritten only if the content changes).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard as sc  # noqa: E402

LOW, HIGH = (1.0, 1.5), (3.0, 4.0)
EDGES = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
VERSION = 1


def median_r(t, p, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.median(p[m] / t[m] - 1.0)) if m.any() else float("nan")


def peak_measures(t, p):
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    meds, bias, ns = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (t >= lo) & (t < hi)
        ns.append(int(m.sum()))
        meds.append(float(np.median(p[m])) if m.any() else float("nan"))
        bias.append(100 * median_r(t, p, lo, hi))
    ok = np.isfinite(meds) & (np.array(ns) >= 50)
    slope = float(np.polyfit(np.log(centres[ok]), np.log(np.array(meds)[ok]), 1)[0]) if ok.sum() >= 3 else None
    return {
        "peak_pull": round(100 * (median_r(t, p, *LOW) - median_r(t, p, *HIGH)), 3),
        "pull_slope": None if slope is None else round(slope, 4),
        "n_low": int(((t >= LOW[0]) & (t < LOW[1])).sum()),
        "n_high": int(((t >= HIGH[0]) & (t < HIGH[1])).sum()),
        "bias_curve": [{"lo": float(a), "hi": float(b), "n": n, "median_r_pct": round(x, 3)}
                       for a, b, n, x in zip(EDGES[:-1], EDGES[1:], ns, bias)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    card = json.load(open(os.path.join(a.run_dir, "scorecard", "scorecard.json")))
    model_dir = card["model_dir"]
    cfg = json.load(open(os.path.join(model_dir, "input.json")))
    smp, true, pred, fp = sc.load_test_predictions(cfg, model_dir)
    if fp != card["test_set"]["fingerprint_sha256"]:
        sys.exit("test-set fingerprint differs from the scorecard's")
    out = {"version": VERSION, "target": "trueE", "definitions": __doc__.split("\n\n")[1].strip(),
           "low_bin_gev": list(LOW), "high_bin_gev": list(HIGH), "test_fingerprint": fp,
           "model_dir": model_dir, "by_sample": {}}
    t_all, p_all = true["trueE"], pred["trueE"]
    for s in ["1", "2", "3"]:
        m = smp == s
        out["by_sample"][s] = peak_measures(t_all[m], p_all[m])
    path = os.path.join(a.run_dir, "scorecard", "peak_metrics.json")
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if not (os.path.exists(path) and open(path).read() == text):
        open(path, "w").write(text)
    for s, v in out["by_sample"].items():
        print(f"S{s}: peak_pull {v['peak_pull']:+.2f} pts, pull_slope {v['pull_slope']}")


if __name__ == "__main__":
    main()
