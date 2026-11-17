#!/usr/bin/env python
"""PROPOSAL (not in the repo): peak measures version 2 for one finished run. v1 fields are kept unchanged.

    python -I peak_metrics_v2.py RUN_DIR --out FILE.json

Per sample, on trueE, frozen test rows. rho = ln(pred/true) clipped to [-1, 1]; r = pred/true - 1.
Windows: low 1.0-1.5 GeV, peak 1.75-2.25 GeV, high 3.0-4.0 GeV. Units: points (100 x).

  level_peak            mean rho in the peak window. The run's overall offset there; NOT pull.
  low_vs_peak           mean rho (low) - mean rho (peak).  + = low-energy events estimated higher than peak events
  peak_vs_high          mean rho (peak) - mean rho (high). + = high-energy events estimated lower than peak events
                        Both: + = towards the peak. Never added together. A global scale shift cancels in each.
  low_vs_peak_median, peak_vs_high_median
                        the same with median r. Their sum is exactly v1 peak_pull. mean minus median = what the
                        tail adds.
  net_towards_low, net_towards_high
                        after subtracting the median rho of the peak window from every event: share of events
                        beyond sigma towards the peak minus share beyond sigma away (sigma = the sample's frozen
                        r05 resolution).
  uncertain_vs_certain  {low, peak, high}: mean rho of the most uncertain third of the window's events minus that
                        of the least uncertain third, natural sign (+ = uncertain events estimated higher).
                        Needs the frozen per-event uncertainty score of the test set (spread of ln pred over the
                        four r07 trainings); null without it. Compare runs window by window: a difference
                        between two runs is the part of the bias that scales with event uncertainty, which is
                        what a training prior produces. Not 0 for a pull-free model (see proposal note).
  curve                 per 0.25 GeV bin, 1-5 GeV: n, mean rho, q16/q50/q84 of r, net fraction as above (signed
                        towards 2 GeV: + over below 2 GeV, + under above).
  peak_pull, pull_slope, bias_curve
                        version 1, unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np

LOW, PEAK, HIGH = (1.0, 1.5), (1.75, 2.25), (3.0, 4.0)
EDGES = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
CLIP = 1.0
VERSION = 2


def _in(t, w):
    return (t >= w[0]) & (t < w[1])


def _rho(t, p):
    return np.clip(np.log(np.maximum(p, 1e-6) / t), -CLIP, CLIP)


def _net(x, sig):
    return 100 * float((x > sig).mean() - (x < -sig).mean())


def _third_gap(x, score):
    a, b = np.quantile(score, [1 / 3, 2 / 3])
    return 100 * float(x[score > b].mean() - x[score < a].mean())


def peak_measures_v2(t, p, sigma, score=None, v1=None):
    """t, p: true and predicted trueE of ONE sample. sigma: that sample's frozen resolution (fraction).
    score: optional per-event uncertainty score aligned with t. v1: optional dict of version-1 fields to carry."""
    x, r = _rho(t, p), p / t - 1
    lo, pk, hi = _in(t, LOW), _in(t, PEAK), _in(t, HIGH)
    if min(lo.sum(), pk.sum(), hi.sum()) < 50:
        raise ValueError("fewer than 50 events in a window")
    c = float(np.median(x[pk]))
    out = dict(v1 or {})
    out.update({
        "level_peak": round(100 * float(x[pk].mean()), 3),
        "low_vs_peak": round(100 * float(x[lo].mean() - x[pk].mean()), 3),
        "peak_vs_high": round(100 * float(x[pk].mean() - x[hi].mean()), 3),
        "low_vs_peak_median": round(100 * float(np.median(r[lo]) - np.median(r[pk])), 3),
        "peak_vs_high_median": round(100 * float(np.median(r[pk]) - np.median(r[hi])), 3),
        "net_towards_low": round(_net(x[lo] - c, sigma), 3),
        "net_towards_high": round(_net(-(x[hi] - c), sigma), 3),
        "sigma": sigma,
        "n_low": int(lo.sum()), "n_peak": int(pk.sum()), "n_high": int(hi.sum()),
        "uncertain_vs_certain": None if score is None else {
            k: round(_third_gap(x[m], score[m]), 3) for k, m in (("low", lo), ("peak", pk), ("high", hi))},
    })
    curve = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = _in(t, (a, b))
        row = {"lo": float(a), "hi": float(b), "n": int(m.sum())}
        if m.sum() >= 20:
            q = np.percentile(r[m], [16, 50, 84]) * 100
            s = 1.0 if 0.5 * (a + b) < 2.0 else -1.0
            row.update(mean_ln_pct=round(100 * float(x[m].mean()), 3), q16_r_pct=round(float(q[0]), 3),
                       q50_r_pct=round(float(q[1]), 3), q84_r_pct=round(float(q[2]), 3),
                       net_towards_pct=round(_net(s * (x[m] - c), sigma), 3))
        curve.append(row)
    out["curve"] = curve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True, help="in the repo version: RUN_DIR/scorecard/peak_metrics.json")
    ap.add_argument("--frozen", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "peak_metrics_frozen.json"))
    ap.add_argument("--eval-dir", default="/data/nova/chow/transformerEE-pcnumu-loop/pcnumu/eval")
    a = ap.parse_args()
    sys.path.insert(0, a.eval_dir)
    import peak_metrics as v1  # noqa: E402
    import scorecard as sc  # noqa: E402

    card = json.load(open(os.path.join(a.run_dir, "scorecard", "scorecard.json")))
    model_dir = card["model_dir"]
    cfg = json.load(open(os.path.join(model_dir, "input.json")))
    smp, true, pred, fp = sc.load_test_predictions(cfg, model_dir)
    if fp != card["test_set"]["fingerprint_sha256"]:
        sys.exit("test-set fingerprint differs from the scorecard's")
    frozen = json.load(open(a.frozen)).get(fp)
    if frozen is None:
        sys.exit("no frozen sigma / uncertainty score for this test set")
    score = None
    if frozen.get("score_file"):
        z = np.load(os.path.join(os.path.dirname(a.frozen), frozen["score_file"]))
        if str(z["fingerprint"]) != fp or z["score"].size != smp.size:
            sys.exit("uncertainty score file does not belong to this test set")
        score = z["score"]
    out = {"version": VERSION, "target": "trueE", "definitions": __doc__.split("\n\n")[2].strip(),
           "low_bin_gev": list(LOW), "peak_bin_gev": list(PEAK), "high_bin_gev": list(HIGH),
           "test_fingerprint": fp, "model_dir": model_dir, "by_sample": {}}
    t_all, p_all = true["trueE"], pred["trueE"]
    for s in ["1", "2", "3"]:
        m = smp == s
        out["by_sample"][s] = peak_measures_v2(t_all[m], p_all[m], frozen["sigma"][s],
                                               None if score is None else score[m], v1.peak_measures(t_all[m], p_all[m]))
    open(a.out, "w").write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for s, v in out["by_sample"].items():
        u = v["uncertain_vs_certain"]
        print(f"S{s}: level {v['level_peak']:+.1f}  low vs peak {v['low_vs_peak']:+.1f}  peak vs high {v['peak_vs_high']:+.1f}  "
              f"(v1 peak_pull {v['peak_pull']:+.2f})" + ("" if u is None else f"  U low/peak/high {u['low']:+.1f}/{u['peak']:+.1f}/{u['high']:+.1f}"))


if __name__ == "__main__":
    main()
