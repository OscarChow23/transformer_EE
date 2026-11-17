"""Is the pull larger for uncertain events? Condition on the spread of the four identical r07 trainings.

    python -I ensemble.py         # writes ensemble_results.json

spread_i = sd (n-1) of ln(pred) over r07, r07b, r07c, r07d for test event i (a model-uncertainty proxy).
Within each true-energy window the events are cut into five equal groups by spread. Per group and run:
mean ln(pred/true) and the net fraction beyond the sample's resolution, both signed towards the 2 GeV peak,
and the resolution. For the r07 runs themselves the spread is taken from the OTHER three trainings, so an
event's own error does not enter its own uncertainty (the four leave-one-out results are averaged; their
sd is the noise). Second proxy for comparison: true hadronic fraction trueHadE/trueE (truth, not usable on data).
Also: how far events move between r05 and r08 (ln pred_r05/pred_r08) in the same groups.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

WINDOWS = {"low": (M.LOW, +1), "mode": (M.MODE, +1), "high": (M.HIGH, -1)}
NQ = 5


def group_stats(t, p, sign, sig, grp):
    out = []
    x = M.rho(t, p)
    for g in range(NQ):
        m = grp == g
        xs = sign * x[m]
        out.append({"mean_log": 100 * xs.mean(), "median": 100 * np.median(xs),
                    "nf_log_sig": 100 * ((xs > sig).mean() - (xs < -sig).mean()),
                    "res": 100 * np.diff(np.percentile(x[m], [16, 84]))[0] / 2, "n": int(m.sum())})
    return out


def cut(v):
    q = np.quantile(v, np.linspace(0, 1, NQ + 1)[1:-1])
    return np.searchsorted(q, v)


if __name__ == "__main__":
    res = {}
    for horn in C.HORNS:
        d = C.load(horn)
        f = np.load(os.path.join(C.CACHE, f"feat_{horn}.npz"))
        lp = np.stack([np.log(d[f"p_{r}"]) for r in C.R07])
        spread_all = lp.std(0, ddof=1)
        yhad = f["trueHadE"] / d["t"]
        for s in C.SAMPLES:
            sig = C.SIGMA[horn][s]
            for wn, (w, sign) in WINDOWS.items():
                m = (d["smp"] == s) & (d["t"] >= w[0]) & (d["t"] < w[1])
                t = d["t"][m]
                key = f"{horn}_S{s}_{wn}"
                g = cut(spread_all[m])
                r = {"spread_median_pct": [100 * float(np.median(spread_all[m][g == k])) for k in range(NQ)]}
                for run in C.MAIN:
                    r[run] = group_stats(t, d[f"p_{run}"][m], sign, sig, g)
                loo = []
                for i, run in enumerate(C.R07):
                    others = np.delete(lp[:, m], i, axis=0).std(0, ddof=1)
                    loo.append(group_stats(t, d[f"p_{run}"][m], sign, sig, cut(others)))
                r["r07_loo"] = [{k: float(np.mean([l[q][k] for l in loo])) for k in loo[0][0]} for q in range(NQ)]
                r["r07_loo_sd"] = [{k: float(np.std([l[q][k] for l in loo], ddof=1)) for k in loo[0][0]} for q in range(NQ)]
                dm = sign * np.log(d["p_r05"][m] / d["p_r08"][m])
                r["move_r05_r08"] = [100 * float(dm[g == k].mean()) for k in range(NQ)]
                gy = cut(yhad[m])
                r["yhad_median"] = [float(np.median(yhad[m][gy == k])) for k in range(NQ)]
                for run in C.MAIN:
                    r[run + "_yhad"] = group_stats(t, d[f"p_{run}"][m], sign, sig, gy)
                r["spread_vs_yhad_corr"] = float(np.corrcoef(spread_all[m], yhad[m])[0, 1])
                res[key] = r
                if s != "2":
                    print(f"\n{key}  spread medians % {np.round(r['spread_median_pct'], 1)}  corr(spread, yhad) {r['spread_vs_yhad_corr']:+.2f}")
                    for run in C.MAIN + ["r07_loo"]:
                        print(f"  {run:8s} mean_log " + " ".join(f"{q['mean_log']:+6.1f}" for q in r[run]) +
                              "   nf " + " ".join(f"{q['nf_log_sig']:+6.1f}" for q in r[run]) +
                              "   res " + " ".join(f"{q['res']:5.1f}" for q in r[run]))
                    print("  r07 sd   mean_log " + " ".join(f"{q['mean_log']:6.1f}" for q in r["r07_loo_sd"]))
                    print("  move r05->r08    " + " ".join(f"{v:+6.1f}" for v in r["move_r05_r08"]))
                    for run in ("r05", "r08"):
                        print(f"  {run} by yhad {np.round(r['yhad_median'], 2)} mean_log " + " ".join(f"{q['mean_log']:+6.1f}" for q in r[run + "_yhad"]))
    json.dump(res, open(os.path.join(C.HERE, "ensemble_results.json"), "w"), indent=1)
