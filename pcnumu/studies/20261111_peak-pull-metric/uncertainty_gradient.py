"""One number per window for 'pulled when uncertain': the uncertainty gradient.

    python -I uncertainty_gradient.py     # writes uncertainty_gradient.json

Uncertainty score of a test event = sd of ln(pred) over the four identical r07 trainings (as ensemble.py).
In a true-energy window, U = mean ln(pred/true) of the most uncertain third of events minus that of the least
uncertain third (points, natural sign: positive = uncertain events are estimated higher than certain ones).
A prior pulls uncertain events and leaves certain ones, so U is the pull with the run's overall offset and
any bias shared by all events removed. For the r07 runs the score comes from the other three trainings.
Also written: U_low - U_mode and U_mode - U_high (what is specific to each side), and the r07 sd of each.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

WIN = {"low": M.LOW, "mode": M.MODE, "high": M.HIGH}


def U(x, score):
    a, b = np.quantile(score, [1 / 3, 2 / 3])
    return 100 * (x[score > b].mean() - x[score < a].mean())


if __name__ == "__main__":
    out = {}
    for horn in C.HORNS:
        d = C.load(horn)
        lp = np.stack([np.log(d[f"p_{r}"]) for r in C.R07])
        for s in C.SAMPLES:
            row = {}
            for wn, w in WIN.items():
                m = (d["smp"] == s) & (d["t"] >= w[0]) & (d["t"] < w[1])
                t = d["t"][m]
                full = lp[:, m].std(0, ddof=1)
                for run in C.MAIN:
                    row.setdefault(run, {})[wn] = U(M.rho(t, d[f"p_{run}"][m]), full)
                for i, run in enumerate(C.R07):
                    row.setdefault(run, {})[wn] = U(M.rho(t, d[f"p_{run}"][m]), np.delete(lp[:, m], i, 0).std(0, ddof=1))
            for run in row:
                row[run]["low_minus_mode"] = row[run]["low"] - row[run]["mode"]
                row[run]["mode_minus_high"] = row[run]["mode"] - row[run]["high"]
            row["r07_sd"] = {k: float(np.std([row[r][k] for r in C.R07], ddof=1)) for k in row["r05"]}
            row["r07_mean"] = {k: float(np.mean([row[r][k] for r in C.R07])) for k in row["r05"]}
            out[f"{horn}_S{s}"] = row
            print(f"\n{horn} S{s}   U_low  U_mode  U_high | low-mode  mode-high")
            for run in C.MAIN + C.R07 + ["r07_mean", "r07_sd"]:
                v = row[run]
                print(f"  {run:9s} {v['low']:+6.1f} {v['mode']:+6.1f} {v['high']:+6.1f} | {v['low_minus_mode']:+6.1f} {v['mode_minus_high']:+6.1f}")
    json.dump(out, open(os.path.join(C.HERE, "uncertainty_gradient.json"), "w"), indent=1)
