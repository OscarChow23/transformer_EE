"""Prior-dependence test: same test events, two runs trained on differently weighted spectra.

    python -I pairs.py            # writes pairs_results.json, pairs_curves.npz

For a pair (A, B) and one sample, per event with true 1-5 GeV:
    d_i  = ln(pred_A / pred_B)                       how far the estimate moved when the training prior changed
    dB_i = B_A(true_i) - B_B(true_i)                 how far the toy says it should move (from the two trial
                                                     specs only; B = common.predicted_pull)
  kappa      = slope of d on dB (with an intercept, so a global offset between the runs does not count).
               1 = estimates follow the training prior as much as an ideal estimator of that sample's
               resolution would; 0 = estimates do not depend on the prior.
  shape_move = event-weighted rms over 0.25 GeV bins of (mean d in bin - overall mean d), in points.
               Model-free: how much the bias curve changed shape between the two runs.
  offset     = overall mean d (points): the global shift between the runs (not pull).
Noise: the same three numbers for the 6 pairs of identical r07 trainings (regressed on the same dB).
The rms of those six values is the spread expected for a pair with NO prior change.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

PAIRS = [("r05", "r06"), ("r05", "r08"), ("r05", "r09"), ("r06", "r08"), ("r06", "r09"), ("r08", "r09"),
         ("r05", "r07"), ("r06", "r07"), ("r07", "r08")]


def stats(t, d, dB):
    bm = M.bin_means(t, d)
    n = np.array([((t >= a) & (t < b)).sum() for a, b in zip(M.EDGES[:-1], M.EDGES[1:])])
    ok = np.isfinite(bm)
    off = d.mean()
    out = {"offset": 100 * off, "shape_move": 100 * np.sqrt(np.sum(n[ok] * (bm[ok] - off) ** 2) / n[ok].sum())}
    if dB is not None and np.std(dB) > 1e-6:
        A = np.stack([np.ones_like(dB), dB], 1)
        out["kappa"] = float(np.linalg.lstsq(A, d, rcond=None)[0][1])
    return out, bm


if __name__ == "__main__":
    res, curves = {}, {}
    for horn in C.HORNS:
        d = C.load(horn)
        for s in C.SAMPLES:
            sig = C.SIGMA[horn][s]
            sel = (d["smp"] == s) & (d["t"] >= 1.0) & (d["t"] < 5.0)
            t = d["t"][sel]
            rho = {r: M.rho(t, d[f"p_{r}"][sel]) for r in C.RUNS}
            B = {r: C.predicted_pull(horn, r, s, sig)(t) for r in ("r05", "r06", "r07", "r08", "r09")}
            for a, b in PAIRS:
                dB = B[a] - B[b]
                st, bm = stats(t, rho[a] - rho[b], dB)
                null = [stats(t, rho[x] - rho[y], dB)[0] for x, y in itertools.permutations(C.R07, 2) if x < y]
                st["null_rms"] = {k: float(np.sqrt(np.mean([n[k] ** 2 for n in null]))) for k in null[0] if k != "shape_move"}
                st["null_rms"]["shape_move"] = float(np.mean([n["shape_move"] for n in null]))  # positive by construction: its typical null value
                st["null_max_shape_move"] = float(np.max([n["shape_move"] for n in null]))
                res[f"{horn}_S{s}_{a}-{b}"] = st
                curves[f"{horn}_S{s}_{a}-{b}"] = 100 * bm
                curves[f"{horn}_S{s}_{a}-{b}_toy"] = 100 * M.bin_means(t, dB)
                k = st.get("kappa", float("nan"))
                print(f"{horn} S{s} {a}-{b}: kappa {k:+.2f} (null rms {st['null_rms'].get('kappa', float('nan')):.2f})  "
                      f"shape_move {st['shape_move']:.2f} (null typ {st['null_rms']['shape_move']:.2f}, max {st['null_max_shape_move']:.2f})  "
                      f"offset {st['offset']:+.2f} (null rms {st['null_rms']['offset']:.2f})")
            for x, y in itertools.permutations(C.R07, 2):
                if x < y:
                    curves[f"{horn}_S{s}_{x}-{y}"] = 100 * M.bin_means(t, rho[x] - rho[y])
    json.dump(res, open(os.path.join(C.HERE, "pairs_results.json"), "w"), indent=1)
    np.savez(os.path.join(C.HERE, "pairs_curves.npz"), **curves)
