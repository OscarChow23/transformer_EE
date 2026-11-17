"""Injection tests: synthetic predictions with a known answer, on the real test events' true energies.

    python -I injection.py            # ~10 min on 12 cores; writes injection_results.json

Per horn and sample: true energies = the test rows with true 1-5 GeV; sigma = that sample's frozen r05
resolution. An 'input reading' m = true x (1 + sigma z) is turned into a prediction by the toy estimator
(common.bayes_table): median of prior x Gauss x 1/E (the MAPE optimum).
  flat prior            -> no pull (only MAPE's constant low lean)
  r05's weighted prior  -> pull as an ideal estimator trained like r05 would have (strength lambda = 1)
Tests (handoff section 7, task 1):
  a   no pull, symmetric errors (flat prior)
  a2  no pull, realistic error shape: r05's own ln(pred/true) residuals at the mode (1.75-2.25 GeV),
      recentred to median 0 and applied at every energy
  bL  pull strength lambda in {0.25, 0.5, 1, 1.5} at fixed resolution: ln p = ln flat + lambda (ln prior - ln flat)
  bS  pull varied through sigma: ideal estimator at k x sigma, k in {0.5, 0.75, 1.25, 1.5}
  c   pull in a subset: 30% of events have 2 x sigma, 70% have 0.6 x sigma; each gets its ideal estimate
  d1  no pull, tilt: 0 at 1 GeV falling linearly to -5% at 4 GeV
  d2  no pull, leakage-like: a share of events rising from 0 at 1 GeV to 30% at 5 GeV loses 5-40% of the estimate
  e   no pull, all predictions x 1.03
  f   no pull, resolution 1.5 x worse
  g   bL(1) plus the r06 failure: 15% of events above 2.5 GeV overestimated by 10-40%
Each metric: central value on 4 independent copies of the events, standard error from 100 bootstraps of the
events at the real sample size.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

LAMS = [0.25, 0.5, 1.0, 1.5]
SIGK = [0.5, 0.75, 1.25, 1.5]
NBOOT, NCOPY = 100, 4
PRIOR_RUN = "r05"


def make(test, t, sig, horn, s, rng, mode_res):
    """Synthetic prediction and the injected pull per event (natural sign, ln units)."""
    z = rng.standard_normal(t.size)

    def est(run, k=1.0, zz=z):
        return C.toy_predict(t, zz, horn, run, s, k * sig)

    flat = est(None)
    pull = np.log(est(PRIOR_RUN) / flat)
    zero = np.zeros_like(t)
    if test == "a":
        return flat, zero
    if test == "a2":
        return t * np.exp(rng.choice(mode_res, t.size)), zero
    if test.startswith("bL"):
        lam = float(test[2:])
        return flat * np.exp(lam * pull), lam * pull
    if test.startswith("bS"):
        k = float(test[2:])
        return est(PRIOR_RUN, k), np.log(est(PRIOR_RUN, k) / est(None, k))
    if test == "c":
        wide = rng.random(t.size) < 0.3
        p = np.where(wide, est(PRIOR_RUN, 2.0), est(PRIOR_RUN, 0.6))
        f = np.where(wide, est(None, 2.0), est(None, 0.6))
        return p, np.log(p / f)
    if test == "d1":
        return flat * (1 - 0.05 * (t - 1) / 3), zero
    if test == "d2":
        hit = rng.random(t.size) < 0.3 * np.clip((t - 1) / 4, 0, 1)
        return flat * np.where(hit, rng.uniform(0.6, 0.95, t.size), 1.0), zero
    if test == "e":
        return flat * 1.03, zero
    if test == "f":
        return est(None, 1.5), zero
    if test == "g":
        hit = (t > 2.5) & (rng.random(t.size) < 0.15)
        return flat * np.exp(pull) * np.where(hit, rng.uniform(1.1, 1.4, t.size), 1.0), pull
    raise ValueError(test)


TESTS = ["a", "a2"] + [f"bL{x}" for x in LAMS] + [f"bS{x}" for x in SIGK] + ["c", "d1", "d2", "e", "f", "g"]


def one(args):
    horn, s = args
    sig = C.SIGMA[horn][s]
    d = C.load(horn)
    sel = (d["smp"] == s) & (d["t"] >= 1.0) & (d["t"] < 5.0)
    t0 = d["t"][sel]
    p05 = d["p_r05"][sel]
    mm = (t0 >= M.MODE[0]) & (t0 < M.MODE[1])
    mode_res = np.log(p05[mm] / t0[mm])
    mode_res = mode_res - np.median(mode_res)
    B = C.predicted_pull(horn, PRIOR_RUN, s, sig)
    wfun = C.weighter(horn, PRIOR_RUN)
    w0 = wfun(t0, np.full(t0.size, int(s)))
    res = {}
    for i, test in enumerate(TESTS):
        rng = np.random.default_rng([20261111, i, int(s), horn == "rhc"])
        tt, ww = np.tile(t0, NCOPY), np.tile(w0, NCOPY)
        p, pull = make(test, tt, sig, horn, s, rng, mode_res)
        cen = M.scalar_metrics(tt, p, sig, B, ww)
        lo, hi = (tt >= M.LOW[0]) & (tt < M.LOW[1]), (tt >= M.HIGH[0]) & (tt < M.HIGH[1])
        cen["inj_low_mean"], cen["inj_high_mean"] = 100 * pull[lo].mean(), -100 * pull[hi].mean()
        mo = (tt >= M.MODE[0]) & (tt < M.MODE[1])
        cen["inj_low_vs_peak"] = 100 * (pull[lo].mean() - pull[mo].mean())
        cen["inj_peak_vs_high"] = 100 * (pull[mo].mean() - pull[hi].mean())
        cen["inj_low_median"], cen["inj_high_median"] = 100 * np.median(pull[lo]), -100 * np.median(pull[hi])
        cen["res_pct"] = 100 * np.diff(np.percentile(p / tt - 1, [16, 84]))[0] / 2
        n = t0.size
        boots = []
        for _ in range(NBOOT):
            j = rng.integers(0, n, n)
            boots.append(M.scalar_metrics(tt[j], p[j], sig, B, ww[j]))
        se = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in boots[0]}
        res[test] = {"value": {k: float(v) for k, v in cen.items()}, "se": se}
        print(horn, s, test, f"v1 {cen['v1_peak_pull']:+.2f}  mean_low {cen['mean_log_low']:+.2f}  inj {cen['inj_low_mean']:+.2f}", flush=True)
    return (horn, s), res


if __name__ == "__main__":
    jobs = [(h, s) for h in C.HORNS for s in C.SAMPLES]
    with mp.Pool(6) as pool:
        out = dict(pool.map(one, jobs))
    json.dump({f"{h}_S{s}": v for (h, s), v in out.items()}, open(os.path.join(C.HERE, "injection_results.json"), "w"), indent=1)
    print("written injection_results.json")
