"""Injection tests for the uncertainty gradient U (uncertainty_gradient.py), where events differ in uncertainty.

    python -I injection_u.py      # writes injection_u_results.json

Each event gets its own sigma_i = k_i x (sample resolution), k_i drawn from {0.5, 0.75, 1, 1.5, 2} with
probabilities (0.25, 0.25, 0.2, 0.2, 0.1), and its ideal toy estimate for that sigma_i (common.toy_predict).
The uncertainty score the metric sees is sigma_i blurred by 30% (an imperfect proxy, like the ensemble spread).
Tests: a no pull; b05 pull under r05's weighted spectrum; b06 under r06's; b08 under r08's; d1 tilt;
d2 leakage-like low tail unrelated to uncertainty; d3 leakage-like low tail that hits uncertain events
3x more often; e scale x1.03; f every sigma x1.5 (no pull); g = b05 + r06-style high-side overestimates.
Standard errors: 100 bootstraps of the events at the real sample size.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import metrics as M  # noqa: E402
from uncertainty_gradient import U, WIN  # noqa: E402

KS, PK = np.array([0.5, 0.75, 1.0, 1.5, 2.0]), [0.25, 0.25, 0.2, 0.2, 0.1]
TESTS = ["a", "b05", "b06", "b08", "d1", "d2", "d3", "e", "f", "g"]


def est(t, z, k, horn, run, s, sig, scale=1.0):
    p = np.empty_like(t)
    for kk in KS:
        m = k == kk
        p[m] = C.toy_predict(t[m], z[m], horn, run, s, kk * scale * sig)
    return p


def make(test, t, horn, s, sig, rng):
    z, k = rng.standard_normal(t.size), rng.choice(KS, t.size, p=PK)
    score = k * np.exp(0.3 * rng.standard_normal(t.size))
    flat = est(t, z, k, horn, None, s, sig)
    if test == "a":
        p = flat
    elif test in ("b05", "b06", "b08"):
        p = est(t, z, k, horn, "r" + test[1:], s, sig)
    elif test == "d1":
        p = flat * (1 - 0.05 * (t - 1) / 3)
    elif test in ("d2", "d3"):
        q = 0.3 * np.clip((t - 1) / 4, 0, 1) * (np.where(k >= 1.5, 2.0, 0.67) if test == "d3" else 1.0)
        p = flat * np.where(rng.random(t.size) < q, rng.uniform(0.6, 0.95, t.size), 1.0)
    elif test == "e":
        p = flat * 1.03
    elif test == "f":
        p = est(t, z, k, horn, None, s, sig, 1.5)
    elif test == "g":
        hit = (t > 2.5) & (rng.random(t.size) < 0.15)
        p = est(t, z, k, horn, "r05", s, sig) * np.where(hit, rng.uniform(1.1, 1.4, t.size), 1.0)
    pull = np.log(p / flat) if test.startswith("b") else np.zeros_like(t)
    return p, score, pull


def measure(t, p, score):
    out = {}
    for wn, w in WIN.items():
        m = (t >= w[0]) & (t < w[1])
        out[f"U_{wn}"] = U(M.rho(t[m], p[m]), score[m])
    out["U_low_minus_mode"], out["U_mode_minus_high"] = out["U_low"] - out["U_mode"], out["U_mode"] - out["U_high"]
    return out


if __name__ == "__main__":
    res = {}
    for horn, s in [("fhc", "3"), ("rhc", "3"), ("fhc", "1")]:
        sig = C.SIGMA[horn][s]
        d = C.load(horn)
        t0 = d["t"][(d["smp"] == s) & (d["t"] >= 1.0) & (d["t"] < 5.0)]
        for i, test in enumerate(TESTS):
            rng = np.random.default_rng([20261112, i, int(s), horn == "rhc"])
            t = np.tile(t0, 4)
            p, score, pull = make(test, t, horn, s, sig, rng)
            val = measure(t, p, score)
            for wn, w in WIN.items():  # injected truth: mean pull of the most minus least uncertain third
                m = (t >= w[0]) & (t < w[1])
                val[f"inj_U_{wn}"] = U(pull[m], score[m])
            boots = []
            for _ in range(100):
                j = rng.integers(0, t0.size, t0.size)
                boots.append(measure(t[j], p[j], score[j]))
            se = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in boots[0]}
            res[f"{horn}_S{s}_{test}"] = {"value": val, "se": se}
            print(f"{horn} S{s} {test:4s} " + "  ".join(f"{k} {val[k]:+5.1f}±{se[k]:.1f}" for k in ("U_low", "U_mode", "U_high")) +
                  "   injected " + " ".join(f"{val['inj_U_' + w]:+5.1f}" for w in WIN))
    json.dump(res, open(os.path.join(C.HERE, "injection_u_results.json"), "w"), indent=1)
