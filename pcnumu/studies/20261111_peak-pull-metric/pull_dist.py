import json, os, sys
import numpy as np
sys.path.insert(0, "/data/nova/chow/transformerEE-pcnumu-loop/pcnumu/eval")
import scorecard as sc
R = "/data/nova/chow/pcnumu-runs"
TR = {"r05": "r05_v3data-allsamples_v3features", "r06": "r06_flat-alpha001", "r08": "r08_per-sample-flat", "r09": "r09_per-sample-flat-r05shares"}
W = {"low": (1.0, 1.5), "high": (3.0, 4.0)}
for h in ["fhc", "rhc"]:
    for k, t in TR.items():
        card = json.load(open(f"{R}/{t}/{h}/scorecard/scorecard.json"))
        md = card["model_dir"]; cfg = json.load(open(os.path.join(md, "input.json")))
        smp, true, pred, fp = sc.load_test_predictions(cfg, md)
        assert fp == card["test_set"]["fingerprint_sha256"]
        for s in ["1", "3"]:
            m0 = smp == s; T = true["trueE"][m0]; P = pred["trueE"][m0]
            out = []
            for w, (lo, hi) in W.items():
                m = (T >= lo) & (T < hi); r = P[m] / T[m] - 1
                sign = 1 if w == "low" else -1   # toward peak: over below, under above
                rt = sign * r
                q = np.percentile(r, [16, 50, 84]) * 100
                tw = (rt > 0.10).mean() * 100; aw = (rt < -0.10).mean() * 100
                tw20 = (rt > 0.20).mean() * 100; aw20 = (rt < -0.20).mean() * 100
                out.append(f"{w}: n={m.sum():6d} med={q[1]:+5.1f} mean={r.mean()*100:+5.1f} q16={q[0]:+6.1f} q84={q[2]:+6.1f} | >10% toward {tw:4.1f} away {aw:4.1f} | >20% toward {tw20:4.1f} away {aw20:4.1f}")
            print(f"{h} {k} S{s}  " + "\n               ".join(out))
