"""Follow-up to alpha_tradeoff.py: same S3 measures, varying where flattening stops (edge/ease-off) as well as alpha."""
import numpy as np
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz")
    y = d["y"][d["s"] == 3]; sig = ns["SIG"][horn]
    print(f"--- {horn.upper()} S3")
    for hi, tp in ((5, 3), (7, 3), (10, 2), (15, 3)):
        for a in (0.1, 0.05, 0.02, 0.01, 0.001):
            w = ns["persample"](y, 1.3, hi, tp, a); mx, ess = ns["describe"](y, w)
            print(f"flat to {hi:>2} GeV, ease off {tp}  α{a:<6} biggest weight {mx:5.1f}x  ESS {ess:4.1f}%  toy peak pull {ns['toy_peak_pull'](y, w, sig):+5.1f}")
