"""Markdown tables for report.md from the json outputs.   python -I make_tables.py > tables.md"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

inj = json.load(open(os.path.join(C.HERE, "injection_results.json")))
real = json.load(open(os.path.join(C.HERE, "real_metrics.json")))
ug = json.load(open(os.path.join(C.HERE, "uncertainty_gradient.json")))
pr = json.load(open(os.path.join(C.HERE, "pairs_results.json")))
iu = json.load(open(os.path.join(C.HERE, "injection_u_results.json")))

# metric key, label, injected truth it should follow (key in the injection output)
MET = [("v1_peak_pull", "v1 peak pull (median low − median high)", "inj_v1"),
       ("v1_pull_slope", "v1 pull slope, as 100 × (1 − slope)", "inj_v1"),
       ("nf_r_sig_low", "proposal as written, low: net fraction beyond 1σ", "inj_low_mean"),
       ("nf_r_sig_high", "proposal as written, high", "inj_high_mean"),
       ("nfc_sig_low", "net fraction after removing the level at the peak, low", "inj_low_vs_peak"),
       ("nfc_sig_high", "same, high", "inj_peak_vs_high"),
       ("median_low", "per-side median, low (v1 piece, not referenced)", "inj_low_mean"),
       ("mean_log_low_ref", "**low vs peak** (mean ln ratio)", "inj_low_vs_peak"),
       ("mean_log_high_ref", "**peak vs high** (mean ln ratio)", "inj_peak_vs_high"),
       ("median_low_ref", "low vs peak (median)", "inj_low_vs_peak"),
       ("median_high_ref", "peak vs high (median)", "inj_peak_vs_high"),
       ("tail_asym_low", "tail asymmetry, low", "inj_low_mean"),
       ("shrink_slope", "log-space shrinkage slope", "inj_v1"),
       ("squeeze_low", "response-slope deficit 1–2 GeV", "inj_low_vs_peak"),
       ("width_ratio", "width of predicted / true spectrum (×100)", "inj_v1"),
       ("proj_c", "projection on predicted pull (×100)", "inj_v1")]
SCALE = {"v1_pull_slope": 100, "width_ratio": 100, "proj_c": 100}


def val(cfg, test, k):
    v = inj[cfg][test]["value"]
    if k == "inj_v1":
        return v["inj_low_mean"] + v["inj_high_mean"]
    if k == "v1_pull_slope":
        return 100 * (1 - v[k])
    return v[k] * SCALE.get(k, 1)


def se(cfg, test, k):
    return inj[cfg][test]["se"][k] * abs(SCALE.get(k, 1))


def f(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


print("## T1. Injection tests\n")
for cfg in ("fhc_S3", "rhc_S3", "fhc_S1"):
    print(f"### {cfg.replace('_', ' ').upper()}\n")
    print("Readings in the metric's own units (points). 'a' and 'a2' should be 0. The d/e/f columns show the change "
          "from 'a' (should be 0; **bold** = a false reading larger than 3 bootstrap errors and than 20% of the response to an r05-size pull). "
          "'g − b' is the change when r06-style high-side overestimates are added to an r05-size pull.\n")
    print("| metric | a no pull | a2 real shape | b r05-size pull | rises with pull? | c share of subset pull seen | d1 tilt | d2 leakage | e scale ×1.03 | f resolution ×1.5 | g − b | boot. error |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, lab, truth in MET:
        a, b = val(cfg, "a", k), val(cfg, "bL1.0", k)
        resp = b - a
        serr = se(cfg, "a", k)
        seq = [val(cfg, t, k) for t in ("a", "bL0.25", "bL0.5", "bL1.0", "bL1.5")]
        seqs = [val(cfg, t, k) for t in ("bS0.5", "bS0.75", "bL1.0", "bS1.25", "bS1.5")]
        mono = "yes" if all(np.diff(seq) > 0) and all(np.diff(seqs) > 0) else ("λ only" if all(np.diff(seq) > 0) else "no")
        slope = resp / val(cfg, "bL1.0", truth) if abs(val(cfg, "bL1.0", truth)) > 1e-9 else np.nan
        share = (val(cfg, "c", k) - a) / slope / val(cfg, "c", truth) if np.isfinite(slope) and abs(slope) > 1e-9 else np.nan

        def fp(test):
            dlt = val(cfg, test, k) - a
            bad = abs(dlt) > 3 * serr and abs(dlt) > 0.2 * abs(resp)
            return f"**{f(dlt)}**" if bad else f(dlt)

        share = f"{share:.2f}" if np.isfinite(share) and -0.5 < share < 2 and abs(val(cfg, "c", truth)) > 1 else "–"
        print(f"| {lab} | {f(a)} | {f(val(cfg, 'a2', k))} | {f(b)} | {mono} | {share} | {fp('d1')} | {fp('d2')} | {fp('e')} | {fp('f')} | {f(val(cfg, 'g', k) - b)} | {serr:.2f} |")
    v = inj[cfg]["bL1.0"]["value"]
    vc = inj[cfg]["c"]["value"]
    print(f"\nInjected in b: mean pull low window {v['inj_low_mean']:+.1f}, high window {v['inj_high_mean']:+.1f} (towards 2 GeV); "
          f"low vs peak {v['inj_low_vs_peak']:+.1f}, peak vs high {v['inj_peak_vs_high']:+.1f}. "
          f"In c: mean pull low {vc['inj_low_mean']:+.1f} but median pull only {vc['inj_low_median']:+.1f}.\n")

print("### Uncertainty gradient U (events differ in uncertainty; injection_u.py)\n")
print("| case | test | U low | U peak | U high | U low − U peak | U peak − U high | injected (low, peak, high) |")
print("|---|---|---|---|---|---|---|---|")
NAMES = {"a": "no pull", "b05": "pull, r05 spectrum", "b06": "pull, r06 spectrum", "b08": "pull, r08 spectrum", "d1": "tilt", "d2": "leakage, unrelated to uncertainty",
         "d3": "leakage, 3× more in uncertain events", "e": "scale ×1.03", "f": "resolution ×1.5", "g": "r05 pull + high-side overestimates"}
for cfg in ("fhc_S3", "fhc_S1"):
    for t, n in NAMES.items():
        v, s = iu[f"{cfg}_{t}"]["value"], iu[f"{cfg}_{t}"]["se"]
        print(f"| {cfg.replace('_', ' ').upper()} | {n} | {f(v['U_low'])} ± {s['U_low']:.1f} | {f(v['U_mode'])} ± {s['U_mode']:.1f} | {f(v['U_high'])} ± {s['U_high']:.1f} | "
              f"{f(v['U_low_minus_mode'])} | {f(v['U_mode_minus_high'])} | {f(v['inj_U_low'])}, {f(v['inj_U_mode'])}, {f(v['inj_U_high'])} |")

print("\n## T2. Smallest pull a single-run comparison can show (2.8 × r07 sd ÷ response)\n")
print("Response = change of the metric per point of injected pull (b series). 'Pull' is the quantity in the last column of each block, in points. "
      "For scale, the r05-size injected pull is given.\n")
print("| horn sample | metric | r07 sd | response per point | smallest convincing difference (points of pull) | r05-size pull (points) |")
print("|---|---|---|---|---|---|")
for horn in C.HORNS:
    for s in C.SAMPLES:
        cfg = f"{horn}_S{s}"
        for k, lab, truth in [MET[0], MET[2], MET[4], MET[7], MET[9], MET[8], MET[10]]:
            resp = (val(cfg, "bL1.0", k) - val(cfg, "a", k)) / val(cfg, "bL1.0", truth)
            sd = real["r07_noise"][cfg][k]["sd"] * abs(SCALE.get(k, 1))
            print(f"| {horn.upper()} S{s} | {lab} | {sd:.2f} | {resp:.2f} | {2.83 * sd / abs(resp):.1f} | {val(cfg, 'bL1.0', truth):+.1f} ({truth[4:].replace('_', ' ')}) |")

print("\n## T3. Real runs\n")
ROWS = [("level_peak", "level at the peak: mean ln ratio, 1.75–2.25 GeV (offset, not pull)"),
        ("mean_log_low_ref", "**low vs peak** (mean)"), ("pred_low_ref", "  predicted from training spectrum"),
        ("mean_log_high_ref", "**peak vs high** (mean)"), ("pred_high_ref", "  predicted from training spectrum"),
        ("median_low_ref", "low vs peak (median)"), ("median_high_ref", "peak vs high (median)"),
        ("v1_peak_pull", "v1 peak pull (= the two medians added)"),
        ("nfc_sig_low", "net fraction beyond 1σ, low, peak level removed"), ("nfc_sig_high", "same, high"),
        ("nf_r_sig_low", "proposal as written, low"), ("nf_r_sig_high", "proposal as written, high"),
        ("squeeze_low", "response-slope deficit 1–2 GeV (%)"), ("squeeze_high", "response-slope deficit 2.5–4.5 GeV (%)"),
        ("res_low", "resolution in the low window (%)"), ("res_high", "resolution in the high window (%)")]
for horn in C.HORNS:
    for s in C.SAMPLES:
        cfg = f"{horn}_S{s}"
        print(f"### {horn.upper()} S{s}\n")
        print("| measure (points) | r05 | r06 | r08 | r09 | r07 mean of 4 | r07 sd | 2.8 sd |")
        print("|---|---|---|---|---|---|---|---|")
        for k, lab in ROWS:
            n = real["r07_noise"][cfg][k]
            vals = " | ".join(f(real["runs"][f"{horn}_{r}_S{s}"]["value"][k]) for r in C.MAIN)
            noise = f"{n['sd']:.2f} | {2.83 * n['sd']:.1f}" if not k.startswith("pred") else "– | –"
            print(f"| {lab} | {vals} | {f(n['mean'])} | {noise} |")
        print()

print("## T4. Uncertainty gradient U on the real runs (points; most uncertain third minus least uncertain third)\n")
for horn in C.HORNS:
    for s in ("3", "1"):
        row = ug[f"{horn}_S{s}"]
        print(f"### {horn.upper()} S{s}\n")
        print("| window | r05 | r06 | r08 | r09 | r07 mean of 4 | r07 sd | 2.8 sd |")
        print("|---|---|---|---|---|---|---|---|")
        for k, lab in (("low", "U low (1.0–1.5 GeV)"), ("mode", "U peak (1.75–2.25)"), ("high", "U high (3.0–4.0)"),
                       ("low_minus_mode", "U low − U peak"), ("mode_minus_high", "U peak − U high")):
            print(f"| {lab} | " + " | ".join(f(row[r][k]) for r in C.MAIN) + f" | {f(row['r07_mean'][k])} | {row['r07_sd'][k]:.1f} | {2.83 * row['r07_sd'][k]:.1f} |")
        print()

print("## T5. Prior-dependence test between pairs of runs (S3)\n")
print("| horn | pair | κ (1 = moves as the toy predicts) | κ of identical-retrain pairs (rms) | shape change (points rms) | same for retrain pairs: typical / largest | offset between runs | retrain pairs (rms) |")
print("|---|---|---|---|---|---|---|---|")
for horn in C.HORNS:
    for pair in ("r05-r06", "r05-r08", "r05-r09", "r06-r08", "r08-r09"):
        v = pr[f"{horn}_S3_{pair}"]
        k = f(v["kappa"], 2) if "kappa" in v else "n/a (same spectrum shape)"
        kn = f"{v['null_rms']['kappa']:.2f}" if "kappa" in v else "–"
        print(f"| {horn.upper()} | {pair} | {k} | {kn} | {v['shape_move']:.2f} | {v['null_rms']['shape_move']:.2f} / {v['null_max_shape_move']:.2f} | {f(v['offset'])} | {v['null_rms']['offset']:.1f} |")

print("\n## Model directories used\n\n| run | horn | model_dir |\n|---|---|---|")
print(C.model_dirs_md())
