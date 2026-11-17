"""Regional comparison of pc-numu models on the same held-out test events, one horn and sample at a time.

    python pcnumu/eval/regional_comparison.py --horn fhc --sample 3 \
        --out-dir /data/nova/chow/pcnumu-runs/analysis/regional-comparison-v1

Each model directory comes from its trial's saved <trial>/<horn>/scorecard/scorecard.json. Predictions
are aligned to the dataset's test rows by position (result.csv has no event IDs) with the same checks
as scorecard.load_test_predictions, plus cross-model checks; any failure stops the run.

Writes OUT_DIR/<horn>/s<sample>/{summary.json, summary.csv, counts.csv, checks.json}. Refuses to write
into an existing <horn>/s<sample> folder. Plots and tables come from render_regional_comparison.py,
which reads summary.json only.

Definitions (unit event weights; windows on true trueE, lower-inclusive, upper-exclusive):
  r_nu = (pred_trueE - trueE) / trueE, r_mu likewise for trueLepE (ellipse only)
  bias = median(r_nu); resolution = (q84 - q16) / 2, stats() from eval_by_sample.py
  ellipse fraction = mean((100 r_nu / a%)^2 + (100 r_mu / b%)^2 <= 1), frozen axes from ellipse_axes()
  migration: D = pred_trueE in [1, 4); m_S = #(S and D); rate = m_S / N_S; contribution = m_S / N_D
Values are stored as fractions; renderers show them x100 as percent.
Intervals: paired event bootstrap (same resampled events for every model), central 68% of replicates.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_by_sample import stats  # noqa: E402
from scorecard import ID_COLUMNS, code_version, ellipse_axes, ellipse_fraction, load_test_predictions, sha256_file  # noqa: E402

SCHEMA = "pcnumu-regional-comparison/1"
RUNS_ROOT = "/data/nova/chow/pcnumu-runs"
MODELS = [("r05", "r05_v3data-allsamples_v3features"), ("r06", "r06_flat-alpha001"),
          ("r08", "r08_per-sample-flat"), ("r09", "r09_per-sample-flat-r05shares")]
REFERENCE = "r05"
INF = float("inf")
CUTS = dict(
    units="GeV",
    convention="lower-inclusive, upper-exclusive; null upper edge = no upper limit",
    selection_variable="true trueE (source CSV, Float64); reconstructed = pred_trueE",
    resolution_windows=[[1.0, 3.0], [3.0, 6.0], [0.5, 1.0]],  # plot order: the third bar is [0.5, 1)
    bias_windows=[[1.0, 1.5], [1.5, 2.5], [2.5, 4.0]],
    ellipse_windows=[[1.0, 3.0], [3.0, 6.0]],
    migration_sources=[[4.0, 6.0], [6.0, None]],
    overflow_breakdown=[[6.0, 10.0], [10.0, 20.0], [20.0, None]],
    migration_all=[4.0, None],
    reco_destination=[1.0, 4.0],
    legacy_window=dict(lo=0.5, hi=6.0, upper_inclusive=True),
)
LOW_COUNT = 10  # migration numerators below this are flagged as bootstrap-limited
LEGACY_ATOL = 1e-12


def edges(w):
    return float(w[0]), (INF if w[1] is None else float(w[1]))


def wkey(w):
    lo, hi = edges(w)
    return f"{lo:g}-{'inf' if hi == INF else f'{hi:g}'}"


def in_window(x, w):
    lo, hi = edges(w)
    return (x >= lo) & (x < hi)


def stat_windows(cuts):
    out = []
    for w in cuts["resolution_windows"] + cuts["bias_windows"]:
        if w not in out:
            out.append(w)
    return out


def source_windows(cuts):
    return cuts["migration_sources"] + cuts["overflow_breakdown"] + [cuts["migration_all"]]


def model_metrics(t, r, inside, p, cuts):
    """Flat {metric|window: scalar} for one model. t, r, inside, p are aligned event arrays
    (true trueE, r_nu, ellipse membership, pred_trueE). Undefined values are NaN, never 0."""
    m = {}
    for w in stat_windows(cuts):
        s = stats(r[in_window(t, w)])
        k = wkey(w)
        m[f"n|{k}"] = s["n"]
        m[f"bias|{k}"] = float(s["bias"])
        m[f"resolution|{k}"] = float(s["res"])
    for w in cuts["ellipse_windows"]:
        sel = in_window(t, w)
        n, k = int(sel.sum()), wkey(w)
        m[f"ellipse_n|{k}"] = n
        m[f"ellipse_inside|{k}"] = int(inside[sel].sum())
        m[f"ellipse_fraction|{k}"] = float(inside[sel].mean()) if n else np.nan
    dest = in_window(p, cuts["reco_destination"])
    n_d = int(dest.sum())
    dk = wkey(cuts["reco_destination"])
    m[f"N_D|{dk}"] = n_d
    lo_d, hi_d = edges(cuts["reco_destination"])
    for w in source_windows(cuts):
        src, k = in_window(t, w), wkey(w)
        n_s, m_s = int(src.sum()), int((src & dest).sum())
        m[f"N_S|{k}"] = n_s
        m[f"m_S|{k}"] = m_s
        m[f"below_D|{k}"] = int((src & (p < lo_d)).sum())
        m[f"above_D|{k}"] = int((src & (p >= hi_d)).sum())
        m[f"migration_rate|{k}"] = m_s / n_s if n_s else np.nan
        m[f"contribution|{k}"] = m_s / n_d if n_d else np.nan
    return m


def wilson68(k, n):
    """Wilson score interval at z = 1 (about 68%); conditional on n. None if n == 0."""
    if n == 0:
        return None
    z, ph = 1.0, k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def interval(reps):
    reps = np.asarray(reps, float)
    ok = np.isfinite(reps)
    if not ok.any():
        return dict(lo68=None, hi68=None, n_valid_reps=0, n_undefined_reps=int(reps.size))
    lo, hi = np.quantile(reps[ok], [0.16, 0.84])
    return dict(lo68=float(lo), hi68=float(hi), n_valid_reps=int(ok.sum()), n_undefined_reps=int((~ok).sum()))


def jnum(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if np.isfinite(v) else None


# ---------- inputs ----------
def resolve_models(horn, runs_root):
    out = []
    for label, trial in MODELS:
        sc_path = os.path.join(runs_root, trial, horn, "scorecard", "scorecard.json")
        sc = json.load(open(sc_path))
        out.append(dict(label=label, trial=trial, scorecard_path=sc_path, scorecard_sha256=sha256_file(sc_path),
                        scorecard=sc, model_dir=sc["model_dir"]))
    return out


def scan_ids(data_path, split_column):
    test = (pl.scan_csv(data_path, infer_schema_length=0).filter(pl.col(split_column) == "test")
            .select(ID_COLUMNS + ["sample"]).collect())
    n_unique = test.select(ID_COLUMNS).n_unique()
    return test.height, n_unique, test["sample"].to_numpy().astype(str)


def load_all(models, sample):
    """Load and cross-check every model's aligned test predictions; exit on any failure."""
    fail = []
    data_paths = set()
    for mod in models:
        cfg = json.load(open(os.path.join(mod["model_dir"], "input.json")))
        mod["cfg"] = cfg
        data_paths.add(cfg["data_path"])
        if cfg["data_path"] != mod["scorecard"]["data_path"]:
            fail.append(f"{mod['label']}: input.json data_path differs from scorecard data_path")
        if cfg["target"][:2] != ["trueE", "trueLepE"]:
            fail.append(f"{mod['label']}: targets {cfg['target']} are not [trueE, trueLepE, ...]")
        saved = mod["scorecard"]["provenance"]["hashes"]
        for name in ("result.csv", "input.json"):
            h = sha256_file(os.path.join(mod["model_dir"], name))
            if h != saved[name]:
                fail.append(f"{mod['label']}: {name} sha256 differs from the hash its scorecard recorded")
            mod.setdefault("hashes", {})[name] = h
    if len(data_paths) != 1:
        fail.append(f"models use different data_path values: {sorted(data_paths)}")
    if fail:
        sys.exit("input checks failed:\n  " + "\n  ".join(fail))

    ref = None
    for mod in models:
        smp, true, pred, fp = load_test_predictions(mod["cfg"], mod["model_dir"])  # exits on mismatch
        print(f"  {mod['label']}: aligned {smp.size} test rows, fingerprint {fp[:12]}")
        if fp != mod["scorecard"]["test_set"]["fingerprint_sha256"]:
            fail.append(f"{mod['label']}: recomputed fingerprint differs from its scorecard")
        if smp.size != mod["scorecard"]["test_set"]["n"]:
            fail.append(f"{mod['label']}: test row count differs from its scorecard")
        for t in ("trueE", "trueLepE"):
            if not np.all(np.isfinite(pred[t])):
                fail.append(f"{mod['label']}: {int(np.sum(~np.isfinite(pred[t])))} nonfinite pred_{t}")
        if ref is None:
            ref = dict(smp=smp, true=true, fp=fp)
        else:
            if fp != ref["fp"] or not np.array_equal(smp, ref["smp"]):
                fail.append(f"{mod['label']}: test rows differ from {models[0]['label']}")
            for t in ("trueE", "trueLepE"):
                if not np.array_equal(true[t], ref["true"][t]):
                    fail.append(f"{mod['label']}: source truths for {t} differ from {models[0]['label']}")
        mod["pred_all"] = pred
        mod["fingerprint"] = fp
    n_rows, n_unique, id_smp = scan_ids(data_paths.pop(), models[0]["cfg"]["split_column"])
    if n_unique != n_rows:
        fail.append(f"{n_rows - n_unique} duplicate test event IDs ({'/'.join(ID_COLUMNS)})")
    if not np.array_equal(id_smp, ref["smp"]):
        fail.append("sample labels from the ID scan differ from the loader's")
    if sample not in set(ref["smp"]):
        fail.append(f"sample {sample} has no test rows")
    if fail:
        sys.exit("alignment checks failed:\n  " + "\n  ".join(fail))
    return ref, dict(n_rows=n_rows, n_unique_ids=n_unique)


# ---------- checks ----------
def legacy_check(mod, smp_all, true_all, axes, sample):
    """Pilot check 1: this module's window/stats/ellipse path reproduces the saved scorecard for this sample."""
    lw = CUTS["legacy_window"]
    t = true_all["trueE"]
    sel = (smp_all == sample) & (t >= lw["lo"]) & (t <= lw["hi"])
    r = {k: (mod["pred_all"][k] - true_all[k]) / true_all[k] for k in ("trueE", "trueLepE")}
    s = stats(r["trueE"][sel])
    x, y = 100 * r["trueE"][sel], 100 * r["trueLepE"][sel]
    a, b = axes["a_percent"], axes["b_percent"]
    frac = float(np.mean(x * x / (a * a) + y * y / (b * b) <= 1.0))
    via_scorecard = ellipse_fraction(x, y, a, b)["unbinned_fraction"]
    saved_m = mod["scorecard"]["metrics"]["trueE"][sample]
    saved_e = mod["scorecard"]["ellipse"]["by_sample"][sample]["fraction"]
    rows = [("bias", float(s["bias"]), saved_m["bias"]), ("resolution", float(s["res"]), saved_m["resolution"]),
            ("ellipse_fraction", frac, saved_e), ("ellipse_fraction_via_scorecard_fn", via_scorecard, saved_e)]
    detail = {k: dict(recomputed=v, saved=sv, abs_diff=abs(v - sv)) for k, v, sv in rows}
    detail["n"] = dict(recomputed=s["n"], saved=saved_m["n"])
    ok = all(d["abs_diff"] <= LEGACY_ATOL for k, d in detail.items() if k != "n") and s["n"] == saved_m["n"]
    return dict(name=f"1 legacy reproduction {mod['label']}", passed=bool(ok), detail=detail)


def count_checks(point, t, cuts):
    out = []
    dk = wkey(cuts["reco_destination"])
    src_keys = [wkey(w) for w in cuts["migration_sources"]]
    brk_keys = [wkey(w) for w in cuts["overflow_breakdown"]]
    all_k, over_k = wkey(cuts["migration_all"]), wkey(cuts["migration_sources"][1])
    bias_keys = [wkey(w) for w in cuts["bias_windows"]]
    n_1to4 = int(in_window(t, [cuts["bias_windows"][0][0], cuts["bias_windows"][-1][1]]).sum())
    for label, m in point.items():
        bad = []
        for w in source_windows(cuts):
            k = wkey(w)
            if not (0 <= m[f"m_S|{k}"] <= m[f"N_S|{k}"] and m[f"m_S|{k}"] <= m[f"N_D|{dk}"]):
                bad.append(f"bounds {k}")
            if m[f"m_S|{k}"] + m[f"below_D|{k}"] + m[f"above_D|{k}"] != m[f"N_S|{k}"]:
                bad.append(f"below+in+above != N_S for {k}")
        for q in ("m_S", "N_S"):
            if sum(m[f"{q}|{k}"] for k in src_keys) != m[f"{q}|{all_k}"]:
                bad.append(f"sources do not sum to >=4 for {q}")
            if sum(m[f"{q}|{k}"] for k in brk_keys) != m[f"{q}|{over_k}"]:
                bad.append(f"overflow breakdown does not sum for {q}")
        if m[f"N_D|{dk}"] and not np.isclose(sum(m[f"contribution|{k}"] for k in src_keys),
                                             m[f"contribution|{all_k}"], rtol=0, atol=1e-12):
            bad.append("contributions do not add to the >=4 contribution")
        if m[f"N_D|{dk}"] and not np.isclose(sum(m[f"contribution|{k}"] for k in brk_keys),
                                             m[f"contribution|{over_k}"], rtol=0, atol=1e-12):
            bad.append("overflow breakdown contributions do not add")
        if sum(m[f"n|{k}"] for k in bias_keys) != n_1to4:
            bad.append("bias windows do not partition [1, 4)")
        out.append(dict(name=f"3 counts {label}", passed=not bad, detail=bad or "ok"))
    labels = list(point)
    common = [k for k in point[labels[0]] if k.split("|")[0] in ("n", "ellipse_n", "N_S")]
    diff = [k for k in common if any(point[l][k] != point[labels[0]][k] for l in labels[1:])]
    out.append(dict(name="3 true-source and window counts match across models", passed=not diff, detail=diff or "ok"))
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    ap.add_argument("--sample", required=True, choices=["1", "2", "3"])
    ap.add_argument("--out-dir", required=True, help="version directory, e.g. analysis/regional-comparison-v1")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()

    out = os.path.join(a.out_dir, a.horn, f"s{a.sample}")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}; choose a new version directory")

    models = resolve_models(a.horn, a.runs_root)
    print(f"{a.horn} sample {a.sample}: loading {len(models)} models")
    ref, id_scan = load_all(models, a.sample)
    axes = ellipse_axes(ref["fp"])
    if axes["provisional"]:
        sys.exit("ellipse axes for this test set are provisional; refusing to compute ellipse fractions")
    a_pct, b_pct = axes["a_percent"], axes["b_percent"]

    checks = [dict(name="0 alignment and provenance", passed=True,
                   detail=dict(fingerprint=ref["fp"], **id_scan,
                               note="positional alignment; per-row truth match rtol=1e-5 atol=1e-6; "
                                    "result.csv/input.json hashes match each scorecard"))]
    checks += [legacy_check(mod, ref["smp"], ref["true"], axes, a.sample) for mod in models]

    sel = ref["smp"] == a.sample
    t = ref["true"]["trueE"][sel]
    tmu = ref["true"]["trueLepE"][sel]
    ev = {}
    for mod in models:
        p, pmu = mod["pred_all"]["trueE"][sel], mod["pred_all"]["trueLepE"][sel]
        r, rmu = (p - t) / t, (pmu - tmu) / tmu
        x, y = 100 * r, 100 * rmu
        ev[mod["label"]] = dict(p=p, r=r, inside=(x * x / (a_pct * a_pct) + y * y / (b_pct * b_pct) <= 1.0))

    point = {l: model_metrics(t, e["r"], e["inside"], e["p"], CUTS) for l, e in ev.items()}
    checks += count_checks(point, t, CUTS)
    bounds = [0.5, 1, 1.5, 2.5, 3, 4, 6]
    at_bound = {f"{b:g}": int(np.sum(t == b)) for b in bounds}
    checks.append(dict(name="2/4 boundaries and availability", passed=True,
                       detail=dict(events_exactly_at_boundaries=at_bound, n_sample=int(t.size),
                                   true_min=float(t.min()), true_max=float(t.max()),
                                   n_true_ge_4=int(np.sum(t >= 4)), n_true_ge_6=int(np.sum(t >= 6)),
                                   n_true_below_0p5=int(np.sum(t < 0.5)))))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        print(json.dumps(failed, indent=2, default=str))
        sys.exit(f"{len(failed)} pilot checks failed; nothing written")

    # paired event bootstrap: one index draw per replicate, shared by all models
    rng = np.random.default_rng(a.seed)
    keys = list(point[REFERENCE])
    reps = {l: {k: np.empty(a.n_boot) for k in keys} for l in ev}
    for i in range(a.n_boot):
        idx = rng.integers(0, t.size, t.size)
        tb = t[idx]
        for l, e in ev.items():
            mb = model_metrics(tb, e["r"][idx], e["inside"][idx], e["p"][idx], CUTS)
            for k in keys:
                reps[l][k][i] = mb[k]
        if (i + 1) % 100 == 0:
            print(f"  bootstrap {i + 1}/{a.n_boot}")

    results, paired = {}, {}
    dk = wkey(CUTS["reco_destination"])
    for l in ev:
        results[l], paired[l] = {}, {}
        for k in keys:
            v = point[l][k]
            entry = dict(value=jnum(v), **interval(reps[l][k]))
            metric, win = k.split("|")
            if metric in ("migration_rate", "contribution"):
                m_s = point[l][f"m_S|{win}"]
                n = point[l][f"N_S|{win}"] if metric == "migration_rate" else point[l][f"N_D|{dk}"]
                entry.update(numerator=m_s, denominator=n, binomial68_wilson=wilson68(m_s, n),
                             binomial_note=None if metric == "migration_rate" else "conditional on observed N_D",
                             bootstrap_limited=bool(m_s < LOW_COUNT or m_s == n))
            results[l][k] = entry
            if l != REFERENCE and metric in ("bias", "resolution", "ellipse_fraction", "migration_rate", "contribution"):
                d = point[l][k] - point[REFERENCE][k]
                paired[l][k] = dict(value=jnum(d), **interval(reps[l][k] - reps[REFERENCE][k]))

    git = code_version()
    data_path = models[0]["cfg"]["data_path"]
    print("hashing dataset ...")
    summary = dict(
        schema=SCHEMA, created=datetime.datetime.now().isoformat(timespec="seconds"),
        horn=a.horn, sample=a.sample, reference_model=REFERENCE, models=[m["label"] for m in models],
        cuts=CUTS, units="fractions (display x100 as percent)", weights="unit event weights",
        quantile_convention="numpy.quantile default (linear) via eval_by_sample.stats()",
        bootstrap=dict(n_boot=a.n_boot, seed=a.seed, rng="numpy default_rng", method="paired event bootstrap within "
                       "horn/sample: one resample of events per replicate, shared by all models; every statistic, "
                       "including N_D, recomputed per replicate", interval="central 68% of defined replicates",
                       meaning="finite-event uncertainty conditional on the trained models; not seed-to-seed variation",
                       low_count_flag=f"migration numerator < {LOW_COUNT} or equal to its denominator"),
        ellipse=dict(lookup=axes["frozen_as"], a_percent=a_pct, b_percent=b_pct, provisional=False,
                     calibration_population="r05 sample 3, about 66% inside",
                     calibration_window="true trueE in [0.5, 6] GeV inclusive", regional_recalibration=False,
                     definition="unbinned: (100 r_nu / a)^2 + (100 r_mu / b)^2 <= 1, event-level denominator"),
        provenance=dict(code=git, script_sha256=sha256_file(os.path.abspath(__file__)), numpy=np.__version__,
                        dataset=dict(path=data_path, sha256=sha256_file(data_path), bytes=os.path.getsize(data_path)),
                        test_fingerprint_sha256=ref["fp"], id_scan=id_scan,
                        models=[dict(label=m["label"], trial=m["trial"], scorecard=m["scorecard_path"],
                                     scorecard_sha256=m["scorecard_sha256"], model_dir=m["model_dir"],
                                     hashes=m["hashes"], seed=m["cfg"].get("seed"), weight=m["cfg"].get("weight"),
                                     input_noise=m["cfg"].get("noise"), model_phys_name=m["cfg"].get("model_phys_name"))
                                for m in models]),
        results=results, paired_diff_vs_reference=paired, checks=checks,
    )
    os.makedirs(out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        f.write(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    with open(os.path.join(out, "checks.json"), "w") as f:
        f.write(json.dumps(checks, indent=1, default=str) + "\n")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horn", "sample", "model", "metric", "window", "value", "lo68", "hi68", "n_undefined_reps"])
        for l, res in results.items():
            for k, e in res.items():
                w.writerow([a.horn, a.sample, l, *k.split("|"), e["value"], e["lo68"], e["hi68"], e["n_undefined_reps"]])
    with open(os.path.join(out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "source_true_window", "N_S", "m_S_into_reco_" + dk, "below_D", "above_D", "N_D",
                    "migration_rate", "contribution", "bootstrap_limited"])
        for l in results:
            for sw in source_windows(CUTS):
                k = wkey(sw)
                pm = point[l]
                w.writerow([l, k, pm[f"N_S|{k}"], pm[f"m_S|{k}"], pm[f"below_D|{k}"], pm[f"above_D|{k}"],
                            pm[f"N_D|{dk}"], pm[f"migration_rate|{k}"], pm[f"contribution|{k}"],
                            results[l][f"migration_rate|{k}"]["bootstrap_limited"]])
        w.writerow([])
        w.writerow(["window_kind", "true_window", "n_events"])
        for wd in stat_windows(CUTS):
            w.writerow(["bias/resolution", wkey(wd), point[REFERENCE][f"n|{wkey(wd)}"]])
        for wd in CUTS["ellipse_windows"]:
            w.writerow(["ellipse", wkey(wd), point[REFERENCE][f"ellipse_n|{wkey(wd)}"]])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
