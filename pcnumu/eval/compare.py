#!/usr/bin/env python
"""Compare finished TransformerEE models.

    python -I pcnumu/eval/compare.py runs --out-dir DIR --run "r04|.../result.csv" --run "r05 S3|r05.json|3"
    python -I pcnumu/eval/compare.py regional --horn fhc --sample 3 --out-dir ANALYSIS/regional-comparison-vN
    python -I pcnumu/eval/compare.py render ANALYSIS/regional-comparison-vN/fhc/s3 [.../rhc/s3]

runs      like-for-like metrics of any runs, each on its own test events (not paired). Each --run is
          LABEL|RESULT_CSV (all rows of an older run) or LABEL|CONFIG|SAMPLE (a split_column run's test rows
          of one sample). Writes summary.csv/.md, relerr_<t>.png, vs_true_<t>.png and, when all runs share
          the same events, spectrum_<t>.png.
regional  r05/r06/r08/r09 on the same held-out events of one horn and sample: regional bias/resolution,
          ellipse fraction, migration into reco [1, 4) GeV, paired-bootstrap 68% intervals and pilot checks
          (definitions: pcnumu/core/regional.py). Model folders come from each trial's saved scorecard.
          Writes OUT/<horn>/s<sample>/{summary.json, summary.csv, counts.csv, checks.json}; refuses to
          overwrite. Nothing is written if a check fails.
render    plots and comparison.md from regional summary.json files only (no metric is computed). Pass the
          FHC and RHC folders of one sample together for shared y scales. Derived files are regenerated.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pcnumu.core import data, plots, regional as rg, report, style  # noqa: E402
from pcnumu.core.metrics import (WINDOW, ellipse_axes, ellipse_fraction, in_headline_window,  # noqa: E402
                                 inside_ellipse, relative_error, stats)

RUNS_ROOT = "/data/nova/chow/pcnumu-runs"
MODELS = [("r05", "r05_v3data-allsamples_v3features"), ("r06", "r06_flat-alpha001"),
          ("r08", "r08_per-sample-flat"), ("r09", "r09_per-sample-flat-r05shares")]
REFERENCE = "r05"
REGIONAL_SCHEMA = "pcnumu-regional-comparison/1"
TARGETS = ["trueE", "trueLepE"]


# ======================= runs =======================
def load_run(spec):
    parts = spec.split("|")
    if len(parts) == 2:
        label, path = parts
        true, pred = data.load_result_csv(path, TARGETS)
        return label, path, true, pred
    label, config, sample = parts
    cfg = json.load(open(config))
    model_dir = data.find_model_dir(cfg["save_path"])
    smp, true, pred, _ = data.load_test_predictions(cfg, model_dir)
    m = smp == sample
    return (label, os.path.join(model_dir, "result.csv") + f" (sample {sample})",
            {t: true[t][m] for t in TARGETS}, {t: pred[t][m] for t in TARGETS})


def cmd_runs(a):
    if os.path.exists(a.out_dir):
        sys.exit(f"refusing to overwrite {a.out_dir}")
    runs = [load_run(s) for s in a.run]
    os.makedirs(a.out_dir)
    rows = []
    for t in TARGETS:
        for label, _, true, pred in runs:
            r = relative_error(pred[t], true[t])
            in_win = in_headline_window(true["trueE"])
            for win, wm in (("all E", np.ones(r.size, bool)), (f"trueE {WINDOW[0]}-{WINDOW[1]} GeV", in_win)):
                rows.append(dict(target=t, run=label, window=win, **stats(r[wm])))
    pl.DataFrame(rows).write_csv(os.path.join(a.out_dir, "summary.csv"))
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write("# Run comparison\n\n" + "".join(f"- {label}: `{src}`\n" for label, src, _, _ in runs) +
                "\nr = (pred - true)/true; bias = median(r); resolution = (q84 - q16)/2. "
                "Each run on its own test events (not paired).\n\n")
        f.write(report.stats_table(rows, "run"))

    style.use()
    sty = style.SERIES
    for t in TARGETS:
        series = [(label, st, relative_error(pred[t], true[t])[in_headline_window(true["trueE"])])
                  for (label, _, true, pred), st in zip(runs, sty)]
        fig = plots.relerr_hist(series, f"{t}: (pred − true) / true", tag=f"trueE {WINDOW[0]:g}–{WINDOW[1]:g} GeV")
        style.save(fig, a.out_dir, f"relerr_{t}", formats=("png",))
        hi = 6.0 if t == "trueE" else 5.0
        fig, _ = plots.bias_resolution_panels(
            [(label, st, true[t], relative_error(pred[t], true[t])) for (label, _, true, pred), st in zip(runs, sty)],
            np.arange(0.0, hi + 1e-9, 0.25), f"True {t} (GeV)", tag=t)
        style.save(fig, a.out_dir, f"vs_true_{t}", formats=("png",))
        # Shapes, not just errors: does the predicted spectrum reproduce the true one?
        if all(np.array_equal(runs[0][2][t], tr[t]) for _, _, tr, _ in runs):
            bins = np.arange(0.0, hi + 1e-9, 0.1)
            fig, (ax1, ax2) = style.figure(nrows=2, size=(6.4, 3.0), sharex=True,
                                           gridspec_kw=dict(height_ratios=(3, 1), hspace=0))
            n_true = np.histogram(runs[0][2][t], bins)[0]
            ax1.stairs(n_true, bins, fill=True, color="#d0d0d0", label="True")
            for (label, _, _, pred), st in zip(runs, sty):
                n = np.histogram(pred[t], bins)[0]
                ax1.stairs(n, bins, color=st["color"], ls=st["ls"], lw=1.6, label=f"Reco {label}")
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax2.stairs(n / n_true, bins, color=st["color"], ls=st["ls"], lw=1.6)
            ax2.axhline(1, color=style.MUTED, lw=1, ls=":")
            ax2.set_ylim(0.5, 1.5)
            style.labels(ax1, None, "Test events / 0.1 GeV")
            style.labels(ax2, f"{t} (GeV)", "Reco / true")
            ax1.legend(loc="upper right", fontsize=10)
            style.save(fig, a.out_dir, f"spectrum_{t}", formats=("png",))
    print(f"wrote {a.out_dir}")
    print(open(os.path.join(a.out_dir, "summary.md")).read())


# ======================= regional =======================
def resolve_models(horn, runs_root):
    out = []
    for label, trial in MODELS:
        sc_path = os.path.join(runs_root, trial, horn, "scorecard", "scorecard.json")
        sc = json.load(open(sc_path))
        out.append(dict(label=label, trial=trial, scorecard_path=sc_path, scorecard_sha256=data.sha256_file(sc_path),
                        scorecard=sc, model_dir=sc["model_dir"]))
    return out


def load_all(models, sample):
    """Load and cross-check every model's aligned test predictions; exit on any failure."""
    fail, data_paths = [], set()
    for mod in models:
        cfg = json.load(open(os.path.join(mod["model_dir"], "input.json")))
        mod["cfg"] = cfg
        data_paths.add(cfg["data_path"])
        if cfg["data_path"] != mod["scorecard"]["data_path"]:
            fail.append(f"{mod['label']}: input.json data_path differs from scorecard data_path")
        if cfg["target"][:2] != TARGETS:
            fail.append(f"{mod['label']}: targets {cfg['target']} are not [trueE, trueLepE, ...]")
        saved = mod["scorecard"]["provenance"]["hashes"]
        mod["hashes"] = {}
        for name in ("result.csv", "input.json"):
            h = data.sha256_file(os.path.join(mod["model_dir"], name))
            if h != saved[name]:
                fail.append(f"{mod['label']}: {name} sha256 differs from the hash its scorecard recorded")
            mod["hashes"][name] = h
    if len(data_paths) != 1:
        fail.append(f"models use different data_path values: {sorted(data_paths)}")
    if fail:
        sys.exit("input checks failed:\n  " + "\n  ".join(fail))

    ref = None
    for mod in models:
        smp, true, pred, fp = data.load_test_predictions(mod["cfg"], mod["model_dir"])  # exits on mismatch
        print(f"  {mod['label']}: aligned {smp.size} test rows, fingerprint {fp[:12]}")
        if fp != mod["scorecard"]["test_set"]["fingerprint_sha256"]:
            fail.append(f"{mod['label']}: recomputed fingerprint differs from its scorecard")
        if smp.size != mod["scorecard"]["test_set"]["n"]:
            fail.append(f"{mod['label']}: test row count differs from its scorecard")
        for t in TARGETS:
            if not np.all(np.isfinite(pred[t])):
                fail.append(f"{mod['label']}: {int(np.sum(~np.isfinite(pred[t])))} nonfinite pred_{t}")
        if ref is None:
            ref = dict(smp=smp, true=true, fp=fp)
        else:
            if fp != ref["fp"] or not np.array_equal(smp, ref["smp"]):
                fail.append(f"{mod['label']}: test rows differ from {models[0]['label']}")
            for t in TARGETS:
                if not np.array_equal(true[t], ref["true"][t]):
                    fail.append(f"{mod['label']}: source truths for {t} differ from {models[0]['label']}")
        mod["pred_all"] = pred
    n_rows, n_unique, id_smp = data.scan_test_ids(data_paths.pop(), models[0]["cfg"]["split_column"])
    if n_unique != n_rows:
        fail.append(f"{n_rows - n_unique} duplicate test event IDs ({'/'.join(data.ID_COLUMNS)})")
    if not np.array_equal(id_smp, ref["smp"]):
        fail.append("sample labels from the ID scan differ from the loader's")
    if sample not in set(ref["smp"]):
        fail.append(f"sample {sample} has no test rows")
    if fail:
        sys.exit("alignment checks failed:\n  " + "\n  ".join(fail))
    return ref, dict(n_rows=n_rows, n_unique_ids=n_unique)


def legacy_check(mod, smp_all, true_all, axes, sample):
    """Pilot check 1: this code path reproduces the saved whole-window scorecard values exactly."""
    sel = (smp_all == sample) & in_headline_window(true_all["trueE"])
    r = {k: relative_error(mod["pred_all"][k], true_all[k]) for k in TARGETS}
    s = stats(r["trueE"][sel])
    x, y = 100 * r["trueE"][sel], 100 * r["trueLepE"][sel]
    a, b = axes["a_percent"], axes["b_percent"]
    frac = float(np.mean(inside_ellipse(x, y, a, b)))
    via_fn = ellipse_fraction(x, y, a, b)["unbinned_fraction"]
    saved_m = mod["scorecard"]["metrics"]["trueE"][sample]
    saved_e = mod["scorecard"]["ellipse"]["by_sample"][sample]["fraction"]
    rows = [("bias", float(s["bias"]), saved_m["bias"]), ("resolution", float(s["res"]), saved_m["resolution"]),
            ("ellipse_fraction", frac, saved_e), ("ellipse_fraction_via_scorecard_fn", via_fn, saved_e)]
    detail = {k: dict(recomputed=v, saved=sv, abs_diff=abs(v - sv)) for k, v, sv in rows}
    detail["n"] = dict(recomputed=s["n"], saved=saved_m["n"])
    ok = all(d["abs_diff"] <= rg.LEGACY_ATOL for k, d in detail.items() if k != "n") and s["n"] == saved_m["n"]
    return dict(name=f"1 legacy reproduction {mod['label']}", passed=bool(ok), detail=detail)


def cmd_regional(a):
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
    t, tmu = ref["true"]["trueE"][sel], ref["true"]["trueLepE"][sel]
    events = {}
    for mod in models:
        p, pmu = mod["pred_all"]["trueE"][sel], mod["pred_all"]["trueLepE"][sel]
        r, rmu = relative_error(p, t), relative_error(pmu, tmu)
        events[mod["label"]] = dict(p=p, r=r, inside=inside_ellipse(100 * r, 100 * rmu, a_pct, b_pct))
    point = {label: rg.model_metrics(t, e["r"], e["inside"], e["p"]) for label, e in events.items()}
    checks += rg.count_checks(point, t)
    checks.append(dict(name="2/4 boundaries and availability", passed=True,
                       detail=dict(events_exactly_at_boundaries={f"{b:g}": int(np.sum(t == b))
                                                                 for b in (0.5, 1, 1.5, 2.5, 3, 4, 6)},
                                   n_sample=int(t.size), true_min=float(t.min()), true_max=float(t.max()),
                                   n_true_ge_4=int(np.sum(t >= 4)), n_true_ge_6=int(np.sum(t >= 6)),
                                   n_true_below_0p5=int(np.sum(t < 0.5)))))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        print(json.dumps(failed, indent=2, default=str))
        sys.exit(f"{len(failed)} pilot checks failed; nothing written")

    reps = rg.paired_bootstrap(t, events, a.n_boot, a.seed, progress=lambda i: print(f"  bootstrap {i}/{a.n_boot}"))
    results, paired = rg.summarise(point, reps, REFERENCE)
    data_path = models[0]["cfg"]["data_path"]
    print("hashing dataset ...")
    summary = dict(
        schema=REGIONAL_SCHEMA, created=datetime.datetime.now().isoformat(timespec="seconds"),
        horn=a.horn, sample=a.sample, reference_model=REFERENCE, models=[m["label"] for m in models],
        cuts=rg.CUTS, units="fractions (display x100 as percent)", weights="unit event weights",
        quantile_convention="numpy.quantile default (linear) via pcnumu.core.metrics.stats()",
        bootstrap=dict(n_boot=a.n_boot, seed=a.seed, rng="numpy default_rng", method="paired event bootstrap within "
                       "horn/sample: one resample of events per replicate, shared by all models; every statistic, "
                       "including N_D, recomputed per replicate", interval="central 68% of defined replicates",
                       meaning="finite-event uncertainty conditional on the trained models; not seed-to-seed variation",
                       low_count_flag=f"migration numerator < {rg.LOW_COUNT} or equal to its denominator"),
        ellipse=dict(lookup=axes["frozen_as"], a_percent=a_pct, b_percent=b_pct, provisional=False,
                     calibration_population="r05 sample 3, about 66% inside",
                     calibration_window="true trueE in [0.5, 6] GeV inclusive", regional_recalibration=False,
                     definition="unbinned: (100 r_nu / a)^2 + (100 r_mu / b)^2 <= 1, event-level denominator"),
        provenance=dict(code=data.code_version(), code_sha256=data.code_hashes(os.path.abspath(__file__)),
                        numpy=np.__version__,
                        dataset=dict(path=data_path, sha256=data.sha256_file(data_path), bytes=os.path.getsize(data_path)),
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
        for label, res in results.items():
            for k, e in res.items():
                w.writerow([a.horn, a.sample, label, *k.split("|"), e["value"], e["lo68"], e["hi68"], e["n_undefined_reps"]])
    dk = rg.wkey(rg.CUTS["reco_destination"])
    with open(os.path.join(out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "source_true_window", "N_S", "m_S_into_reco_" + dk, "below_D", "above_D", "N_D",
                    "migration_rate", "contribution", "bootstrap_limited"])
        for label in results:
            pm = point[label]
            for sw in rg.source_windows():
                k = rg.wkey(sw)
                w.writerow([label, k, pm[f"N_S|{k}"], pm[f"m_S|{k}"], pm[f"below_D|{k}"], pm[f"above_D|{k}"],
                            pm[f"N_D|{dk}"], pm[f"migration_rate|{k}"], pm[f"contribution|{k}"],
                            results[label][f"migration_rate|{k}"]["bootstrap_limited"]])
        w.writerow([])
        w.writerow(["window_kind", "true_window", "n_events"])
        for wd in rg.stat_windows():
            w.writerow(["bias/resolution", rg.wkey(wd), point[REFERENCE][f"n|{rg.wkey(wd)}"]])
        for wd in rg.CUTS["ellipse_windows"]:
            w.writerow(["ellipse", rg.wkey(wd), point[REFERENCE][f"ellipse_n|{rg.wkey(wd)}"]])
    print(f"wrote {out}")


# ======================= render =======================
def tex(w, var="true"):
    e = rf"E_\nu^{{\rm {var}}}"
    return rf"${e} \geq {w[0]:g}$ GeV" if w[1] is None else rf"${w[0]:g} \leq {e} < {w[1]:g}$ GeV"


def txt(w):
    return f"≥ {w[0]:g}" if w[1] is None else f"[{w[0]:g}, {w[1]:g})"


def get(s, model, metric, w):
    return s["results"][model][f"{metric}|{rg.wkey(w)}"]


def pct(e):
    """(value, lo, hi) in percent; NaN where undefined."""
    f = lambda v: np.nan if v is None else 100.0 * v  # noqa: E731
    return f(e["value"]), f(e["lo68"]), f(e["hi68"])


def plot_specs(cuts):
    d = cuts["reco_destination"]
    return [  # (file, title, ylabel, metric, windows, legend-N metric, zero line)
        ("resolution", "Neutrino-energy resolution", "Resolution [%]", "resolution", cuts["resolution_windows"], "n", False),
        ("bias", "Neutrino-energy bias", "Bias [%]", "bias", cuts["bias_windows"], "n", True),
        ("migration_rate", f"Migration into {tex(d, 'reco')}", "Fraction of true-source events [%]",
         "migration_rate", cuts["migration_sources"], "N_S", False),
        ("ellipse", "Joint neutrino/muon accuracy", "Ellipse fraction [%]", "ellipse_fraction",
         cuts["ellipse_windows"], "ellipse_n", False),
    ]


def series(s, metric, windows, nmetric):
    out = []
    for w in windows:
        vals = np.array([pct(get(s, m, metric, w)) for m in s["models"]])  # (model, [v, lo, hi])
        n = get(s, s["reference_model"], nmetric, w)["value"]
        out.append((f"{tex(w)}  (N={n:,})", vals))
    return out


def limits(all_series, zero):
    v = np.concatenate([x[1].ravel() for ss in all_series for x in ss])
    v = v[np.isfinite(v)]
    lo, hi = (min(v.min(), 0.0), max(v.max(), 0.0)) if v.size else (0.0, 1.0)
    span = hi - lo or 1.0
    if zero:
        return lo - 0.15 * span, hi + 0.6 * span
    return 0.0, hi * 1.45 if hi > 0 else 1.0


def bar_axes(s, title, ylabel):
    fig, ax = style.figure(size=(8.8, 6))
    style.header(ax, title, f"{style.HORN[s['horn']]} · S{s['sample']}")
    x = np.arange(len(s["models"]))
    ax.set_xticks(x, s["models"])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel("Model", loc="right")
    ax.set_ylabel(ylabel, loc="top")
    return fig, ax, x


def grouped(s, out, name, title, ylabel, ss, ylim, zero):
    fig, ax, x = bar_axes(s, title, ylabel)
    width = 0.72 / len(ss)
    for j, (label, vals) in enumerate(ss):
        xs = x + (j - (len(ss) - 1) / 2) * width
        v = vals[:, 0]
        ok = np.isfinite(v)
        bars = ax.bar(xs[ok], v[ok], width=width, color=style.BARS[j], edgecolor="black", linewidth=0.85,
                      label=label, zorder=2)
        assert [b.get_height() for b in bars] == v[ok].tolist(), f"{name}: drawn bar != summary value"
        e = ok & np.isfinite(vals[:, 1])
        ax.errorbar(xs[e], v[e], yerr=[v[e] - vals[e, 1], vals[e, 2] - v[e]], fmt="none", ecolor="black",
                    elinewidth=1, capsize=2.5, zorder=3)
        for xi in xs[~ok]:
            ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=9)
    if zero:
        ax.axhline(0, color="black", linewidth=1, zorder=3)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=11)
    style.save(fig, out, name)


def contribution_values(s):
    c = s["cuts"]
    seg = [np.array([pct(get(s, m, "contribution", w)) for m in s["models"]]) for w in c["migration_sources"]]
    tot = np.array([pct(get(s, m, "contribution", c["migration_all"])) for m in s["models"]])
    return seg, tot


def contribution_plot(s, out, ylim):
    c = s["cuts"]
    fig, ax, x = bar_axes(s, f"Contribution to {tex(c['reco_destination'], 'reco')}",
                          "Fraction of reconstructed-window events [%]")
    seg, tot = contribution_values(s)
    bottom = np.zeros(len(x))
    for j, (w, vals) in enumerate(zip(c["migration_sources"], seg)):
        v = np.nan_to_num(vals[:, 0])
        bars = ax.bar(x, v, bottom=bottom, width=0.6, color=style.BARS[j], edgecolor="black", linewidth=0.85,
                      label=tex(w), zorder=2)
        # matplotlib stores a stacked height as (bottom + h) - bottom, so compare to float precision
        assert np.allclose([b.get_height() for b in bars], v, rtol=0, atol=1e-9), \
            "contribution: drawn segment != summary value"
        for i, (b0, h) in enumerate(zip(bottom, v)):
            if h > 0.06 * ylim[1]:
                ax.text(i, b0 + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=11, color="white")
        bottom = bottom + v
    assert np.allclose(bottom, np.nan_to_num(tot[:, 0]), rtol=0, atol=1e-9), "stack total != >=4 contribution"
    e = np.isfinite(tot[:, 1])
    ax.errorbar(x[e], tot[e, 0], yerr=[tot[e, 0] - tot[e, 1], tot[e, 2] - tot[e, 0]], fmt="none", ecolor="black",
                elinewidth=1, capsize=3, zorder=3, label=f"Total {tex(c['migration_all'])}, 68% interval")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=11)
    style.save(fig, out, "migration_contribution")


def cell(e, digits=2):
    v, lo, hi = pct(e)
    if not np.isfinite(v):
        return "undefined"
    if not np.isfinite(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render_md(s, summary_sha):
    c, ms, ref = s["cuts"], s["models"], s["reference_model"]
    n = lambda w, k="n": f"{get(s, ref, k, w)['value']:,}"  # noqa: E731
    dk = rg.wkey(c["reco_destination"])
    table = report.table
    L = [f"# Regional comparison: {s['horn'].upper()} sample {s['sample']}", "",
         f"Generated by `pcnumu/eval/compare.py render` from `summary.json` (schema `{s['schema']}`, "
         f"sha256 `{summary_sha[:16]}`, created {s['created']}); do not edit by hand.",
         "Values in percent. Brackets: central 68% paired event bootstrap interval "
         f"({s['bootstrap']['n_boot']} resamples, seed {s['bootstrap']['seed']}): finite-event uncertainty "
         "for these trained models, not training (seed-to-seed) variation. Unit event weights; MC entries, "
         "not physical yields. True-energy windows are lower-inclusive, upper-exclusive, in GeV.", "",
         "## Models", "",
         table(["Model", "Trial", "Seed", "Weighting", "Input noise"],
               [[m["label"], m["trial"], m["seed"], m["weight"]["name"] if m["weight"] else "none",
                 "yes" if m["input_noise"] else "none"] for m in s["provenance"]["models"]]),
         "", "r05/r06 (seed 0) vs r08/r09 (seed 1) differ in both weighting and seed. r07 repeats give the "
         "training-variation context; they are not error bars here.", "",
         "## Resolution [%]", "",
         table(["Model"] + [f"{txt(w)} (N={n(w)})" for w in c["resolution_windows"]],
               [[m] + [cell(get(s, m, "resolution", w)) for w in c["resolution_windows"]] for m in ms]),
         "", "## Bias [%]", "",
         table(["Model"] + [f"{txt(w)} (N={n(w)})" for w in c["bias_windows"]],
               [[m] + [cell(get(s, m, "bias", w)) for w in c["bias_windows"]] for m in ms]),
         "", f"## Migration into reconstructed {txt(c['reco_destination'])}", "",
         "Rate = m / N_S (source leakage into the window). Contribution = m / N_D (source share of the "
         "window); N_D counts every reconstructed-window event of any true energy and depends on the model. "
         "Contributions of the two sources add; rates do not.", "",
         table(["Model", "True source", "m", "N_S", "Rate [%]", "Contribution [%]", "N_D"],
               [[m, txt(w), get(s, m, "m_S", w)["value"], get(s, m, "N_S", w)["value"],
                 cell(get(s, m, "migration_rate", w)), cell(get(s, m, "contribution", w)),
                 s["results"][m][f"N_D|{dk}"]["value"]]
                for m in ms for w in c["migration_sources"]]),
         "", "## Companion tables", "",
         "### Ellipse fraction [%]", "",
         f"Frozen axes `{s['ellipse']['lookup']}` a = {s['ellipse']['a_percent']}%, b = {s['ellipse']['b_percent']}%, "
         f"calibrated on {s['ellipse']['calibration_population']} over {s['ellipse']['calibration_window']}. "
         "Reused here without regional recalibration, so fractions are not expected to be near 66%.", "",
         table(["Model"] + [f"{txt(w)} (N={n(w, 'ellipse_n')})" for w in c["ellipse_windows"]],
               [[m] + [cell(get(s, m, "ellipse_fraction", w)) for w in c["ellipse_windows"]] for m in ms]),
         "", "### Broad-region bias [%]", "",
         table(["Model"] + [txt(w) for w in c["resolution_windows"]],
               [[m] + [cell(get(s, m, "bias", w)) for w in c["resolution_windows"]] for m in ms]),
         "", f"### Overflow breakdown into reconstructed {txt(c['reco_destination'])}", "",
         table(["Model", "True source", "m", "N_S", "Rate [%]", "Contribution [%]", "Low-count flag"],
               [[m, txt(w), get(s, m, "m_S", w)["value"], get(s, m, "N_S", w)["value"],
                 cell(get(s, m, "migration_rate", w)), cell(get(s, m, "contribution", w)),
                 "yes" if get(s, m, "migration_rate", w)["bootstrap_limited"] else ""]
                for m in ms for w in c["migration_sources"][1:] + c["overflow_breakdown"]]),
         "", f"### Paired differences vs {ref} [percentage points]", "",
         "Same bootstrap resamples for both models; the interval is on the difference.", ""]
    hdr = [f"res {txt(w)}" for w in c["resolution_windows"]] + [f"bias {txt(w)}" for w in c["bias_windows"]]
    rows = [[m] + [cell(s["paired_diff_vs_reference"][m][f"resolution|{rg.wkey(w)}"]) for w in c["resolution_windows"]]
            + [cell(s["paired_diff_vs_reference"][m][f"bias|{rg.wkey(w)}"]) for w in c["bias_windows"]]
            for m in ms if m != ref]
    L += [table(["Model"] + hdr, rows), "", "## Checks", ""]
    L += [f"- {'pass' if ck['passed'] else 'FAIL'}: {ck['name']}" for ck in s["checks"]]
    L += ["", f"Code `{s['provenance']['code']['commit']}` (dirty: {s['provenance']['code']['dirty']}); "
          f"test-set fingerprint `{s['provenance']['test_fingerprint_sha256'][:12]}`; "
          f"rendered {datetime.datetime.now().isoformat(timespec='seconds')}.", ""]
    return "\n".join(L)


def cmd_render(a):
    loaded = []
    for d in a.dirs:
        raw = open(os.path.join(d, "summary.json"), "rb").read()
        loaded.append((d, json.loads(raw), hashlib.sha256(raw).hexdigest()))
    if len({s["sample"] for _, s, _ in loaded}) > 1:
        sys.exit("pass folders of one sample only (shared scales are per sample)")
    style.use()
    sp = plot_specs(loaded[0][1]["cuts"])
    shared = {name: limits([series(s, metric, windows, nm) for _, s, _ in loaded], zero)
              for name, _, _, metric, windows, nm, zero in sp}
    tops = [max(np.nanmax(np.r_[t[:, 0], t[:, 2]]), 1e-9) for _, s, _ in loaded for t in [contribution_values(s)[1]]]
    shared["migration_contribution"] = (0.0, max(tops) * 1.45)
    for d, s, sha in loaded:
        for name, title, ylabel, metric, windows, nm, zero in sp:
            grouped(s, d, name, title, ylabel, series(s, metric, windows, nm), shared[name], zero)
        contribution_plot(s, d, shared["migration_contribution"])
        with open(os.path.join(d, "comparison.md"), "w") as f:
            f.write(render_md(s, sha))
        print(f"rendered {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("runs", help="like-for-like metrics of any runs")
    p.add_argument("--run", action="append", required=True, help="LABEL|RESULT_CSV or LABEL|CONFIG|SAMPLE")
    p.add_argument("--out-dir", required=True, help="fresh output directory; refuses an existing directory")
    p.set_defaults(fn=cmd_runs)
    p = sp.add_parser("regional", help="paired regional comparison of r05/r06/r08/r09")
    p.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    p.add_argument("--sample", required=True, choices=["1", "2", "3"])
    p.add_argument("--out-dir", required=True, help="version directory, e.g. pcnumu-analysis/studies/YYYYMMDD_regional-comparison-v2")
    p.add_argument("--runs-root", default=RUNS_ROOT)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261116)
    p.set_defaults(fn=cmd_regional)
    p = sp.add_parser("render", help="plots and comparison.md from regional summary.json files")
    p.add_argument("dirs", nargs="+", help="folders holding summary.json (same sample, FHC and/or RHC)")
    p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
