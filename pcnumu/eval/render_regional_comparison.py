"""Plots and Markdown tables from regional_comparison.py's summary.json; no metric is computed here.

    python pcnumu/eval/render_regional_comparison.py OUT/fhc/s3 [OUT/rhc/s3 ...]

Writes into each summary folder: resolution, bias, migration_rate, migration_contribution, ellipse
(.png and .pdf) and comparison.md. These files are derived and are regenerated on every call. Pass
the FHC and RHC folders of one sample together to give corresponding plots shared y scales.
Every drawn bar height is checked against summary.json before saving.
"""
import argparse
import datetime
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 1.2,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "xtick.major.size": 7, "ytick.major.size": 7, "legend.frameon": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
COLORS = ["#0000ff", "#ff0000", "#008000"]


def wkey(w):
    hi = "inf" if w[1] is None else f"{w[1]:g}"
    return f"{w[0]:g}-{hi}"


def tex(w, var="true"):
    e = rf"E_\nu^{{\rm {var}}}"
    if w[1] is None:
        return rf"${e} \geq {w[0]:g}$ GeV"
    return rf"${w[0]:g} \leq {e} < {w[1]:g}$ GeV"


def txt(w):
    return f"≥ {w[0]:g}" if w[1] is None else f"[{w[0]:g}, {w[1]:g})"


def get(s, model, metric, w):
    return s["results"][model][f"{metric}|{wkey(w)}"]


def pct(e):
    """(value, lo, hi) in percent; NaN where undefined."""
    f = lambda v: np.nan if v is None else 100.0 * v  # noqa: E731
    return f(e["value"]), f(e["lo68"]), f(e["hi68"])


# ---------- plot specs: (file, title, ylabel, metric, windows, legend-N metric, zero line) ----------
def specs(cuts):
    d = cuts["reco_destination"]
    return [
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


def base(s, title, ylabel):
    f, a = plt.subplots(figsize=(8.8, 6))
    f.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.91)
    a.set_title(title, loc="left", fontsize=12, pad=7, fontweight="normal")
    a.text(1, 1.014, f"{s['horn'].upper()} · S{s['sample']}", transform=a.transAxes, ha="right", va="bottom",
           fontsize=10, color="#555555")
    x = np.arange(len(s["models"]))
    a.set_xticks(x, s["models"])
    a.set_xlim(-0.6, len(x) - 0.4)
    a.set_ylabel(ylabel)
    a.set_xlabel("Model")
    a.minorticks_on()
    a.tick_params(axis="x", which="minor", bottom=False, top=False)
    return f, a, x


def save(f, out, name):
    for ext in ("png", "pdf"):
        f.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160)
    plt.close(f)


def grouped(s, out, name, title, ylabel, ss, ylim, zero):
    f, a, x = base(s, title, ylabel)
    width = 0.72 / len(ss)
    for j, (label, vals) in enumerate(ss):
        xs = x + (j - (len(ss) - 1) / 2) * width
        v = vals[:, 0]
        ok = np.isfinite(v)
        bars = a.bar(xs[ok], v[ok], width=width, color=COLORS[j], edgecolor="black", linewidth=0.85,
                     label=label, zorder=2)
        assert [b.get_height() for b in bars] == v[ok].tolist(), f"{name}: drawn bar != summary value"
        e = ok & np.isfinite(vals[:, 1])
        a.errorbar(xs[e], v[e], yerr=[v[e] - vals[e, 1], vals[e, 2] - v[e]], fmt="none", ecolor="black",
                   elinewidth=1, capsize=2.5, zorder=3)
        for xi in xs[~ok]:
            a.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=9)
    if zero:
        a.axhline(0, color="black", linewidth=1, zorder=3)
    a.set_ylim(*ylim)
    a.legend(loc="upper right", fontsize=10.5)
    save(f, out, name)


def contribution_values(s):
    c = s["cuts"]
    seg = [np.array([pct(get(s, m, "contribution", w)) for m in s["models"]]) for w in c["migration_sources"]]
    tot = np.array([pct(get(s, m, "contribution", c["migration_all"])) for m in s["models"]])
    return seg, tot


def contribution_plot(s, out, ylim):
    c = s["cuts"]
    f, a, x = base(s, f"Contribution to {tex(c['reco_destination'], 'reco')}",
                   "Fraction of reconstructed-window events [%]")
    seg, tot = contribution_values(s)
    bottom = np.zeros(len(x))
    for j, (w, vals) in enumerate(zip(c["migration_sources"], seg)):
        v = np.nan_to_num(vals[:, 0])
        bars = a.bar(x, v, bottom=bottom, width=0.6, color=COLORS[j], edgecolor="black", linewidth=0.85,
                     label=tex(w), zorder=2)
        # matplotlib stores a stacked height as (bottom + h) - bottom, so compare to float precision
        assert np.allclose([b.get_height() for b in bars], v, rtol=0, atol=1e-9), \
            "contribution: drawn segment != summary value"
        for i, (b0, h) in enumerate(zip(bottom, v)):
            if h > 0.06 * ylim[1]:
                a.text(i, b0 + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=11, color="white")
        bottom = bottom + v
    assert np.allclose(bottom, np.nan_to_num(tot[:, 0]), rtol=0, atol=1e-9), "stack total != >=4 contribution"
    e = np.isfinite(tot[:, 1])
    a.errorbar(x[e], tot[e, 0], yerr=[tot[e, 0] - tot[e, 1], tot[e, 2] - tot[e, 0]], fmt="none", ecolor="black",
               elinewidth=1, capsize=3, zorder=3, label=f"total {tex(c['migration_all'])}, 68% interval")
    a.set_ylim(*ylim)
    a.legend(loc="upper right", fontsize=10.5)
    save(f, out, "migration_contribution")


# ---------- markdown ----------
def cell(e, digits=2):
    v, lo, hi = pct(e)
    if not np.isfinite(v):
        return "undefined"
    if not np.isfinite(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_md(s, summary_sha):
    c, ms, ref = s["cuts"], s["models"], s["reference_model"]
    n = lambda w, k="n": f"{get(s, ref, k, w)['value']:,}"  # noqa: E731
    dk = wkey(c["reco_destination"])
    L = [f"# Regional comparison: {s['horn'].upper()} sample {s['sample']}", "",
         f"Generated by `pcnumu/eval/render_regional_comparison.py` from `summary.json` (schema `{s['schema']}`, "
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
    rows = []
    for m in ms:
        if m == ref:
            continue
        pd = s["paired_diff_vs_reference"][m]
        rows.append([m] + [cell(pd[f"resolution|{wkey(w)}"]) for w in c["resolution_windows"]]
                    + [cell(pd[f"bias|{wkey(w)}"]) for w in c["bias_windows"]])
    L += [table(["Model"] + hdr, rows), "", "## Checks", ""]
    L += [f"- {'pass' if ck['passed'] else 'FAIL'}: {ck['name']}" for ck in s["checks"]]
    L += ["", f"Code `{s['provenance']['code']['commit']}` (dirty: {s['provenance']['code']['dirty']}); "
          f"test-set fingerprint `{s['provenance']['test_fingerprint_sha256'][:12]}`; "
          f"rendered {datetime.datetime.now().isoformat(timespec='seconds')}.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="folders holding summary.json (same sample, FHC and/or RHC)")
    a = ap.parse_args()
    loaded = []
    for d in a.dirs:
        raw = open(os.path.join(d, "summary.json"), "rb").read()
        loaded.append((d, json.loads(raw), hashlib.sha256(raw).hexdigest()))
    if len({s["sample"] for _, s, _ in loaded}) > 1:
        raise SystemExit("pass folders of one sample only (shared scales are per sample)")
    sp = specs(loaded[0][1]["cuts"])
    shared = {}
    for name, _, _, metric, windows, nm, zero in sp:
        shared[name] = limits([series(s, metric, windows, nm) for _, s, _ in loaded], zero)
    tops = [max(np.nanmax(np.r_[t[:, 0], t[:, 2]]), 1e-9) for _, s, _ in loaded for t in [contribution_values(s)[1]]]
    shared["migration_contribution"] = (0.0, max(tops) * 1.45)
    for d, s, sha in loaded:
        for name, title, ylabel, metric, windows, nm, zero in sp:
            grouped(s, d, name, title, ylabel, series(s, metric, windows, nm), shared[name], zero)
        contribution_plot(s, d, shared["migration_contribution"])
        with open(os.path.join(d, "comparison.md"), "w") as f:
            f.write(render_md(s, sha))
        print(f"rendered {d}")


if __name__ == "__main__":
    main()
