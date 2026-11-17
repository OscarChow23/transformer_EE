#!/usr/bin/env python
"""Leaderboard CSVs (one row per model, per horn) and the model-comparison note.

    python -I pcnumu/eval/leaderboard.py [--runs-root DIR] [--reference TRIAL] [--md PATH] [--regional DIR]

Reads RUNS_ROOT/<trial>/<horn>/scorecard/{scorecard.json,peak_metrics.json} and the trial's
trial.yaml / resolved.json / go-record.md. Only runs whose test-set fingerprint equals the
reference trial's (default r05, the frozen training_v3 test set) enter a horn's table; others
(smoke tests, other datasets) are listed as excluded. Writes:
  RUNS_ROOT/leaderboard_<horn>.csv   every metric, one row per trial
  --md PATH                          markdown note: per-sample regional tables (from --regional, the
                                     summary.json files written by compare.py regional) + one entry per trial
Nothing is typed by hand: rerun after each new scorecard.
"""
import argparse
import csv
import glob
import json
import os

import yaml

RUNS_ROOT = "/data/nova/chow/pcnumu-runs"
REPO_URL = "https://github.com/OscarChow23/transformer_EE"
REGIONAL = "/data/nova/chow/pcnumu-analysis/studies/20261116_regional-comparison-v1"
HORNS = ["fhc", "rhc"]
SAMPLES = ["1", "2", "3", "all"]


def pct(x, nd=2):
    return "" if x is None else f"{100 * x:.{nd}f}"


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def spec_file(tdir):
    """The repo's living spec (its status is updated after training); the run folder's copy is the
    GO-signed snapshot and is only the fallback."""
    p = os.path.join(REPO, "pcnumu", "trials", os.path.basename(tdir), "trial.yaml")
    return p if os.path.exists(p) else os.path.join(tdir, "trial.yaml")


def load_trial(tdir):
    spec = yaml.safe_load(open(spec_file(tdir)))
    resolved = os.path.join(tdir, "resolved.json")
    res = json.load(open(resolved)) if os.path.exists(resolved) else {}
    commit = None
    gr = os.path.join(tdir, "go-record.md")
    if os.path.exists(gr):
        for ln in open(gr):
            if ln.startswith("commit:"):
                commit = ln.split()[1]
    return spec, res, commit


def change_text(spec, res, horn):
    if spec.get("imported"):
        return "root (imported run)"
    diff = res.get("runs", {}).get(horn, {}).get("settings_vs_baseline", {})
    parts = [f"{k} {v[0]} → {v[1]}" for k, v in sorted(diff.items())]
    base = str(spec.get("baseline", "")).split("_")[0]
    return (f"vs {base}: " + "; ".join(parts)) if parts else f"vs {base}: no setting change"


def row_for(tdir, horn):
    card_p = os.path.join(tdir, horn, "scorecard", "scorecard.json")
    if not os.path.exists(card_p):
        return None
    card = json.load(open(card_p))
    peak_p = os.path.join(tdir, horn, "scorecard", "peak_metrics.json")
    peak = json.load(open(peak_p)) if os.path.exists(peak_p) else None
    spec, res, commit = load_trial(tdir)
    r = {"trial": spec["trial"], "horn": horn, "status": spec.get("status", ""),
         "change": change_text(spec, res, horn), "baseline": spec.get("baseline", ""),
         "commit": commit or "",  # the GO-signed commit; imported root trials have none
         "fingerprint": card["test_set"]["fingerprint_sha256"],
         "best_epoch": card["training"]["best_epoch"], "epochs_run": card["training"]["epochs_run"],
         "model_dir": card["model_dir"], "run_dir": card["run_dir"],
         "hypothesis": spec.get("hypothesis", "")}
    for t in ["trueE", "trueLepE"]:
        for s in SAMPLES:
            m = card["metrics"][t][s]
            for k in ["bias", "resolution", "band_05", "band_10", "band_15"]:
                r[f"{t}_S{s}_{k}"] = m[k]
    for s in SAMPLES:
        r[f"ellipse_S{s}"] = card["ellipse"]["by_sample"][s]["fraction"]
    for s in ["1", "2", "3"]:
        r[f"peak_pull_S{s}"] = peak["by_sample"][s]["peak_pull"] if peak else None
        r[f"pull_slope_S{s}"] = peak["by_sample"][s]["pull_slope"] if peak else None
    return r


REGIONAL_SAMPLES = [("3", "S3: peripheral, μ uncontained"), ("2", "S2: peripheral, μ contained"),
                    ("1", "S1: contained"), ("23", "S2+S3 pooled")]  # note order: S3 first


def wtxt(w):
    return f"≥ {w[0]:g}" if w[1] is None else f"[{w[0]:g}, {w[1]:g})"


def wkey(w):
    return f"{w[0]:g}-{'inf' if w[1] is None else f'{w[1]:g}'}"


def regional_cell(e, signed):
    """value ± half the 68% bootstrap interval, in percent."""
    v, lo, hi = e["value"], e["lo68"], e["hi68"]
    if v is None:
        return "undefined"
    nd = 2 if abs(100 * v) >= 0.1 or v == 0 else 3  # small migration shares would round to 0.00
    out = f"{100 * v:{'+' if signed else ''}.{nd}f}"
    return out if lo is None or hi is None else out + f" ± {50 * (hi - lo):.{nd}f}"


def regional_sections(root):
    """Per sample: main tables (one per plot) and companion tables, FHC and RHC side by side.
    Values are read from compare.py regional's summary.json; nothing is recomputed.
    Returns (markdown lines, trials shown)."""
    md, shown = [], set()
    for smp, name in REGIONAL_SAMPLES:
        sums = {h: os.path.join(root, h, f"s{smp}", "summary.json") for h in HORNS}
        if not all(os.path.exists(p) for p in sums.values()):
            md += [f"### {name}", "", f"No regional comparison for both horns in `{root}`.", ""]
            continue
        S = {h: json.load(open(p)) for h, p in sums.items()}
        s0 = S[HORNS[0]]
        c, models = s0["cuts"], s0["models"]
        trials = {m["label"]: m["trial"] for m in s0["provenance"]["models"]}
        shown.update(trials.values())
        dest = wtxt(c["reco_destination"])

        def tab(metric, windows, nkey, signed=False, show_n=True):
            head = ["model"] + [f"{h.upper()} {wtxt(w)}" + (f" (N={S[h]['results'][s0['reference_model']][f'{nkey}|{wkey(w)}']['value']:,})" if show_n else "")
                                for h in HORNS for w in windows]
            rows = [[f"[[#{trials[m]}\\|{m}]]"] + [regional_cell(S[h]["results"][m][f"{metric}|{wkey(w)}"], signed)
                                                    for h in HORNS for w in windows] for m in models]
            return ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)] + \
                   ["| " + " | ".join(r) + " |" for r in rows] + [""]

        md += [f"### {name}", "",
               f"Plots and full tables with intervals: `{root}/<horn>/s{smp}/` (`comparison.md`, `*.png`).", ""]
        if len(smp) > 1:
            md += ["Samples 2 and 3 pooled: test events combined at MC proportions, unit weights.", ""]
        md += ["**Resolution [%]** (`resolution.png`)", ""] + tab("resolution", c["resolution_windows"], "n")
        md += ["**Bias [%]** (`bias.png`)", ""] + tab("bias", c["bias_windows"], "n", signed=True)
        md += [f"**Migration rate into reco {dest} [%]** (`migration_rate.png`): share of each true-energy "
               "source that is reconstructed in the window", ""] + tab("migration_rate", c["migration_sources"], "N_S")
        md += [f"**Migration contribution to reco {dest} [%]** (`migration_contribution.png`): share of the "
               "reconstructed window that comes from each source; the sources add", ""] + \
              tab("contribution", c["migration_sources"], "N_S", show_n=False)
        md += ["*Companion: ellipse fraction [%]* (`ellipse.png`; frozen 22% axes, not recalibrated per region)", ""] + \
              tab("ellipse_fraction", c["ellipse_windows"], "ellipse_n")
        md += ["*Companion: broad-region bias [%]*", ""] + \
              tab("bias", c["resolution_windows"], "n", signed=True, show_n=False)
    return md, shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--reference", default="r05_v3data-allsamples_v3features")
    ap.add_argument("--md")
    ap.add_argument("--regional", default=REGIONAL, help="compare.py regional version directory")
    a = ap.parse_args()
    tdirs = sorted(d for d in glob.glob(os.path.join(a.runs_root, "r*")) if os.path.exists(spec_file(d)))
    md = ["---", "title: TransformerEE — model comparison", "type: note", "created: 2026-11-04",
          "tags: [nova, numu, pc-numu, machine-learning, energy-estimation, transformeree]", "---", "",
          "# TransformerEE model comparison", "",
          "Generated by `pcnumu/eval/leaderboard.py` from each run's scorecard; do not edit by hand. "
          "FHC and RHC have different test events: compare models within a horn, not across. "
          "Resolution = (q84 − q16)/2 of r = pred/true − 1; bias = median r; windows are true trueE in GeV, "
          "lower-inclusive. ± = half the central 68% paired event bootstrap interval: finite test-event "
          "uncertainty for these trained models, not seed-to-seed variation. "
          "Only the models in the regional comparison are listed. Headline 0.5–6 GeV metrics for every "
          "trial, peak pull and pull slope: "
          "`" + a.runs_root + "/leaderboard_<horn>.csv`.", ""]
    entries, excluded, fps = {}, [], []
    for horn in HORNS:
        ref = row_for(os.path.join(a.runs_root, a.reference), horn)
        rows = []
        for d in tdirs:
            r = row_for(d, horn)
            if r is None:
                continue
            if r["fingerprint"] != ref["fingerprint"]:
                excluded.append(f"{r['trial']} ({horn}): different test set")
                continue
            rows.append(r)
            entries[r["trial"]] = entries.get(r["trial"], {})
            entries[r["trial"]][horn] = r
        keys = list(rows[0].keys()) if rows else []
        with open(os.path.join(a.runs_root, f"leaderboard_{horn}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        fps.append(f"{horn.upper()} `{ref['fingerprint'][:12]}`")
    md += ["## Regional comparison by sample", "",
           f"What the plots and error bars mean: `{a.regional}/README.md`.", "",
           f"Paired comparison of the models in each summary.json on the same held-out events per horn "
           f"(test sets {', '.join(fps)}).", ""]
    reg_md, shown = regional_sections(a.regional)
    md += reg_md
    md += ["## Models", ""]
    for trial in sorted(t for t in entries if t in shown):
        e = entries[trial]
        any_r = next(iter(e.values()))
        md += [f"### {trial}", "", f"- **Status:** {any_r['status']}. **Baseline:** {any_r['baseline'] or '—'}.",
               f"- **Hypothesis:** {any_r['hypothesis']}"]
        if any_r["commit"]:
            md.append(f"- **Definition:** [trial.yaml at {any_r['commit'][:7]}]"
                      f"({REPO_URL}/blob/{any_r['commit']}/pcnumu/trials/{trial}/trial.yaml), the GO-signed commit")
        else:
            md.append(f"- **Definition:** [trial.yaml on pcnumu-loop]"
                      f"({REPO_URL}/blob/pcnumu-loop/pcnumu/trials/{trial}/trial.yaml); trained before the loop, "
                      "config imported and pinned by sha256")
        for horn, r in e.items():
            md.append(f"- **{horn.upper()}:** {r['change']}; best epoch {r['best_epoch']} of {r['epochs_run']}. "
                      f"Config `{r['model_dir']}/input.json`; scorecard `{r['run_dir']}/scorecard/`.")
        gr = os.path.join(a.runs_root, trial, "go-record.md")
        if os.path.exists(gr):
            md.append(f"- **GO record:** `{gr}`")
        md.append("")
    if excluded:
        md += ["## Not in the tables", ""] + [f"- {x}" for x in excluded] + [""]
    up = "01-Projects/pc-numu-recovery/sub-projects/transformeree-adaptation"
    md += ["## Connections", "",
           f"**Topics:** [[{up}/_index|TransformerEE Adaptation]]", "",
           "**Tags:** #nova #numu #pc-numu #machine-learning #energy-estimation #transformeree", "",
           f"**Up:** [[{up}/_index|TransformerEE Adaptation]]", "",
           "**Related pages:**", "",
           f"- [[{up}/flattening|Flattening — diagnosis and weight-preview evidence]]",
           f"- [[{up}/runs|Runs]]", f"- [[{up}/code-figures|Figure records]]", ""]
    if a.md:
        open(a.md, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
