#!/usr/bin/env python
"""One row per model, one table per horn: generated from every run's scorecard.

    python -I pcnumu/eval/leaderboard.py [--runs-root DIR] [--reference TRIAL] [--md PATH]

Reads RUNS_ROOT/<trial>/<horn>/scorecard/{scorecard.json,peak_metrics.json} and the trial's
trial.yaml / resolved.json / go-record.md. Only runs whose test-set fingerprint equals the
reference trial's (default r05, the frozen training_v3 test set) enter a horn's table; others
(smoke tests, other datasets) are listed as excluded. Writes:
  RUNS_ROOT/leaderboard_<horn>.csv   every metric, one row per trial
  --md PATH                          markdown note: per-horn headline tables + one entry per trial
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


HEAD = {  # per-horn tables: title -> [(column title, key, formatter)]; full names so no header is ambiguous
    "Energy resolution and bias": [
        ("trueE res S1 %", "trueE_S1_resolution", pct), ("trueE res S2 %", "trueE_S2_resolution", pct),
        ("trueE res S3 %", "trueE_S3_resolution", pct), ("trueE bias S3 %", "trueE_S3_bias", pct),
        ("S3 within ±10 %", "trueE_S3_band_10", lambda x: pct(x, 1)),
        ("trueLepE res S3 %", "trueLepE_S3_resolution", pct)],
    "Ellipse and peak learning": [
        ("ellipse S1 %", "ellipse_S1", lambda x: pct(x, 1)), ("ellipse S2 %", "ellipse_S2", lambda x: pct(x, 1)),
        ("ellipse S3 %", "ellipse_S3", lambda x: pct(x, 1)),
        ("peak pull S1", "peak_pull_S1", lambda x: "" if x is None else f"{x:+.1f}"),
        ("peak pull S3", "peak_pull_S3", lambda x: "" if x is None else f"{x:+.1f}"),
        ("pull slope S1", "pull_slope_S1", lambda x: "" if x is None else f"{x:.3f}"),
        ("pull slope S3", "pull_slope_S3", lambda x: "" if x is None else f"{x:.3f}"),
        ("best epoch", "best_epoch", lambda x: f"{x}")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--reference", default="r05_v3data-allsamples_v3features")
    ap.add_argument("--md")
    a = ap.parse_args()
    tdirs = sorted(d for d in glob.glob(os.path.join(a.runs_root, "r*")) if os.path.exists(spec_file(d)))
    md = ["---", "title: TransformerEE — model comparison", "type: note", "created: 2026-11-04",
          "tags: [nova, numu, pc-numu, machine-learning, energy-estimation, transformeree]", "---", "",
          "# TransformerEE model comparison", "",
          "Generated by `pcnumu/eval/leaderboard.py` from each run's scorecard; do not edit by hand. "
          "One table per horn: FHC and RHC have different test events, so compare rows only within a horn. "
          "Window: true trueE 0.5–6 GeV. Resolution = (q84 − q16)/2 of r = pred/true − 1; bias = median r; "
          "ellipse = share inside the frozen 22% circle in the (trueE, trueLepE) error plane; "
          "peak pull = median r at 1.0–1.5 GeV minus 3–4 GeV (0 = no pull towards the ~2 GeV peak); "
          "pull slope = log-log slope of median prediction vs true over 1–5 GeV (1 = no pull). "
          "Full metrics per row: `" + a.runs_root + "/leaderboard_<horn>.csv`.", ""]
    entries, excluded = {}, []
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
        md += [f"## {horn.upper()} (test set `{ref['fingerprint'][:12]}`)", ""]
        for title, cols in HEAD.items():
            md += [f"**{title}**", "", "| model | baseline | " + " | ".join(h for h, _, _ in cols) + " |",
                   "|---|---|" + "---:|" * len(cols)]
            for r in rows:
                link = f"[[#{r['trial']}\\|{r['trial'].split('_')[0]}]]"  # \| : a bare pipe would split the table cell
                vs = r["baseline"].split("_")[0] or "root"  # full change is in the model's section below
                md.append(f"| {link} | {vs} | " + " | ".join(fmt(r[k]) for _, k, fmt in cols) + " |")
            md.append("")
    md += ["## Models", ""]
    for trial in sorted(entries):
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
