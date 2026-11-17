"""p7: compact summary r05/r06/r07 from scorecard.json / peak_metrics.json only (nothing recomputed).

    python -I scripts/p7_summary.py --horn fhc|rhc

Panels: aggregate trueE resolution per sample (0.5-6 GeV window), S3 share of events inside the
frozen 22% circle in the (trueE, trueLepE) fractional-error plane, S3 peak pull (percentage points).
Also writes p7_summary.tex (booktabs) and p7_summary.md with the same numbers.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

import common as cm  # noqa: E402
from common import rs  # noqa: E402

OFFSET = {"r05": -0.22, "r06": 0.0, "r07": 0.22}
CLAIM = {h: "In both horns r06 and r07 have a smaller S3 peak pull than r05, together with a larger aggregate "
            "S3 resolution and a smaller S3 share inside the circle." for h in ("fhc", "rhc")}
YLIM = dict(res=(0, 24), ell=(50, 72), pull=(0, 13))  # shared by FHC and RHC


def texify(t):
    return t.replace("α", "$\\alpha$").replace("½", "$\\tfrac{1}{2}$")


def numbers(cards):
    out = {}
    for r, c in cards.items():
        ell = c["ellipse"]
        if ell["provisional"] or ell["a_percent"] != 22.0 or ell["b_percent"] != 22.0:
            sys.exit(f"{r}: ellipse not the frozen 22% circle")
        out[r] = dict(res={s: 100 * c["metrics"]["trueE"][str(s)]["resolution"] for s in (1, 2, 3)},
                      n={s: c["metrics"]["trueE"][str(s)]["n"] for s in (1, 2, 3)},
                      ell=100 * ell["by_sample"]["3"]["fraction"],
                      pull=c["_peak"]["by_sample"]["3"]["peak_pull"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    a = ap.parse_args()
    rs.use()
    cards = cm.load_cards(a.horn)
    v = numbers(cards)
    H = rs.HORN[a.horn]

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 4.0), gridspec_kw=dict(width_ratios=[3, 1.25, 1.25], wspace=0.55))
    for r in cm.SCORECARDS:
        rs.model_points(axs[0], [1, 2, 3], [v[r]["res"][s] for s in (1, 2, 3)], None, r, offset=OFFSET[r], ms=7)
        rs.model_points(axs[1], [1], [v[r]["ell"]], None, r, offset=OFFSET[r], ms=7)
        rs.model_points(axs[2], [1], [v[r]["pull"]], None, r, offset=OFFSET[r], ms=7)
    axs[0].set_xticks([1, 2, 3], ["S1", "S2", "S3"])
    axs[0].set_xlim(0.5, 3.5)
    for ax in axs[1:]:
        ax.set_xticks([1], ["S3"])
        ax.set_xlim(0.5, 1.5)
    for ax, key, yl in zip(axs, ("res", "ell", "pull"),
                           ("Resolution (%)", "Inside 22% circle (%)", "Peak pull (percentage points)")):
        ax.set_ylim(*YLIM[key])
        rs.labels(ax, None, yl)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="x", which="major", length=0, pad=8)
    rs.tag(axs[0], H)
    axs[0].legend(loc="lower right", fontsize=11, handletextpad=0.3)
    model_line = "; ".join(f"{r} = {c['model_dir']}" for r, c in cards.items())
    rs.save(fig, os.path.join(cm.ROOT, a.horn), "p7_summary",
            f"Left: aggregate resolution (q84 - q16)/2 of r = E_reco/E_true - 1 (trueE), per sample, true E 0.5-6 GeV (%). "
            f"Middle: S3 share of events (%) with (100 r_trueE, 100 r_trueLepE) inside an axis-aligned circle of "
            f"radius 22% centred at (0, 0) in the (trueE, trueLepE) fractional-residual plane (frozen in "
            f"ellipse_frozen.json as a = b = 22%, theta = 0, unbinned); this panel's y axis starts at 50%.\n"
            f"Right: S3 peak pull = 100 x [median r for true E 1.0-1.5 GeV - median r for true E 3.0-4.0 GeV], in "
            f"percentage points (peak_metrics.json).\n"
            f"Events: {H} test split (fingerprint {cards['r05']['test_set']['fingerprint_sha256'][:12]}), window "
            f"N = S1 {v['r05']['n'][1]:,}, S2 {v['r05']['n'][2]:,}, S3 {v['r05']['n'][3]:,}.\n"
            f"Models: {model_line}.\n"
            f"Numbers copied from scorecard.json / peak_metrics.json, not recomputed. No uncertainties are shown "
            f"(none in the scorecard; no training-seed variation), so no significance is claimed.\n" +
            CLAIM[a.horn])

    rows = [(r, cm.MODEL_DESC[r], v[r]["res"][1], v[r]["res"][2], v[r]["res"][3], v[r]["ell"], v[r]["pull"])
            for r in cm.SCORECARDS]
    tex = [r"% " + f"{H} r05/r06/r07 summary; from scorecard.json / peak_metrics.json (talk-plots_20261106/scripts/p7_summary.py)",
           r"\begin{tabular}{llrrrrr}", r"\toprule",
           r" & & \multicolumn{3}{c}{Resolution (\%)} & \multicolumn{2}{c}{S3} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
           r"Run & Training weights & S1 & S2 & S3 & In 22\% circle (\%) & Peak pull (pts) \\", r"\midrule"]
    tex += [f"{r} & {texify(d)} & {a1:.2f} & {a2:.2f} & {a3:.2f} & {e:.1f} & {p:+.2f} \\\\"
            for r, d, a1, a2, a3, e, p in rows]
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    open(os.path.join(cm.ROOT, a.horn, "p7_summary.tex"), "w").write("\n".join(tex))
    md = [f"{H} test set, true E 0.5-6 GeV. From scorecard.json / peak_metrics.json (not recomputed).", "",
          "| Run | Training weights | Res. S1 (%) | Res. S2 (%) | Res. S3 (%) | S3 in 22% circle (%) | S3 peak pull (pts) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r} | {d} | {a1:.2f} | {a2:.2f} | {a3:.2f} | {e:.1f} | {p:+.2f} |" for r, d, a1, a2, a3, e, p in rows]
    open(os.path.join(cm.ROOT, a.horn, "p7_summary.md"), "w").write("\n".join(md) + "\n")

    d = dict(cards=cards, hashes={r: {f: cm.sha256_file(os.path.join(c["model_dir"], f))
                                      for f in ("result.csv", "input.json")} for r, c in cards.items()},
             fingerprint=cards["r05"]["test_set"]["fingerprint_sha256"], data_path=cards["r05"]["data_path"])
    cm.manifest(a.horn, "p7_summary", d, dict(
        source="scorecard.json metrics.trueE.<s>.resolution, ellipse.by_sample.3.fraction; "
               "peak_metrics.json by_sample.3.peak_pull", values=v,
        event_alignment="not needed (no per-event computation); fingerprints identical"))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
