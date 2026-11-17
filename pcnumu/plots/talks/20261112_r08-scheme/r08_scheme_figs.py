"""Talk figures: r08 per-sample flattening scheme, one figure per sample and horn plus a 3-panel overview.
Weighted trueE spectra of the train rows (unit area over 0-10 GeV within the sample): unflattened,
r05 pooled alpha 0.05 (faint), r08 per-sample. Flat range shaded, ease-back region fading out.
Weights rebuilt with transformer_ee's create_weighter from the r05/r08 run configs (as the loader does).
usage: python -I r08_scheme_figs.py OUT_DIR"""
import json
import sys

sys.path[:0] = ["/data/nova/chow/talk-plots_20261106/style", "/data/nova/chow/transformerEE-pcnumu-loop"]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import rootstyle as rs  # noqa: E402
from transformer_ee.utils.weights import create_weighter  # noqa: E402

out = sys.argv[1]
rs.use()
COLOR = {s: rs.SAMPLES[s]["line"] for s in (1, 2, 3)}  # deck colours; e.g. {1: "#c81e1e", 2: "#2e7d32", 3: "#0b5fbf"}
FILL = {s: rs.SAMPLES[s]["fill"] for s in (1, 2, 3)}
BINS = np.array(rs.SPEC_BINS)
YLAB = "Normalised events / 0.2 GeV"


def load(horn):
    c08 = json.load(open(f"/data/nova/chow/pcnumu-runs/r08_per-sample-flat/{horn}/effective_config.json"))
    c05 = json.load(open(f"/data/nova/chow/transformerEE/runs/pc-numu/{horn}/pcnumu_{horn}_r05_v3data-allsamples_v3features_20261029/"
                         f"pcnumu_{horn}_r05_v3data-allsamples_v3features.json"))
    assert c08["data_path"] == c05["data_path"] and c08["split_column"] == c05["split_column"]
    t = (pl.scan_csv(c08["data_path"], infer_schema_length=0).filter(pl.col(c08["split_column"]) == "train")
         .select(["sample", "trueE"]).collect())
    y = t["trueE"].cast(pl.Float64).to_numpy()
    smp = t["sample"].to_numpy().astype(str)
    w08 = create_weighter(c08, y, sample=smp).getweight(y, smp)
    w05 = create_weighter(c05, y).getweight(y)
    return y, smp.astype(int), w05, w08, c08["weight"]["kwargs"]["per_sample"]


def hist(y, w):
    h, _ = np.histogram(y, BINS, weights=w)
    return h / h.sum()


def draw(ax, s, y, w05, w08, p, horn, legend=True):
    lo, hi, top = p["lo"], p["hi"], p["hi"] + p["ease"]
    curves = [hist(y, np.ones_like(y)), hist(y, w05), hist(y, w08)]
    ymax = 1.25 * max(c.max() for c in curves)
    ax.axvspan(lo, hi, color=FILL[s], alpha=0.45, lw=0, zorder=0)
    xs = np.linspace(hi, top, 60)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.axvspan(a, b, color=FILL[s], alpha=0.45 * (1 - (a - hi) / p["ease"]), lw=0, zorder=0)
    ax.stairs(curves[0], BINS, color="#7f7f7f", ls="--", lw=1.4, label="Unflattened")
    ax.stairs(curves[1], BINS, color=rs.MODELS["r05"]["color"], lw=1.4, alpha=0.35,
              label=r"Pooled, $\alpha$ = 0.05 (r05)")
    ax.stairs(curves[2], BINS, color=COLOR[s], lw=2.6, label=rf"Per-sample, $\alpha$ = {p['alpha']:g} (r08)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ymax)
    yt = 0.80 * ymax
    ax.text((lo + hi) / 2, yt, f"flattened\n{lo:g}–{hi:g} GeV", ha="center", va="center", fontsize=11, color=COLOR[s])
    ax.text((hi + top) / 2, yt, f"ease back\nby {top:g} GeV", ha="center", va="center", fontsize=11, color=COLOR[s])
    rs.tag(ax, f"{rs.HORN[horn]} · {rs.SAMPLES[s]['short']} · train", loc="upper right")
    if legend:
        ax.legend(loc="center right", fontsize=11, bbox_to_anchor=(1.0, 0.52))


for horn in ("fhc", "rhc"):
    y, smp, w05, w08, ps = load(horn)
    caption_common = (
        f"{rs.HORN[horn]} training rows, true neutrino energy 0-10 GeV in 0.2 GeV bins. Each curve is normalised to unit area over 0-10 GeV "
        "within the sample (shapes, not sample sizes). Dashed grey: no weights. Faint black: r05 pooled FlatSpectraWeights "
        "(alpha 0.05, 500 bins, one map for all samples). Coloured: r08 PerSampleFlatWeights. Shaded: flat range; fading band: "
        "weights ease back (log-linear) to the sample average. Weights rebuilt with transformer_ee create_weighter from the "
        "run configs. Preview of the training input, not a trained-model result.")
    for s in (1, 2, 3):
        m = smp == s
        p = ps[str(s)]
        fig, ax = rs.figure(size=(7.2, 4.8))
        draw(ax, s, y[m], w05[m], w08[m], p, horn)
        rs.labels(ax, rs.E_TRUE, YLAB)
        rs.save(fig, out, f"r08-scheme-{horn}-s{s}",
                f"{rs.SAMPLES[s]['long']}, N = {m.sum():,}. r08: flat {p['lo']:g}-{p['hi']:g} GeV, alpha {p['alpha']:g}, "
                f"back to average by {p['hi'] + p['ease']:g} GeV. " + caption_common)
    fig, axes = rs.figure(ncols=3, size=(5.6, 4.6), sharey=False)
    for i, (ax, s) in enumerate(zip(axes, (1, 2, 3))):
        m = smp == s
        draw(ax, s, y[m], w05[m], w08[m], ps[str(s)], horn, legend=True)
        ax.legend(loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.5))
        rs.labels(ax, rs.E_TRUE if i == 2 else None, YLAB if i == 0 else None)
    fig.tight_layout()
    rs.save(fig, out, f"r08-scheme-{horn}-overview", "Three panels S1/S2/S3. " + caption_common)
    print("done", horn)
