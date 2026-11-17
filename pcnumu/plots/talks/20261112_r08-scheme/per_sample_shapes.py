"""Per-sample trueE shapes of the training rows: unweighted vs pooled FlatSpectraWeights (r05 a=0.05, r06 a=0.01).
Each curve is normalised to unit area over the displayed range within its own sample.
usage: python -I per_sample_shapes.py HORN OUT_PNG"""
import json
import sys

sys.path.insert(0, "/data/nova/chow/transformerEE-pcnumu-loop")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from transformer_ee.utils.weights import create_weighter  # noqa: E402

horn, out = sys.argv[1], sys.argv[2]
cfgs = {
    "r05": f"/data/nova/chow/transformerEE/runs/pc-numu/{horn}/pcnumu_{horn}_r05_v3data-allsamples_v3features_20261029/"
           f"pcnumu_{horn}_r05_v3data-allsamples_v3features.json",
    "r06": f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/{horn}/effective_config.json",
}
cfgs = {k: json.load(open(v)) for k, v in cfgs.items()}
c = cfgs["r06"]
assert cfgs["r05"]["data_path"] == c["data_path"] and cfgs["r05"]["split_column"] == c["split_column"]
train = (pl.scan_csv(c["data_path"], infer_schema_length=0).filter(pl.col(c["split_column"]) == "train")
         .select(["sample", "trueE"]).collect())
y = train["trueE"].cast(pl.Float64).to_numpy()
smp = train["sample"].to_numpy().astype(str)
w = {"Unweighted": np.ones_like(y)}
for k, lab in (("r05", "Pooled, alpha 0.05 (r05)"), ("r06", "Pooled, alpha 0.01 (r06)")):
    assert cfgs[k]["weight"]["name"] == "FlatSpectraWeights"
    w[lab] = create_weighter(cfgs[k], y).getweight(y)
style = {"Unweighted": dict(color="#6b6b6b", ls="--"), "Pooled, alpha 0.05 (r05)": dict(color="#2b2b2b", ls="-"),
         "Pooled, alpha 0.01 (r06)": dict(color="#2a78d6", ls="-")}
names = {"1": "S1 contained", "2": "S2 peripheral, μ contained", "3": "S3 peripheral, μ uncontained"}
edges = np.arange(0, 10 + 1e-9, 0.2)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
lines = []
for ax, k in zip(axes, ("1", "2", "3")):
    m = smp == k
    for lab, ww in w.items():
        h, _ = np.histogram(y[m], edges, weights=ww[m])
        frac_out = 1 - h.sum() / ww[m].sum()
        lines.append(f"{horn} S{k} {lab}: weight above 10 GeV {frac_out:.1%}")
        ax.stairs(h / h.sum(), edges, lw=2, label=lab, **style[lab])
    ax.set_title(f"{horn.upper()} · {names[k]}  (N = {m.sum():,})", fontsize=10)
    ax.set_xlabel("true neutrino energy (GeV)")
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("fraction of the sample's weight / 0.2 GeV\n(unit area over 0–10 GeV)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Per-sample true-energy shape under the same pooled weights", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("\n".join(lines))
print("wrote", out)
