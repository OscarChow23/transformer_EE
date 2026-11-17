#!/usr/bin/env python
"""Permutation importance of every model input, for one frozen TransformerEE model.

    python -I permutation_importance.py --model-dir MODEL_DIR --out-dir OUT [--seeds 5]

Shuffles one input at a time across the validation events (the model is not retrained) and
records how much bias, resolution and ellipse fraction change against the unshuffled baseline.
Before any shuffle, an unshuffled pass over the test rows must reproduce the model's result.csv
and scorecard.json, or the run stops. Plan and decisions D1-D22:
pcnumu-analysis/plans/feature-importance-permutation-grill-handoff-20261110.md

Writes importance.csv (one row per entry, seed, target, region, sample, metric), figures
(importance_{resolution,bias,ellipse_migration}_sample{1,2,3,all}.png/pdf) and README.md into OUT. If OUT/importance.csv exists, only the figures are redrawn.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(os.path.dirname(HERE)))]  # python -I drops the cwd
from feature_correlation import group  # noqa: E402
from pcnumu.core import style  # noqa: E402
from pcnumu.core.data import ID_COLUMNS, code_version  # noqa: E402
from pcnumu.core.metrics import SAMPLES, WINDOW, ellipse_axes, ellipse_fraction, stats  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set by pcnumu.core.style)

STYLE = style.SAMPLE_STYLE
from transformer_ee.inference.pred import Predictor  # noqa: E402

# Regions are cuts on true trueE [GeV], applied to both targets. The last is the scorecard's window.
REGIONS = {"0.5-3": lambda e: (e >= 0.5) & (e < 3), ">=3": lambda e: e >= 3,
           "0.5-6": lambda e: (e >= WINDOW[0]) & (e <= WINDOW[1])}


def load(cfg, split):
    """Rows of one split in file order, every column as text (as scorecard.py reads them)."""
    cols = ID_COLUMNS + ["sample"] + cfg["target"] + cfg["scalar"] + cfg["vector"]
    return (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(cfg["split_column"]) == split).select(cols).collect())


def truth(df, cfg):
    return {t: df[t].cast(pl.Float64).to_numpy() for t in cfg["target"]}, df["sample"].to_numpy().astype(str)


def tensors(model_dir, df, cfg):
    """Model plus its inputs on the GPU, built once by the real inference loader (Predictor:
    checkpoint, trainset_stat normalisation, padding, mask with True = PAD)."""
    cols = df.select(cfg["vector"] + [pl.col(cfg["scalar"]).cast(pl.Float64)])
    p = Predictor(model_dir, pd.DataFrame(cols.to_dict(as_series=False)), num_workers=4)  # no pyarrow here
    v, s, m = (torch.cat(x).to(p.device) for x in zip(*[b[:3] for b in p.dataloader]))
    return p.net, v, s, m


@torch.inference_mode()
def infer(net, v, s, m, bs):
    out = [net(v[i:i + bs], s[i:i + bs], m[i:i + bs]) for i in range(0, len(v), bs)]
    return torch.cat(out).cpu().numpy().astype(np.float64)


def metrics(pred, true, smp, axes):
    """{(target, region, sample, metric): value}, definitions as scorecard.py."""
    t0, t1 = true
    r = {t: (pred[:, i] - true[t]) / true[t] for i, t in enumerate(true)}
    out = {}
    for reg, sel in REGIONS.items():
        for k, _ in SAMPLES:
            m = sel(true[t0]) & ((smp == k) | (k == "all"))
            for t in true:
                st = stats(r[t][m])
                out[t, reg, k, "bias"], out[t, reg, k, "resolution"] = float(st["bias"]), float(st["res"])
            out["both", reg, k, "ellipse_fraction"] = ellipse_fraction(  # unbinned = the scorecard's headline
                100 * r[t0][m], 100 * r[t1][m], axes["a_percent"], axes["b_percent"])["unbinned_fraction"]
            if reg == ">=3":  # share of true >= 3 GeV events reconstructed below 3 GeV
                out[t0, reg, k, "migration_below_3"] = float(np.mean(pred[m, 0] < 3))
    return out


def check_test(model_dir, cfg):
    """Unshuffled test pass must reproduce result.csv and scorecard.json (plan D13)."""
    sc = json.load(open(os.path.join(os.path.dirname(model_dir), "scorecard", "scorecard.json")))
    assert os.path.realpath(sc["model_dir"]) == os.path.realpath(model_dir), "scorecard is for another model"
    df, res = load(cfg, "test"), pl.read_csv(os.path.join(model_dir, "result.csv"))
    fp = hashlib.sha256(df.select(ID_COLUMNS + ["sample"]).write_csv().encode()).hexdigest()
    assert fp == sc["test_set"]["fingerprint_sha256"], "test rows differ from the scorecard's"
    assert df.height == res.height, f"{df.height} test rows vs {res.height} in result.csv"
    true, smp = truth(df, cfg)
    pred = infer(*tensors(model_dir, df, cfg), cfg["batch_size_test"])
    dmax = 0.0
    for i, t in enumerate(true):
        assert np.isclose(res[f"true_{t}"].to_numpy(), true[t], rtol=1e-5, atol=1e-6).all(), f"{t}: rows not aligned"
        dmax = max(dmax, float(np.abs(pred[:, i] - res[f"pred_{t}"].to_numpy()).max()))
    assert dmax < 1e-4, f"max |pred - result.csv| = {dmax:.3g} GeV"
    axes = ellipse_axes(fp)
    got = metrics(pred, true, smp, axes)
    diffs = [abs(got["both", "0.5-6", k, "ellipse_fraction"] - sc["ellipse"]["by_sample"][k]["fraction"]) for k, _ in SAMPLES]
    diffs += [abs(got[t, "0.5-6", k, q] - sc["metrics"][t][k][q])
              for t in true for k, _ in SAMPLES for q in ("bias", "resolution")]
    assert max(diffs) < 2e-4, f"scorecard metrics differ by up to {max(diffs):.3g}"
    return axes, fp, dmax, max(diffs)


def entries(cfg):
    """(name, 's' or 'v', column indices): slice inputs, the sample flags as one entry, prong inputs."""
    flags = [i for i, n in enumerate(cfg["scalar"]) if n.startswith("sample")]
    return ([(n, "s", [i]) for i, n in enumerate(cfg["scalar"]) if i not in flags]
            + [("sample1+2+3", "s", flags)] + [(n, "v", [j]) for j, n in enumerate(cfg["vector"])])


def shuffled(v, s, m, kind, cols, seed):
    """Copy of the inputs with one entry's values reassigned at random. The mask is never changed."""
    g = torch.Generator().manual_seed(seed)
    if kind == "s":  # one permutation of the events, applied to every column of the entry
        s = s.clone()
        s[:, cols] = s[torch.randperm(len(s), generator=g).to(s.device)][:, cols]
        return v, s
    # shortcut: zero-prong slices' fake prong (loader [0]) is pooled like a real prong; ~0.06% of values,
    # revisit if the loader's empty-slice handling changes
    v2, real = v.clone(), ~m
    col = v2[..., cols[0]]  # view: one input across all prong slots
    col[real] = col[real][torch.randperm(int(real.sum()), generator=g).to(v.device)]
    assert torch.equal(col[real].sort().values, v[..., cols[0]][real].sort().values), "values not preserved"
    assert torch.equal(col[m], v[..., cols[0]][m]), "padding changed"
    return v2, s


def run(net, v, s, m, true, smp, axes, cfg, seeds):
    """CSV rows: baseline first, then every entry x seed."""
    bs = cfg["batch_size_test"]
    base = metrics(infer(net, v, s, m, bs), true, smp, axes)
    rows = [("baseline", -1) + k + (x, x) for k, x in base.items()]
    for i, (name, kind, cols) in enumerate(entries(cfg)):
        for seed in range(seeds):
            got = metrics(infer(net, *shuffled(v, s, m, kind, cols, 1000 * i + seed), m, bs), true, smp, axes)
            rows += [(name, seed) + k + (x, base[k]) for k, x in got.items()]
        print(f"{i + 1}/{len(entries(cfg))} {name}", flush=True)
    return rows


B, O, G, INKC = STYLE["1"]["color"], STYLE["2"]["color"], STYLE["3"]["color"], "#2b2b2b"
LO, HI = "0.5-3", ">=3"
REG = {LO: "true trueE 0.5–3 GeV", HI: "true trueE ≥ 3 GeV"}
# figure -> (y label, [(target, metric, region, legend name, colour, marker, filled, sign, relative, baseline format)])
FIGURES = {
    "resolution": ("resolution change [%]  ↑ worse", [
        (t, "resolution", r, f"{t} resolution", c, mk, r == LO, 1, True, lambda v: f"{100 * v:.1f}%")
        for t, c, mk in (("trueE", B, "o"), ("trueLepE", O, "s")) for r in (LO, HI)]),
    "bias": ("bias shift [pp]", [
        (t, "bias", r, f"{t} bias", c, mk, r == LO, 1, False, lambda v: f"{100 * v:+.1f}%")
        for t, c, mk in (("trueE", B, "o"), ("trueLepE", O, "s")) for r in (LO, HI)]),
    "ellipse_migration": ("event share change [pp]  ↑ worse", [
        ("both", "ellipse_fraction", r, "share outside ellipse", G, "^", r == LO, -1, False,
         lambda v: f"{100 * (1 - v):.0f}%") for r in (LO, HI)] + [
        ("trueE", "migration_below_3", HI, "share of true trueE ≥ 3 GeV reconstructed < 3 GeV", INKC, "D", False, 1,
         False, lambda v: f"{100 * v:.1f}%")]),
}


def band_name(g, members):
    """Group band label in config terms: the shared prefix, or first … last input."""
    if g == "slice":
        return "scalar (slice)"
    if g in ("dir", "start", "cvnpart", "bpf[0]", "bpf[1]", "bpf[2]"):
        return f"png.{g}.*"
    return f"{members[0]} … {members[-1]}"


def plot(out, model_name):
    """One figure per sample and metric family, inputs along x (D23). Legend values in brackets are
    the unshuffled baseline; point = mean of the seeds, bar = lowest to highest (see README)."""
    d = pd.read_csv(os.path.join(out, "importance.csv"), dtype={"sample": str})
    base, d = d[d["entry"] == "baseline"], d[d["entry"] != "baseline"]
    names = list(dict.fromkeys(d["entry"]))
    groups = [group(n) if n.startswith("png.") else "slice" for n in names]
    labels = ["sample1+sample2+sample3" if n.startswith("sample") else n for n in names]
    x = np.arange(len(names))
    edges = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]] + [len(groups)]
    run = model_name.split("_fhc_")[-1].split("_rhc_")[-1].split("_")[0] + (" FHC" if "_fhc_" in model_name else " RHC")
    for k, sample_name in SAMPLES:
        b0 = base[base["sample"] == k].set_index(["target", "metric", "region"])["value"]
        for fig_name, (ylabel, series) in FIGURES.items():
            fig, ax = plt.subplots(figsize=(18, 6.5))
            for n, (a, b) in enumerate(zip(edges, edges[1:])):
                if n % 2:
                    ax.axvspan(a - 0.5, b - 0.5, color="#f0efeb", lw=0, zorder=0)
                ax.text((a + b - 1) / 2, 1.01, band_name(groups[a], names[a:b]), transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=8, color="#6b6b6b")
            for t, m, reg, name, c, mk, filled, sign, rel, fmt in series:
                q = d[(d["sample"] == k) & (d["target"] == t) & (d["metric"] == m) & (d["region"] == reg)]
                y = sign * 100 * (q["value"] - q["baseline"]) / (q["baseline"] if rel else 1)
                s = y.groupby(q["entry"], sort=False).agg(["mean", "min", "max"]).loc[names]
                ax.plot(x, s["mean"], ls="-" if filled else "--", lw=0.7, color=c, alpha=0.35)  # reading aid only
                ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], fmt=mk, ms=5,
                            color=c, mfc=c if filled else "white", capsize=2, lw=1,
                            label=f"{name}{'' if m == 'migration_below_3' else ', ' + REG[reg]}   (unshuffled {fmt(b0[t, m, reg])})")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel(ylabel)
            style.labels(ax)
            ax.legend(frameon=False, fontsize=8.5, title=f"one input shuffled at a time · point = mean of {d['seed'].nunique()} shuffles, bar = range", title_fontsize=8, alignment="left")
            ax.set_xticks(x, labels, rotation=90, fontsize=8)
            ax.set_xlim(-0.5, len(names) - 0.5)
            ax.set_title(f"{sample_name} · {run}, validation", fontsize=11, pad=18)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(out, f"importance_{fig_name}_sample{k}.{ext}"), dpi=150)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="from the scorecard or leaderboard; never a glob")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "input.json")))
    if not os.path.exists(os.path.join(a.out_dir, "importance.csv")):
        os.makedirs(a.out_dir, exist_ok=True)
        t0 = time.time()
        axes, fp, dmax, dmetric = check_test(a.model_dir, cfg)
        print(f"test check passed: max |dpred| {dmax:.3g} GeV, max metric difference {dmetric:.3g}", flush=True)
        df = load(cfg, "valid")
        true, smp = truth(df, cfg)
        rows = run(*tensors(a.model_dir, df, cfg), true, smp, axes, cfg, a.seeds)
        with open(os.path.join(a.out_dir, "importance.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry", "seed", "target", "region", "sample", "metric", "value", "baseline"])
            w.writerows(rows)
        with open(os.path.join(a.out_dir, "README.md"), "w") as f:
            f.write(f"""# Permutation importance: {cfg['model_phys_name']}

- command: `{' '.join(sys.argv)}`
- model: `{a.model_dir}`
- code: {code_version()}
- data: `{cfg['data_path']}`
- shuffled on: validation split, {df.height} events; {len(entries(cfg))} entries x {a.seeds} seeds
  (torch.Generator seed = 1000 x entry index + seed)
- test check: baseline reproduces scorecard ✓ ({fp[:16]}; max |dpred| {dmax:.3g} GeV;
  max metric difference {dmetric:.3g})
- ellipse: a = {axes['a_percent']}%, b = {axes['b_percent']}%, calibrated on the test set's 0.5-6 GeV window,
  so every ellipse number here (validation, other regions) is provisional
- run time: {(time.time() - t0) / 60:.1f} min
""")
    plot(a.out_dir, cfg["model_phys_name"])


if __name__ == "__main__":
    main()
