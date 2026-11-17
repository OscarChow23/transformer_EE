#!/usr/bin/env python
"""Pearson or Spearman correlations between model inputs: prong vs prong and slice vs slice (train split).

Two matrices, each with its own observation unit:
  * prong matrix: one observation = one real prong the model keeps (the first max_num_prongs in
    file order, as pl_dataset.__getitem__ does; padding never enters because it is not in the lists);
  * slice matrix: one observation = one slice (the scalar inputs, never repeated per prong).

Unweighted, on a uniform sample of the derived split CSV's train rows (before any r07 subsample).
r per pair uses only observations where both features are valid (pandas pairwise-complete; Spearman ranks
within that set);
invalid measurements become NaN by the rules in invalid_mask(), never zero.

  python -I pcnumu/studies/20261110_feature-importance/feature_correlation.py --config MODEL_DIR/input.json --raw RAW.csv.xz \
      --out-dir DIR [--same-as OTHER/input.json ...] [--n-slices 100000] [--seed 20261109] [--cache F] [--method spearman]
"""
import argparse
import json
import lzma
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

ID_COLS = ["run", "subrun", "cycle", "evt", "subevt"]
VALID_COLS = [f"png.bpf[{k}].IsValid" for k in range(3)]


def load_sample(cfg, raw_path, n, seed, cache=None):
    """Sampled train slices: vector lists (str), scalars, and the raw file's BPF IsValid lists."""
    if cache and os.path.exists(cache):
        df = pl.read_parquet(cache)
        return df, json.loads(df["_meta"][0])
    cols = ID_COLS + ["sample", cfg["split_column"]] + cfg["vector"] + cfg["scalar"]
    df = pl.read_csv(cfg["data_path"], columns=cols, infer_schema_length=0)  # all str, as written
    n_all = df.height
    df = df.filter(pl.col(cfg["split_column"]) == "train")
    assert df.null_count().sum_horizontal().item() == 0, "nulls in train rows (the loader would drop them)"
    n_train = df.height
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_train, size=min(n, n_train), replace=False))
    df = df[idx].with_columns(pl.Series("train_row", idx))

    with lzma.open(raw_path, "rb") as f:
        raw = pl.read_csv(f, columns=ID_COLS + ["png.dir.x"] + VALID_COLS, infer_schema_length=0)
    assert raw.select(ID_COLS).is_duplicated().sum() == 0, "event key not unique in raw file"
    out = df.join(raw.rename({"png.dir.x": "raw_dir_x"}), on=ID_COLS, how="left").sort("train_row")
    assert out.height == df.height and out["raw_dir_x"].null_count() == 0, "sampled slice missing in raw"
    assert (out["raw_dir_x"] == out["png.dir.x"]).all(), "raw and derived prong lists differ"
    meta = {"derived_rows": n_all, "train_rows": n_train, "seed": seed, "requested": n}
    out = out.drop("raw_dir_x").with_columns(pl.lit(json.dumps(meta)).alias("_meta"))
    if cache:
        out.write_parquet(cache)
    return out, meta


def prong_table(df, cfg):
    """One row per real prong kept by the model, 50 features as float (via float32, like the loader)."""
    vec, nmax = cfg["vector"], cfg["max_num_prongs"]
    # zero-prong slices have "" in every vector cell; the loader turns that into one all-zero prong
    # that is not masked. It is not a real prong, so it is left out here (counted in the summary).
    empty = (df["png.dir.x"] == "").to_numpy()
    assert all(((df[c] == "").to_numpy() == empty).all() for c in vec + VALID_COLS), "partial empty cells"
    d = df.filter(~pl.Series(empty))
    lens = d.select([pl.col(c).str.split(",").list.len().alias(c) for c in vec + VALID_COLS])
    n_png = lens[vec[0]].to_numpy().astype(np.int64)
    assert all((lens[c].to_numpy() == n_png).all() for c in lens.columns), "prong lists not aligned"
    t = d.select([pl.col(c).str.split(",").list.head(nmax) for c in vec + VALID_COLS] + ["sample"]).explode(
        vec + VALID_COLS)
    assert t.height == np.minimum(n_png, nmax).sum()
    x = pd.DataFrame(t.select([pl.col(c).cast(pl.Float32).cast(pl.Float64) for c in vec]).to_numpy(), columns=vec)
    valid3 = t.select(VALID_COLS).cast(pl.Int8).to_numpy().all(axis=1)
    info = {"zero_prong_slices": int(empty.sum()), "prongs_per_slice": np.bincount(n_png).tolist(),
            "slices_over_max": int((n_png > nmax).sum()), "prongs_dropped_by_max": int(np.clip(n_png - nmax, 0, None).sum()),
            "prongs": int(t.height), "prongs_by_sample": t["sample"].value_counts().sort("sample").rows()}
    return x, valid3, info


def invalid_mask(x, valid3):
    """Per-feature invalid flags and reason (checked on the FHC train sample, see README)."""
    reasons = {}
    for c in x.columns:
        m = {}
        if c.startswith("png.bpf["):
            m["bpf_not_all_three_valid"] = ~valid3  # helper writes 0 (-5 for startStopAngle) unless all 3 fits valid
        if c == "png.bpf[2].startStopAngle" or "avedEdxlast" in c:
            m["sentinel_-5"] = (x[c] == -5).to_numpy()
        if c in ("png.prox15", "png.prox30"):
            m["sentinel_-1"] = (x[c] == -1).to_numpy()
        reasons[c] = m
    return reasons


def quality(x, reasons, level):
    rows = []
    for c in x.columns:
        v = x[c].to_numpy()
        bad = np.zeros(len(v), bool)
        r = {"feature": c, "level": level, "n_total": len(v), "nonfinite": int((~np.isfinite(v)).sum())}
        bad |= ~np.isfinite(v)
        for k, m in reasons.get(c, {}).items():
            r[f"invalid_{k}"] = int((m & ~bad).sum())  # counted once, first reason wins
            bad |= m
        g = v[~bad]
        r.update(n_valid=len(g), zero_in_valid=int((g == 0).sum()), min=g.min(), max=g.max(), mean=g.mean(),
                 std=g.std(), n_unique=len(np.unique(g)), constant=bool(np.ptp(g) == 0))
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)


def correlate(x, reasons, method="pearson"):
    xv = x.copy()
    for c, m in reasons.items():
        for mm in m.values():
            xv.loc[mm, c] = np.nan
    ok = xv.notna().to_numpy().astype(np.float64)
    n = pd.DataFrame(ok.T @ ok, index=x.columns, columns=x.columns).astype(int)
    r = xv.corr(method=method, min_periods=3)  # pairwise complete: means/sums over the pair's valid rows
    const = [c for c in x.columns if xv[c].nunique() <= 1]
    for c in const:
        r.loc[c, :] = np.nan
        r.loc[:, c] = np.nan
    return xv, r, n, const


def check(xv, r, n, const, pairs, method="pearson"):
    a = r.to_numpy()
    assert np.allclose(a, a.T, equal_nan=True), "not symmetric"
    assert np.nanmax(np.abs(a)) <= 1 + 1e-12, "|r| > 1"
    d = np.diag(a)
    nonconst = [c not in const for c in r.columns]
    assert np.allclose(d[nonconst], 1), "diagonal not 1"
    assert np.isnan(d[[not k for k in nonconst]]).all()
    for i, j in pairs:  # independent library check on matched valid rows
        m = xv[[i, j]].dropna()
        assert len(m) == n.loc[i, j]
        ref = getattr(stats, f"{method}r")(m[i], m[j]).statistic
        assert abs(ref - r.loc[i, j]) < 1e-9, (i, j, ref, r.loc[i, j])


def group(name):
    for g in ("png.dir.", "png.start.", "png.cvnpart.", "png.bpf[0]", "png.bpf[1]", "png.bpf[2]"):
        if name.startswith(g):
            return g.rstrip(".").replace("png.", "")
    if name.startswith(("png.dedx", "png.activity", "png.prox")):
        return "PionReco dE/dx, activity, prox"
    return "prong shape, hits, energy"


def heatmap(r, title, caption, out, annotate=False, groups=False, label="Pearson r"):
    k = len(r)
    size = 17 if k > 20 else 7.5
    fig, ax = plt.subplots(figsize=(size, size * 0.93))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9e9e9e")
    im = ax.imshow(np.ma.masked_invalid(r.to_numpy()), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    for i, j in zip(*np.where(np.isnan(r.to_numpy()))):
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="xx", lw=0, edgecolor="k"))
    fs = 7.5 if k > 20 else 9
    ax.set_xticks(range(k), r.columns, rotation=90, fontsize=fs)
    ax.set_yticks(range(k), r.index, fontsize=fs)
    if annotate:
        for i in range(k):
            for j in range(k):
                v = r.iat[i, j]
                ax.text(j, i, "undef." if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="w" if abs(v) > .6 else "k")
    if groups:
        g = [group(c) for c in r.columns]
        cuts = [i for i in range(1, k) if g[i] != g[i - 1]]
        for c in cuts:
            ax.axhline(c - .5, color="k", lw=1.2)
            ax.axvline(c - .5, color="k", lw=1.2)
        starts = [0] + cuts
        ends = cuts + [k]
        for s, e in zip(starts, ends):
            ax.text((s + e - 1) / 2, -1.0, g[s], ha="center", va="bottom", fontsize=8, rotation=30,
                    rotation_mode="anchor", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label(f"{label}  (grey hatched = undefined, not zero)")
    ax.set_title(title, fontsize=11, pad=60 if groups else 12)
    fig.text(.01, .005, caption, fontsize=8, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170 if ext == "png" else None)
    plotted = im.get_array().filled(np.nan)
    plt.close(fig)
    return plotted


def top_pairs(r, n, k=None, col="pearson_r"):
    rows = [(r.index[i], r.columns[j], r.iat[i, j], n.iat[i, j])
            for i in range(len(r)) for j in range(i + 1, len(r))]
    t = pd.DataFrame(rows, columns=["feature_a", "feature_b", col, "n_valid_pairs"])
    t = t.reindex(t[col].abs().sort_values(ascending=False, na_position="last").index)
    return t.head(k) if k else t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="saved model input.json (feature order, data path)")
    p.add_argument("--same-as", nargs="*", default=[], help="other input.json files that must list the same inputs")
    p.add_argument("--raw", required=True, help="raw .csv.xz with png.bpf[k].IsValid")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--horn", default="FHC")
    p.add_argument("--n-slices", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--cache", help="parquet cache of the sampled slices (skips the 2-3 min read)")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    for o in a.same_as:
        oc = json.load(open(o))
        assert (oc["vector"], oc["scalar"], oc["max_num_prongs"]) == (cfg["vector"], cfg["scalar"], cfg["max_num_prongs"]), o
    assert len(cfg["vector"]) == 50 and len(cfg["scalar"]) == 9
    os.makedirs(a.out_dir, exist_ok=True)

    df, meta = load_sample(cfg, a.raw, a.n_slices, a.seed, a.cache)
    assert set(df[cfg["split_column"]].unique()) == {"train"}

    # prong matrix
    xp, valid3, info = prong_table(df, cfg)
    reasons = invalid_mask(xp, valid3)
    xpv, rp, np_, constp = correlate(xp, reasons, a.method)
    # slice matrix
    xs = pd.DataFrame(df.select([pl.col(c).cast(pl.Float64) for c in cfg["scalar"]]).to_numpy(), columns=cfg["scalar"])
    xsv, rs, ns, consts = correlate(xs, {}, a.method)
    check(xpv, rp, np_, constp, [("png.dedx15", "png.dedx30"), ("png.calE", "png.nhit"),
                                  ("png.bpf[2].scatt15", "png.prox15"), ("png.dir.x", "png.calE")], a.method)
    check(xsv, rs, ns, consts, [("calE", "nHit"), ("sample1", "sample3"), ("calE", "remPngCalE")], a.method)
    assert list(rp.columns) == cfg["vector"] and list(rs.columns) == cfg["scalar"]

    flags = df.select([pl.col(c).cast(pl.Int8) for c in ("sample1", "sample2", "sample3")]).to_numpy()
    summary = {**meta, "config": os.path.abspath(a.config), "same_as": a.same_as, "data_path": cfg["data_path"],
               "raw": a.raw, "horn": a.horn, "selected_slices": df.height,
               "slices_by_sample": df["sample"].value_counts().sort("sample").rows(),
               "sample_flags_one_hot": bool((flags.sum(1) == 1).all()), "max_num_prongs": cfg["max_num_prongs"],
               **info, "constant_prong_features": constp, "constant_slice_features": consts,
               "undefined_prong_cells": int(rp.isna().to_numpy().sum()),
               "undefined_slice_cells": int(rs.isna().to_numpy().sum()),
               "min_pair_count_prong": int(np_.to_numpy().min()), "min_pair_count_slice": int(ns.to_numpy().min()),
               "method": a.method, "vector": cfg["vector"], "scalar": cfg["scalar"]}
    json.dump(summary, open(f"{a.out_dir}/sample_summary.json", "w"), indent=1)

    for name, r, n in (("prong", rp, np_), ("slice", rs, ns)):
        r.to_csv(f"{a.out_dir}/corr_{name}.csv", float_format="%.6f")
        n.to_csv(f"{a.out_dir}/count_{name}.csv")
    pd.concat([quality(xp, reasons, "prong"), quality(xs, {}, "slice")]).to_csv(
        f"{a.out_dir}/feature_quality.csv", index=False, float_format="%.6g")
    top_pairs(rp, np_, 30, f"{a.method}_r").to_csv(f"{a.out_dir}/top_pairs_prong.csv", index=False, float_format="%.4f")
    top_pairs(rs, ns, col=f"{a.method}_r").to_csv(f"{a.out_dir}/top_pairs_slice.csv", index=False, float_format="%.4f")

    M = f"{a.method.capitalize()} r" if a.method == "pearson" else "Spearman rank r"
    pop = f"{a.horn}, train split only, unweighted, uniform sample (seed {a.seed}) of {df.height:,} slices"
    plotted = heatmap(
        rp, f"{M} between the 50 prong inputs: one observation = one real prong\n"
            f"{pop}; {info['prongs']:,} real prongs",
        "Each cell pairs a prong's value of one feature with the same prong's value of the other, pooled over "
        "slices. Real prongs only: padding and the loader's all-zero stand-in for zero-prong slices are excluded; "
        "slices with more prongs contribute more observations.\nInvalid values excluded per pair: BPF features "
        "unless all three BPF fits are valid; -5 in startStopAngle and avedEdxlast*; -1 in prox15/30. Zeros "
        "otherwise kept. Pooled over samples 1-3. Counts per cell: count_prong.csv.",
        f"{a.out_dir}/heatmap_prong", groups=True, label=M)
    assert np.allclose(plotted, rp.to_numpy(), equal_nan=True)
    plotted = heatmap(
        rs, f"{M} between the 9 slice inputs: one observation = one slice\n{pop}",
        "Each slice counts once (scalars are not repeated per prong). sample1-3 are one-hot, so their mutual r "
        "is negative by construction.\nPooled over samples 1-3. Counts per cell: count_slice.csv.",
        f"{a.out_dir}/heatmap_slice", annotate=True, label=M)
    assert np.allclose(plotted, rs.to_numpy(), equal_nan=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("vector", "scalar")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
