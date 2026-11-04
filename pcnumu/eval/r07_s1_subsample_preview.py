"""Preview r07 (fewer sample-1 training events, pooled FlatSpectraWeights) before any loader change.

S1 rows are kept by a hashed, seeded, energy-blind rule on the same packed event key as the split:
  keep S1 row  <=>  splitmix64(packed(run,subrun,cycle,evt) XOR sub_seed) / 2^64 < f
so the kept sets are nested (25% within 50% within 75%) and the selection is independent of the
split hash (different seed). S2 and S3 rows are never touched; test rows are never touched.

Options, all with the r05 weighter settings (FlatSpectraWeights, bins 500, alpha 0.05, no range):
  a  subsample S1 train rows, weighter rebuilt on the kept rows        (what training does today)
  b  subsample S1 train rows, weighter frozen at r05 (built on all train rows)
  c  keep all rows, r05 weighter, S1 weights multiplied by f            (expected weighted spectrum = b)
Reference rows: r05 (f=1, alpha 0.05) and r06 (f=1, alpha 0.01).

The loss is mean(w * l), so besides the shares we report the raw mean event weight (a gradient scale).
Valid uses the train weighter; we report its S1 weight share with valid kept whole and with valid
subsampled by the same rule (decision 4).

Usage (env transformer_ee):
  python -I pcnumu/eval/r07_s1_subsample_preview.py --horn fhc --config <r05 input.json> --out <dir>
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

SAMPLES = {1: "S1 contained", 2: "S2 periph. μ contained", 3: "S3 periph. μ uncontained"}
COLORS = {1: "#2a6fdb", 2: "#e8643c", 3: "#1aa37a", 0: "#222222"}
SPLIT_CODE = {"train": 0, "valid": 1, "test": 2}


def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def packed_key(run, sub, cyc, evt):
    return ((run * np.uint64(1000) + sub) * np.uint64(1000) + cyc) * np.uint64(10**6) + evt


def unit_hash(key, seed):
    with np.errstate(over="ignore"):
        return splitmix64(key ^ np.uint64(seed)).astype(np.float64) / 2.0**64


def load(cfg, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["y"], d["s"], d["sp"], d["key"]
    import polars as pl  # pylint: disable=C0415
    sc = cfg.get("split_column", "split")
    cols = [cfg["target"][0], "sample", sc, "run", "subrun", "cycle", "evt"]
    df = pl.scan_csv(cfg["data_path"]).select(cols).collect()
    y = df.get_column(cols[0]).to_numpy().astype(np.float64)
    s = df.get_column("sample").to_numpy().astype(np.int64)
    sp = df.get_column(sc).replace_strict(SPLIT_CODE, return_dtype=pl.Int8).to_numpy()
    key = packed_key(*(df.get_column(c).cast(pl.UInt64).to_numpy() for c in cols[3:]))
    np.savez(cache, y=y, s=s, sp=sp, key=key)
    return y, s, sp, key


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum() if w.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horn", required=True)
    ap.add_argument("--config", required=True, help="r05 input.json (data_path, target, weight)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--fractions", default="0.25,0.5,0.75")
    ap.add_argument("--sub-seed", type=int, default=20261104)
    ap.add_argument("--split-seed", type=int, default=20261029)
    ap.add_argument("--tail", type=float, default=10.0)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    from transformer_ee.utils.weights import FlatSpectraWeights  # pylint: disable=C0415
    import matplotlib  # pylint: disable=C0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=C0415

    cfg = json.load(open(a.config))
    kw = dict(cfg["weight"]["kwargs"])
    os.makedirs(a.out, exist_ok=True)
    y, s, sp, key = load(cfg, os.path.join(a.out, f"rows_{a.horn}.npz"))
    fracs = [float(x) for x in a.fractions.split(",")]

    # sanity: the cached split equals the split hash; selection is energy-blind
    u_split = unit_hash(key, a.split_seed)
    expect = np.where(u_split < 0.20, 2, np.where(u_split < 0.24, 1, 0))
    assert np.array_equal(expect, sp), "split column does not match the split hash"
    u = unit_hash(key, a.sub_seed)
    tr, va = sp == 0, sp == 1
    s1tr = tr & (s == 1)
    checks = {"rows": int(y.size), "train": int(tr.sum()), "valid": int(va.sum()), "test": int((sp == 2).sum()),
              "split_hash_matches_column": True, "sub_seed": a.sub_seed,
              "corr(u_sub, u_split) all rows": float(np.corrcoef(u, u_split)[0, 1]),
              "corr(u_sub, trueE) S1 train": float(np.corrcoef(u[s1tr], y[s1tr])[0, 1])}
    for f in fracs:
        k = s1tr & (u < f)
        checks[f"S1 train kept at f={f:g}"] = {"n": int(k.sum()), "frac": float(k.sum() / s1tr.sum()),
                                              "mean trueE kept/all": float(y[k].mean() / y[s1tr].mean())}

    def keepmask(f, rows):
        return rows & ((s != 1) | (u < f))

    W05 = FlatSpectraWeights(y[tr], **kw)
    W01 = FlatSpectraWeights(y[tr], **{**kw, "alpha": 0.01})

    def option(name, f):
        """Return (mask of train rows used, their weights, weighter used for valid, valid S1 factor)."""
        if name == "r05":
            return tr, W05.getweight(y[tr]), W05, 1.0
        if name == "r06":
            return tr, W01.getweight(y[tr]), W01, 1.0
        if name == "a":
            m = keepmask(f, tr)
            Wa = FlatSpectraWeights(y[m], **kw)
            return m, Wa.getweight(y[m]), Wa, 1.0
        if name == "b":
            m = keepmask(f, tr)
            return m, W05.getweight(y[m]), W05, 1.0
        if name == "c":
            w = W05.getweight(y[tr])
            w[s[tr] == 1] *= f
            return tr, w, W05, f
        raise ValueError(name)

    cases = [("r05", 1.0), ("r06", 1.0)] + [(o, f) for o in "abc" for f in fracs]
    rows, spectra = [], {}
    tv_edges = np.arange(0, 30.0001, 0.1)
    ref = {}
    for name, f in cases:
        m, w, Wv, cf = option(name, f)
        yy, ss = y[m], s[m]
        W = w.sum()
        r = {"horn": a.horn, "option": name, "alpha": 0.01 if name == "r06" else kw["alpha"], "s1_kept": f,
             "train_n": int(m.sum()), "mean_w": w.mean(), "ess_frac": ess(w) / w.size,
             f"wfrac_gt{a.tail:g}": w[yy > a.tail].sum() / W}
        for k in SAMPLES:
            mk = ss == k
            r[f"s{k}_ev_share"] = mk.sum() / m.sum()
            r[f"s{k}_w_share"] = w[mk].sum() / W
            r[f"s{k}_ess"] = ess(w[mk])
        mk3 = ss == 3
        r[f"s3_wfrac_gt{a.tail:g}_within_s3"] = w[mk3 & (yy > a.tail)].sum() / w[mk3].sum()
        # how much the weighted spectra moved vs r05 (total variation of area-normalized hist, 0-30 GeV)
        hists = {}
        for k in (0, 1, 2, 3):
            mk = np.ones_like(ss, bool) if k == 0 else ss == k
            h, _ = np.histogram(yy[mk], tv_edges, weights=w[mk])
            hists[k] = h
        if name == "r05":
            ref = hists
        r["tv_all_vs_r05"] = 0.5 * np.abs(hists[0] / hists[0].sum() - ref[0] / ref[0].sum()).sum()
        r["tv_s3_vs_r05"] = 0.5 * np.abs(hists[3] / hists[3].sum() - ref[3] / ref[3].sum()).sum()
        # valid, weighted with the train weighter (as load.py does); c also scales valid S1 by f
        for vname, vm in (("validfull", va), ("validsub", va if name in ("c", "r05", "r06") else keepmask(f, va))):
            wv = Wv.getweight(y[vm])
            wv[s[vm] == 1] *= cf  # c: the S1 factor is part of the weighter, so valid gets it too
            r[f"{vname}_n"] = int(vm.sum())
            r[f"{vname}_s1_w_share"] = wv[s[vm] == 1].sum() / wv.sum()
        rows.append(r)
        spectra[(name, f)] = (yy, ss, w)
        print(f"{a.horn} {name} f={f:g}: N={r['train_n']:,} S1/S2/S3 w-share "
              f"{r['s1_w_share']:.0%}/{r['s2_w_share']:.0%}/{r['s3_w_share']:.0%} ESS {r['ess_frac']:.1%} "
              f">{a.tail:g} GeV {r[f'wfrac_gt{a.tail:g}']:.1%} mean_w {r['mean_w']:.3f}", flush=True)

    with open(os.path.join(a.out, f"metrics_{a.horn}.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    json.dump({"config": os.path.abspath(a.config), "weight_kwargs": kw, "fractions": fracs, "checks": checks},
              open(os.path.join(a.out, f"checks_{a.horn}.json"), "w"), indent=1)

    # 1) weighted spectra per sample: rows = option, cols = kept fraction; r05 dashed
    for tag, edges, logy in (("lin", np.arange(0, 10.0001, 0.1), False), ("log", np.arange(0, 30.0001, 0.25), True)):
        fig, axs = plt.subplots(3, len(fracs), figsize=(5 * len(fracs), 10), sharex=True, sharey=True)
        yr, sr, wr_ = spectra[("r05", 1.0)]
        for i, o in enumerate("abc"):
            for j, f in enumerate(fracs):
                ax = axs[i, j]
                yy, ss, w = spectra[(o, f)]
                for k in (1, 2, 3, 0):
                    for (Y, S, Wt, ls, lw, lab) in ((yr, sr, wr_, "--", 0.9, None), (yy, ss, w, "-", 1.5, True)):
                        mk = np.ones_like(S, bool) if k == 0 else S == k
                        h, _ = np.histogram(Y[mk], edges, weights=Wt[mk])
                        ax.stairs(h / Wt.sum(), edges, color=COLORS[k], ls=ls, lw=lw,
                                  label=(SAMPLES.get(k, "All") if lab and i == 0 and j == 0 else None))
                r = next(r for r in rows if r["option"] == o and r["s1_kept"] == f)
                ax.set_title(f"option {o}, S1 kept {f:.0%}: w-share S1/S2/S3 {r['s1_w_share']:.0%}/"
                             f"{r['s2_w_share']:.0%}/{r['s3_w_share']:.0%}\nESS {r['ess_frac']:.0%}, "
                             f">{a.tail:g} GeV {r[f'wfrac_gt{a.tail:g}']:.1%}, mean w {r['mean_w']:.2f}", fontsize=9)
                ax.grid(alpha=0.25)
                if logy:
                    ax.set_yscale("log")
                    ax.axvline(a.tail, color="0.5", ls=":", lw=1)
        for ax in axs[-1]:
            ax.set_xlabel("trueE [GeV]")
        for ax in axs[:, 0]:
            ax.set_ylabel(f"share of total train weight / {edges[1] - edges[0]:g} GeV")
        axs[0, 0].legend(fontsize=8, title="solid: option   dashed: r05", title_fontsize=8)
        fig.suptitle(f"{a.horn.upper()} r07 preview: weighted train spectra per sample (FlatSpectraWeights "
                     f"bins {kw['bins']}, α {kw['alpha']})\na: weights recomputed · b: r05 weights frozen · "
                     f"c: all events, S1 weight × f", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"spectra_{tag}_{a.horn}.png"), dpi=110)
        plt.close(fig)

    # 2) summary metrics vs kept fraction
    keys = [("s1_w_share", "S1 weight share"), (f"wfrac_gt{a.tail:g}", f"weight share > {a.tail:g} GeV"),
            ("ess_frac", "ESS / N"), ("s3_ess", "S3 effective N"), ("mean_w", "mean event weight (raw)"),
            ("tv_s3_vs_r05", "S3 shape change vs r05 (TV)")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ls_ = {"a": "o-", "b": "s--", "c": "^:"}
    for ax, (kk, lab) in zip(axs.flat, keys):
        for o in "abc":
            xs = fracs + [1.0]
            ys = [next(r for r in rows if r["option"] == o and r["s1_kept"] == f)[kk] for f in fracs]
            ys.append(next(r for r in rows if r["option"] == "r05")[kk])
            ax.plot(xs, ys, ls_[o], label=f"option {o}")
        ax.axhline(next(r for r in rows if r["option"] == "r06")[kk], color="crimson", lw=0.8, ls=":", label="r06 (α 0.01)")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("S1 train fraction kept (1 = r05)")
        ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{a.horn.upper()} r07 preview: summary vs S1 kept fraction", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"summary_{a.horn}.png"), dpi=110)
    plt.close(fig)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
