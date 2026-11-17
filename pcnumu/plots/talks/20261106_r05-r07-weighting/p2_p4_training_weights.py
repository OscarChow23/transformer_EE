"""Training-population and training-weight plots p2, p3, p4 (SPEC.md, story steps 2-4).

  p2_train_spectra(_log)          raw trueE counts of the train split, stacked S1/S2/S3
  p3_weighted_alpha               pooled weighted training contribution: unweighted, r05 (a=0.05), r06 (a=0.01)
  p4_sample_contribution_r06_r07  per-sample weighted contribution / pooled total weight, r06 vs r07

Rows: the data file each model was trained on (input.json data_path, identical for r05/r06/r07), split column
== "train". r07 membership is rebuilt with the loader's subsample rule (copied from
transformer_ee/dataloader/load.py, pcnumu-loop branch) applied to the whole file before splitting.
Weights: transformer_ee.utils.weights.create_weighter (FlatSpectraWeights) with each model's own input.json
`weight` block, built on that model's train trueE, exactly as Normalized_Polars_Dataset_with_cache does
(the training logs report no NaN/null rows dropped, so no further filtering applies).
Cross-checks (stop on failure): train-row counts and r07 drop count against each train.log; r05/r06 ESS,
>10 GeV weight share and per-sample weight shares against the r07 preview metrics_<horn>.csv; r07 shares, ESS
and >10 GeV share against the r07 trial hypothesis (rounded values).

  python -I scripts/p2_p4_training_weights.py --horn fhc|rhc
"""
import argparse
import csv
import fcntl
import json
import os
import re
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common  # noqa: E402  (load_cards, sha256_file, CACHE lock dir; shared with the performance plots)
from common import rs  # noqa: E402

LOOP = "/data/nova/chow/transformerEE-pcnumu-loop"
sys.path.insert(0, LOOP)
from transformer_ee.utils.weights import create_weighter  # noqa: E402  (read-only import)
from pcnumu.prep.make_split_csv import splitmix64  # noqa: E402  (the loader's own hash)

PREVIEW = "/data/nova/chow/pcnumu-runs/r07_s1-subsample/preview/metrics_{horn}.csv"
EXPECTED_TRAIN = {"fhc": 2176323, "rhc": 2448068}
# r07 trial hypothesis (pcnumu-runs/model-comparison.md), percent, as rounded there
R07_HYP = {"fhc": dict(s1=25, s3=69, ess=22.3, gt10=35.7), "rhc": dict(s1=21, s3=75, ess=24.1, gt10=38.2)}
TAIL = 10.0
YLABEL = "Training weight fraction / 0.2 GeV"  # identical in p3 and p4
GREY = "#7f7f7f"  # unweighted reference (SPEC table: "unweighted (grey)")


def pct(x):
    """Percent with enough digits that small shares are not printed as 0.0%."""
    x = 100 * x
    return f"{x:.1f}%" if x == 0 or x >= 0.1 else f"{x:.2g}%"


def fail(msg):
    sys.exit(f"CHECK FAILED: {msg}")


def load_rows(cfg):
    sc = cfg["split_column"]
    df = (pl.scan_csv(cfg["data_path"])
          .select([cfg["target"][0], "sample", sc, "run", "subrun", "cycle", "evt"]).collect())
    return df, sc


def subsample_drop(df, sc, sub):
    """Copy of the loader's subsample rule (load.py, polars branch): True = row removed before splitting."""
    if "test" in sub["splits"]:
        fail("subsample touches test rows")
    run, srun, cyc, evt = (df[c].cast(pl.UInt64).to_numpy() for c in ("run", "subrun", "cycle", "evt"))
    key = ((run * np.uint64(1000) + srun) * np.uint64(1000) + cyc) * np.uint64(10**6) + evt
    with np.errstate(over="ignore"):
        u = splitmix64(key ^ np.uint64(sub["seed"])).astype(np.float64) / 2.0**64
    return ((df["sample"].to_numpy() == sub["sample"])
            & df[sc].is_in(sub["splits"]).to_numpy() & (u >= sub["keep"]))


def log_numbers(model_dir):
    """train size and subsample-dropped count printed by the loader in the run's train.log."""
    txt = open(os.path.join(os.path.dirname(model_dir), "train.log")).read() \
        if os.path.exists(os.path.join(os.path.dirname(model_dir), "train.log")) else ""
    n = re.findall(r"train indicies size:\s+(\d+)", txt)
    d = re.findall(r"subsample dropped:\s+(\d+)", txt)
    return (int(n[-1]) if n else None), (int(d[-1]) if d else None)


def summary(y, s, w):
    tot = w.sum()
    out = dict(n=int(y.size), ess=float(tot**2 / (w**2).sum() / y.size), gt10=float(w[y > TAIL].sum() / tot),
               mean_w=float(w.mean()))
    for k in (1, 2, 3):
        m = s == k
        out[f"s{k}_n"] = int(m.sum())
        out[f"s{k}_w"] = float(w[m].sum() / tot)
        out[f"s{k}_gt10"] = float(w[m & (y > TAIL)].sum() / tot)
        out[f"s{k}_n_gt10"] = int((m & (y > TAIL)).sum())
    return out


def compute(horn):
    cards = common.load_cards(horn)  # scorecard model_dirs + identical test fingerprint check
    cfgs, hashes = {}, {}
    for r, c in cards.items():
        p = os.path.join(c["model_dir"], "input.json")
        cfgs[r] = json.load(open(p))
        hashes[r] = common.sha256_file(p)
        if c["provenance"]["hashes"]["input.json"] != hashes[r]:
            fail(f"{r}: input.json sha256 differs from scorecard.json")
    if len({c["data_path"] for c in cfgs.values()}) != 1:
        fail("models use different data files")
    if any(c["weight"]["name"] != "FlatSpectraWeights" for c in cfgs.values()):
        fail("unexpected weighter")
    if "subsample" in cfgs["r05"] or "subsample" in cfgs["r06"] or "subsample" not in cfgs["r07"]:
        fail("subsample blocks not as expected (only r07)")

    df, sc = load_rows(cfgs["r05"])
    tname = cfgs["r05"]["target"][0]
    rows = {}
    for r in ("r05", "r06", "r07"):
        d = df
        if "subsample" in cfgs[r]:
            drop = subsample_drop(df, sc, cfgs[r]["subsample"])
            _, logged_drop = log_numbers(cards[r]["model_dir"])
            if logged_drop is not None and int(drop.sum()) != logged_drop:
                fail(f"{r}: dropped {int(drop.sum())} rows, train.log says {logged_drop}")
            d = df.filter(pl.Series(~drop))
        tr = d.filter(pl.col(sc) == "train")
        y = tr[tname].cast(pl.Float64).to_numpy()
        s = tr["sample"].to_numpy().astype(np.int64)
        if not np.all(np.isfinite(y)):
            fail("non-finite trueE in train rows")
        logged_n, _ = log_numbers(cards[r]["model_dir"])
        if logged_n is not None and y.size != logged_n:
            fail(f"{r}: {y.size} train rows, train.log says {logged_n}")
        if r != "r07" and y.size != EXPECTED_TRAIN[horn]:
            fail(f"{r}: {y.size} train rows, expected {EXPECTED_TRAIN[horn]}")
        w = np.asarray(create_weighter(cfgs[r], y).getweight(y), dtype=np.float64)
        rows[r] = dict(y=y, s=s, w=w, n_logged=logged_n, sum=summary(y, s, w))
    rows["unw"] = dict(y=rows["r05"]["y"], s=rows["r05"]["s"], w=np.ones_like(rows["r05"]["y"]),
                       sum=summary(rows["r05"]["y"], rows["r05"]["s"], np.ones_like(rows["r05"]["y"])))
    checks = cross_check(horn, rows)
    return cards, cfgs, hashes, rows, checks


def cross_check(horn, rows):
    lines = []
    prev = {r["option"]: r for r in csv.DictReader(open(PREVIEW.format(horn=horn)))
            if r["option"] in ("r05", "r06")}
    for r in ("r05", "r06"):
        S, P = rows[r]["sum"], prev[r]
        pairs = [("train_n", S["n"], float(P["train_n"])), ("ess_frac", S["ess"], float(P["ess_frac"])),
                 ("wfrac_gt10", S["gt10"], float(P["wfrac_gt10"])), ("mean_w", S["mean_w"], float(P["mean_w"]))]
        pairs += [(f"s{k}_w_share", S[f"s{k}_w"], float(P[f"s{k}_w_share"])) for k in (1, 2, 3)]
        for name, mine, ref in pairs:
            if not np.isclose(mine, ref, rtol=1e-4, atol=0):  # preview csv keeps 5 significant digits
                fail(f"{r} {name}: {mine:.6g} vs preview {ref:.6g}")
            lines.append(f"{horn} {r} {name}: {mine:.5g} (preview {ref:.5g})")
    S, H = rows["r07"]["sum"], R07_HYP[horn]
    for name, mine, ref, nd in (("s1_w_share", 100 * S["s1_w"], H["s1"], 0), ("s3_w_share", 100 * S["s3_w"], H["s3"], 0),
                                ("ess_frac", 100 * S["ess"], H["ess"], 1), ("wfrac_gt10", 100 * S["gt10"], H["gt10"], 1)):
        if round(mine, nd) != ref:
            fail(f"r07 {name}: {mine:.3f}% vs hypothesis {ref}%")
        lines.append(f"{horn} r07 {name}: {mine:.2f}% (hypothesis {ref}%)")
    lines.append(f"{horn} r07 train rows: {S['n']} (train.log {rows['r07']['n_logged']})")
    return lines


# ------------------------------------------------------------------------------------------- plotting
def hist(y, w, total):
    h, _ = np.histogram(y, bins=rs.SPEC_BINS, weights=w)
    return h / total if total else h


def step(ax, vals, **kw):
    b = np.asarray(rs.SPEC_BINS)
    return ax.stairs(vals, b, baseline=None, **kw)  # no vertical drop lines at the frame edges


def model_hash(cards, r):
    return os.path.basename(cards[r]["model_dir"]).replace("model_", "")[:8]


def p2(horn, rows, out, log):
    y, s = rows["r05"]["y"], rows["r05"]["s"]
    fig, ax = rs.figure()
    base = np.zeros(len(rs.SPEC_BINS) - 1)
    handles = []
    for k in (1, 2, 3):
        h, _ = np.histogram(y[s == k], bins=rs.SPEC_BINS)
        st = rs.SAMPLES[k]
        handles.append(ax.stairs(base + h, rs.SPEC_BINS, baseline=base, fill=True, facecolor=st["fill"],
                                 edgecolor=st["line"], linewidth=1.7 if k == 2 else 1.0, label=st["long"]))
        base = base + h
    ax.stairs(base, rs.SPEC_BINS, color="black", linewidth=1.2)
    ax.set_xlim(0, 10)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(10, base.max() * 30)
    else:
        ax.set_ylim(0, base.max() * 1.32)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    rs.labels(ax, rs.E_TRUE, "Events / 0.2 GeV")
    rs.tag(ax, f"{rs.HORN[horn]}  ·  training set")
    ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]], loc="upper right")
    S = rows["r05"]["sum"]
    n_gt = sum(S[f"s{k}_n_gt10"] for k in (1, 2, 3))
    per = ", ".join(f"S{k} {S[f's{k}_n_gt10']:,} ({pct(S[f's{k}_n_gt10'] / S[f's{k}_n'])} of S{k})"
                    for k in (1, 2, 3))
    name = "p2_train_spectra" + ("_log" if log else "")
    cap = f"""
{rs.HORN[horn]} training set (split column = train), raw event counts per 0.2 GeV of true neutrino energy, 0-10 GeV, stacked S1/S2/S3{' on a log scale' if log else ''}.
N = {S['n']:,} events: S1 {S['s1_n']:,} ({100 * S['s1_n'] / S['n']:.1f}%), S2 {S['s2_n']:,} ({100 * S['s2_n'] / S['n']:.1f}%), S3 {S['s3_n']:,} ({100 * S['s3_n'] / S['n']:.1f}%). No weights applied.
These are the train rows used by r05 and r06; r07 uses the same rows minus half of S1 (see p4).
Above 10 GeV (not drawn): {n_gt:,} events ({100 * n_gt / S['n']:.1f}%): {per}.
No uncertainties drawn (counts of the full training population).
The training population is dominated by S1 below 3 GeV, while S3 extends to high energy with a long tail.
"""
    rs.save(fig, out, name, cap)
    return name, dict(counts_per_bin={f"S{k}": np.histogram(y[s == k], bins=rs.SPEC_BINS)[0].tolist() for k in (1, 2, 3)},
                      n_above_10GeV={f"S{k}": S[f"s{k}_n_gt10"] for k in (1, 2, 3)})


def p3(horn, rows, cards, out):
    fig, ax = rs.figure()
    curves = {}
    unw = rows["unw"]
    curves["unweighted"] = hist(unw["y"], unw["w"], unw["w"].sum())
    step(ax, curves["unweighted"], color=GREY, linestyle=":", linewidth=1.8, label="Unweighted")
    for r in ("r05", "r06"):
        R = rows[r]
        curves[r] = hist(R["y"], R["w"], R["w"].sum())
        st = rs.MODELS[r]
        step(ax, curves[r], color=st["color"], linestyle=st["ls"], label=st["label"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(c.max() for c in curves.values()) * 1.2)
    rs.labels(ax, rs.E_TRUE, YLABEL)
    rs.tag(ax, f"{rs.HORN[horn]}  ·  training")
    ax.legend(loc="upper right")
    S5, S6, U = rows["r05"]["sum"], rows["r06"]["sum"], rows["unw"]["sum"]
    cap = f"""
{rs.HORN[horn]} training set, all samples pooled (N = {U['n']:,}): weighted training contribution per 0.2 GeV of true neutrino energy, 0-10 GeV.
y = sum of event weights in the bin / sum of weights over the whole training set; the drawn bin fractions sum to the 0-10 GeV weight share (1 minus the > 10 GeV share below), and the sum over all bins including > 10 GeV is 1. Each curve is a weighted distribution of the same training events (the event population is unchanged).
Weights: FlatSpectraWeights on true E (500 bins over the train range, w = 1/(n + alpha*max n), mean bin weight 1) as built in training: r05 alpha = 0.05 (model {model_hash(cards, 'r05')}), r06 alpha = 0.01 (model {model_hash(cards, 'r06')}); unweighted = event counts / N.
Weight share above 10 GeV (not drawn): unweighted {100 * U['gt10']:.1f}%, r05 {100 * S5['gt10']:.1f}%, r06 {100 * S6['gt10']:.1f}%.
Effective sample size (sum w)^2/sum w^2 as a fraction of N, a weight-concentration diagnostic: r05 {100 * S5['ess']:.1f}%, r06 {100 * S6['ess']:.1f}%.
No uncertainties drawn (exact weights of the full training set).
Lowering alpha from 0.05 to 0.01 moves training weight out of the 1-3 GeV peak and into the high-energy tail.
"""
    rs.save(fig, out, "p3_weighted_alpha", cap)
    return "p3_weighted_alpha", dict(fraction_per_bin={k: v.tolist() for k, v in curves.items()},
                                     counts_per_bin=np.histogram(unw["y"], bins=rs.SPEC_BINS)[0].tolist())


def p4(horn, rows, cards, out):
    fig, axes = rs.figure(ncols=3, size=(4.3, 4.6), sharey=True, gridspec_kw=dict(wspace=0))
    curves = {}
    for r in ("r06", "r07"):
        R = rows[r]
        curves[r] = {k: hist(R["y"][R["s"] == k], R["w"][R["s"] == k], R["w"].sum()) for k in (1, 2, 3)}
    ymax = max(c.max() for d in curves.values() for c in d.values()) * 1.32
    for i, k in enumerate((1, 2, 3)):
        ax = axes[i]
        for r in ("r06", "r07"):
            st = rs.MODELS[r]
            step(ax, curves[r][k], color=st["color"], linestyle=st["ls"], linewidth=2.2, label=st["label"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, ymax)
        if i < 2:
            ax.set_xticks([0, 2, 4, 6, 8])
        rs.labels(ax, rs.E_TRUE if i == 2 else None, YLABEL if i == 0 else None)
        rs.tag(ax, (f"{rs.HORN[horn]}  ·  " if i == 0 else "") + rs.SAMPLES[k]["short"])
        a6, a7 = rows["r06"]["sum"][f"s{k}_w"], rows["r07"]["sum"][f"s{k}_w"]
        ax.text(0.04, 0.85, f"r06 {100 * a6:.0f}%  →  r07 {100 * a7:.0f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=13.5)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.76))
    S6, S7 = rows["r06"]["sum"], rows["r07"]["sum"]
    share = "; ".join(f"S{k} {100 * S6[f's{k}_w']:.1f}% -> {100 * S7[f's{k}_w']:.1f}%" for k in (1, 2, 3))
    over = "; ".join(f"S{k} {pct(S6[f's{k}_gt10'])} -> {pct(S7[f's{k}_gt10'])}" for k in (1, 2, 3))
    cap = f"""
{rs.HORN[horn]} training set, weighted training contribution of each sample per 0.2 GeV of true neutrino energy (0-10 GeV), r06 vs r07; panels S1, S2, S3 share the y axis.
y = sum of the sample's weights in the bin / pooled sum of weights of all samples (incl. > 10 GeV) for that model; the drawn bin fractions sum to the sample's 0-10 GeV part, and the sum over all bins including > 10 GeV is the sample's weight share shown in the panel (r06 -> r07: {share}). Each curve is a weighted distribution of that model's training events.
r06 (model {model_hash(cards, 'r06')}): all {S6['n']:,} train events, FlatSpectraWeights alpha = 0.01, 500 bins. r07 (model {model_hash(cards, 'r07')}): {S7['n']:,} train events, S1 train/valid rows kept by the loader's hashed, energy-blind rule (keep 0.5, seed 20261104), pooled weights rebuilt on the kept rows with the same settings, which changes the S2 and S3 contributions too: r07 is not r06 with S1 simply scaled by 1/2.
Weight share above 10 GeV (not drawn), r06 -> r07: {over}; pooled {100 * S6['gt10']:.1f}% -> {100 * S7['gt10']:.1f}%. ESS/N {100 * S6['ess']:.1f}% -> {100 * S7['ess']:.1f}%.
No uncertainties drawn (exact weights of the full training sets).
Halving S1 lowers its share of the training weight and raises the S3 share.
"""
    rs.save(fig, out, "p4_sample_contribution_r06_r07", cap)
    return "p4_sample_contribution_r06_r07", dict(
        fraction_per_bin={r: {f"S{k}": v.tolist() for k, v in d.items()} for r, d in curves.items()},
        counts_per_bin={r: {f"S{k}": np.histogram(rows[r]["y"][rows[r]["s"] == k], bins=rs.SPEC_BINS)[0].tolist()
                            for k in (1, 2, 3)} for r in ("r06", "r07")})


def write_manifest(horn, cards, cfgs, hashes, rows, checks, entries):
    path = os.path.join(ROOT, horn, "manifest.json")
    os.makedirs(common.CACHE, exist_ok=True)
    base = dict(
        models={r: dict(scorecard=c["_path"], model_dir=c["model_dir"], input_json_sha256=hashes[r],
                        weight=cfgs[r]["weight"], subsample=cfgs[r].get("subsample")) for r, c in cards.items()},
        test_fingerprint_sha256=cards["r05"]["test_set"]["fingerprint_sha256"],
        data_path=cfgs["r05"]["data_path"], bins_GeV=rs.SPEC_BINS,
        train_rows={r: rows[r]["sum"]["n"] for r in ("r05", "r06", "r07")},
        summaries={r: rows[r]["sum"] for r in ("unw", "r05", "r06", "r07")},
        cross_checks=checks, script=os.path.abspath(sys.argv[0]))
    with open(os.path.join(common.CACHE, f"manifest_{horn}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.load(open(path)) if os.path.exists(path) else {}
        for pid, extra in entries.items():
            m[pid] = dict(base, **extra)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horn", required=True, choices=["fhc", "rhc"])
    a = ap.parse_args()
    rs.use()
    cards, cfgs, hashes, rows, checks = compute(a.horn)
    print("\n".join(checks))
    out = os.path.join(ROOT, a.horn)
    entries = {}
    for name, extra in (p2(a.horn, rows, out, False), p2(a.horn, rows, out, True),
                        p3(a.horn, rows, cards, out), p4(a.horn, rows, cards, out)):
        entries[name] = extra
    write_manifest(a.horn, cards, cfgs, hashes, rows, checks, entries)
    print("wrote", ", ".join(entries), "to", out)


if __name__ == "__main__":
    main()
