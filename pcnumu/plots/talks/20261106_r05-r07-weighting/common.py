"""Shared loading/checking for the r05/r06/r07 performance plots (p1, p5, p6, p56, p7).

Model dirs come only from each run's scorecard.json `model_dir`. Test rows are aligned with
result.csv exactly as pcnumu/eval/scorecard.py load_test_predictions() does (copied here, read-only
source: transformerEE-pcnumu-loop/pcnumu/eval/scorecard.py), plus the cross-model row check of
SPEC rule 10. Nothing in the run folders is written.
"""
import fcntl
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "style"))
import rootstyle as rs  # noqa: E402

RUNS = "/data/nova/chow/pcnumu-runs"
SCORECARDS = {  # run id -> scorecard dir per horn (pcnumu-runs/<run>/<horn>/scorecard)
    "r05": "r05_v3data-allsamples_v3features",
    "r06": "r06_flat-alpha001",
    "r07": "r07_s1-subsample",
}
MODEL_DESC = {"r05": "flat, α = 0.05", "r06": "flat, α = 0.01", "r07": "flat, α = 0.01, ½ S1"}
EXPECTED_FP = {"fhc": "20be926a", "rhc": "b2ac8189"}
ID_COLUMNS = ["run", "subrun", "cycle", "evt", "subevt"]
TAG_SAMPLE = {1: "S1 (contained)", 2: "S2 (peripheral, $\\mu$ contained)",
              3: "S3 (peripheral, $\\mu$ uncontained)"}
CACHE = os.environ.get("TALKPLOTS_CACHE", "/tmp/claude-32382/--data-nova/af4f2585-b866-4bce-bd4c-bc6fbcaf249f/scratchpad/cache")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scorecard_path(run, horn):
    p = os.path.join(RUNS, SCORECARDS[run], horn, "scorecard", "scorecard.json")
    if "smoke" in p:
        sys.exit("refusing smoke path")
    return p


def load_cards(horn):
    cards = {}
    for run in SCORECARDS:
        p = scorecard_path(run, horn)
        c = json.load(open(p))
        if "smoke" in c["model_dir"]:
            sys.exit(f"{run}: smoke model_dir")
        c["_path"] = p
        pk = os.path.join(os.path.dirname(p), "peak_metrics.json")
        c["_peak"] = json.load(open(pk)) if os.path.exists(pk) else None
        cards[run] = c
    fps = {r: c["test_set"]["fingerprint_sha256"] for r, c in cards.items()}
    if len(set(fps.values())) != 1 or not next(iter(fps.values())).startswith(EXPECTED_FP[horn]):
        sys.exit(f"test fingerprints differ or unexpected: {fps}")
    for r, c in cards.items():
        if c["_peak"] is not None and (c["_peak"]["model_dir"] != c["model_dir"]
                                       or c["_peak"]["test_fingerprint"] != fps[r]):
            sys.exit(f"{r}: peak_metrics.json model_dir/fingerprint differs from scorecard.json")
    return cards


def _load_one(cfg, model_dir):
    """Copy of scorecard.load_test_predictions(), also returning event IDs (for the row check)."""
    sc = cfg["split_column"]
    targets = cfg["target"]
    res = pl.read_csv(os.path.join(model_dir, "result.csv"))
    test = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(sc) == "test")
            .select(ID_COLUMNS + ["sample"] + targets)
            .collect())
    if test.height != res.height:
        sys.exit(f"row mismatch: {res.height} predictions vs {test.height} test rows")
    smp = test["sample"].to_numpy().astype(str)
    true = {t: test[t].cast(pl.Float64).to_numpy() for t in targets}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in targets}
    for t in targets:
        rt = res[f"true_{t}"].to_numpy().astype(np.float64)
        bad = int(np.sum(~np.isclose(rt, true[t], rtol=1e-5, atol=1e-6)))
        if bad:
            sys.exit(f"{bad} rows where result.csv true_{t} != test-row {t}: not aligned")
        if not np.all(np.isfinite(true[t]) & (true[t] > 0)):
            sys.exit(f"nonpositive or nonfinite {t}")
    fp = hashlib.sha256(test.select(ID_COLUMNS + ["sample"]).write_csv().encode()).hexdigest()
    ids = test.select(ID_COLUMNS).cast(pl.Int64).to_numpy()
    rtrue = res["true_trueE"].to_numpy().astype(np.float64)
    return smp, true, pred, fp, ids, rtrue


def load(horn):
    """Cards + aligned arrays for r05/r06/r07. Returns dict with sample, trueE, trueLepE, pred[run]."""
    cards = load_cards(horn)
    data_paths = {json.load(open(os.path.join(c["model_dir"], "input.json")))["data_path"] for c in cards.values()}
    if len(data_paths) != 1:
        sys.exit(f"models use different data files: {data_paths}")
    hashes = {r: {f: sha256_file(os.path.join(c["model_dir"], f)) for f in ("result.csv", "input.json")}
              for r, c in cards.items()}
    for r, c in cards.items():
        for f, h in hashes[r].items():
            if c["provenance"]["hashes"][f] != h:
                sys.exit(f"{r}: {f} sha256 differs from the one recorded in scorecard.json")
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    cpath = os.path.join(CACHE, f"{horn}_{key}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        out = dict(sample=z["sample"], trueE=z["trueE"], trueLepE=z["trueLepE"],
                   pred={r: z[f"pred_{r}"] for r in cards}, predLep={r: z[f"predLep_{r}"] for r in cards},
                   fingerprint=str(z["fingerprint"]))
    else:
        ref = None
        out = dict(pred={}, predLep={})
        for r, c in cards.items():
            cfg = json.load(open(os.path.join(c["model_dir"], "input.json")))
            smp, true, pred, fp, ids, rtrue = _load_one(cfg, c["model_dir"])
            if fp != c["test_set"]["fingerprint_sha256"]:
                sys.exit(f"{r}: recomputed fingerprint differs from scorecard")
            if ref is None:
                ref = (ids, rtrue, smp, true)
                out.update(sample=smp.astype(np.int8), trueE=true["trueE"], trueLepE=true["trueLepE"], fingerprint=fp)
            else:  # SPEC rule 10: identical event IDs and true_trueE row by row
                if not np.array_equal(ids, ref[0]) or not np.array_equal(smp, ref[2]):
                    sys.exit(f"{r}: event IDs/sample differ row-by-row from r05")
                if not np.array_equal(rtrue, ref[1]):
                    sys.exit(f"{r}: result.csv true_trueE differs row-by-row from r05")
            out["pred"][r] = pred["trueE"]
            out["predLep"][r] = pred["trueLepE"]
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cpath, sample=out["sample"], trueE=out["trueE"], trueLepE=out["trueLepE"],
                 fingerprint=out["fingerprint"], **{f"pred_{r}": v for r, v in out["pred"].items()},
                 **{f"predLep_{r}": v for r, v in out["predLep"].items()})
    out.update(cards=cards, hashes=hashes, data_path=data_paths.pop())
    sanity(out)
    return out


def sanity(d):
    """Aggregate S1/S2/S3 trueE bias/resolution over the window must equal scorecard.json."""
    t = d["trueE"]
    win = (t >= rs.WINDOW[0]) & (t <= rs.WINDOW[1])
    rows = []
    for r, c in d["cards"].items():
        for s in (1, 2, 3):
            m = win & (d["sample"] == s)
            x = d["pred"][r][m] / t[m] - 1.0
            q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84])
            ref = c["metrics"]["trueE"][str(s)]
            if m.sum() != ref["n"] or not np.isclose(q50, ref["bias"], atol=1e-9) \
                    or not np.isclose((q84 - q16) / 2, ref["resolution"], atol=1e-9):
                sys.exit(f"SANITY FAIL {r} S{s}: n {m.sum()} vs {ref['n']}, bias {q50} vs {ref['bias']}, "
                         f"res {(q84 - q16) / 2} vs {ref['resolution']}")
            rows.append(f"{r} S{s}: res {100 * (q84 - q16) / 2:.2f}% (card {100 * ref['resolution']:.2f}), "
                        f"bias {100 * q50:+.2f}% (card {100 * ref['bias']:+.2f})")
    d["sanity"] = rows


def model_lines(d):
    return "; ".join(f"{r} = {d['cards'][r]['model_dir']} (result.csv sha256 {d['hashes'][r]['result.csv'][:12]})"
                     for r in d["cards"])


def manifest(horn, plot_id, d, extra):
    """Append/replace this plot's entry in <horn>/manifest.json (locked; shared with other scripts)."""
    path = os.path.join(ROOT, horn, "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = dict(
        models={r: dict(scorecard=c["_path"], model_dir=c["model_dir"],
                        result_csv_sha256=d["hashes"][r]["result.csv"],
                        input_json_sha256=d["hashes"][r]["input.json"]) for r, c in d["cards"].items()},
        test_fingerprint_sha256=d["fingerprint"], data_path=d["data_path"],
        event_alignment="test-row run/subrun/cycle/evt/subevt, sample and result.csv true_trueE identical "
                        "row by row across r05/r06/r07",
        script=os.path.abspath(sys.argv[0]))
    entry.update(extra)
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f"manifest_{horn}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.load(open(path)) if os.path.exists(path) else {}
        m[plot_id] = entry
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
