#!/usr/bin/env python
"""Make a column-pruned training CSV with a fixed per-event train/valid/test split.

The split is a deterministic hash of (seed, run, subrun, cycle, evt), so:
  * every sample gets the same fractions (test/valid/train = 20/4/76 by default),
    up to binomial fluctuation (<0.3% for the smallest sample);
  * an event lands in the same split in every derived file made with the same seed,
    whatever columns are kept or rows dropped, so later runs (other features, a reduced
    sample 1, ...) are evaluated on the same test events.

Also drops events with non-finite values (e.g. "nan" inside vector strings) in any kept
feature or target, and writes a small smoke-test file and a manifest.

  python make_split_csv.py --input X.csv.xz --config CONFIG.json [CONFIG2.json ...] \
      --out-dir DIR --name NAME [--seed 20261029] [--test 0.20 --valid 0.04]
"""
import argparse
import hashlib
import json
import lzma
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import polars as pl

ID_COLS = ["run", "subrun", "cycle", "evt", "subevt"]
KEEP_ALWAYS = ID_COLS + ["sample", "trueE", "trueLepE", "trueHadE", "weight", "mode"]
NONFINITE = ["nan", "-nan", "inf", "-inf", "+inf", "infinity", "-infinity"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def assign_split(df, seed, f_test, f_valid):
    run, sub, cyc, evt = (df[c].cast(pl.UInt64).to_numpy() for c in ID_COLS[:4])
    if sub.max() >= 1000 or cyc.max() >= 1000 or evt.max() >= 10**6 or run.max() >= 10**7:
        raise ValueError("ID out of range for the packed event key; widen the packing")
    key = ((run * np.uint64(1000) + sub) * np.uint64(1000) + cyc) * np.uint64(10**6) + evt
    with np.errstate(over="ignore"):
        u = splitmix64(key ^ np.uint64(seed)).astype(np.float64) / 2.0**64
    return np.where(u < f_test, "test", np.where(u < f_test + f_valid, "valid", "train"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--config", nargs="+", required=True, help="configs whose vector/scalar/target columns to keep")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True, help="output basename (no extension)")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--test", type=float, default=0.20)
    p.add_argument("--valid", type=float, default=0.04)
    p.add_argument("--smoke-fraction", type=float, default=0.02)
    p.add_argument("--max-rows", type=int, default=None, help="process only the first N rows (testing)")
    p.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/tmp"))
    a = p.parse_args()

    header = lzma.open(a.input, "rt").readline().rstrip("\n").split(",") if a.input.endswith(".xz") \
        else open(a.input).readline().rstrip("\n").split(",")
    feats, vec = [], set()
    for c in a.config:
        cfg = json.load(open(c))
        feats += cfg["vector"] + cfg["scalar"] + cfg["target"]
        vec |= set(cfg["vector"])
    keep = list(dict.fromkeys(KEEP_ALWAYS + feats))
    missing = [c for c in keep if c not in header]
    if missing:
        sys.exit(f"columns missing from input: {missing}")
    keep = [c for c in header if c in keep]  # preserve the input's column order
    check = list(dict.fromkeys(feats))

    os.makedirs(a.out_dir, exist_ok=True)
    out_csv = os.path.join(a.out_dir, a.name + ".csv")
    smoke_csv = os.path.join(a.out_dir, a.name + "_smoke.csv")
    for f in (out_csv, smoke_csv):
        if os.path.exists(f):
            sys.exit(f"refusing to overwrite {f}")

    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=a.tmpdir)
    try:
        with (lzma.open(a.input, "rb") if a.input.endswith(".xz") else open(a.input, "rb")) as src, \
                os.fdopen(fd, "wb") as dst:
            for i, line in enumerate(src):
                if a.max_rows is not None and i > a.max_rows:
                    break
                dst.write(line)
        reader = pl.read_csv_batched(tmp, batch_size=200_000, infer_schema_length=0, columns=keep)
        counts, dropped = {}, []
        rng = np.random.default_rng(a.seed)
        first = True
        with open(out_csv, "w") as fo, open(smoke_csv, "w") as fs:
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                df = batches[0]
                bad = np.zeros(df.height, bool)
                for c in check:
                    low = df[c].fill_null("").str.to_lowercase()
                    if c in vec:
                        toks = low.str.split(",").list.eval(pl.element().str.strip_chars().is_in(NONFINITE)).list.any()
                        bad |= toks.fill_null(False).to_numpy()
                    else:
                        bad |= low.is_in(NONFINITE).to_numpy()
                    bad |= df[c].is_null().to_numpy()
                if bad.any():
                    dropped += df.filter(pl.Series(bad)).select(ID_COLS + ["sample"]).rows()
                    df = df.filter(pl.Series(~bad))
                df = df.with_columns(pl.Series("split", assign_split(df, a.seed, a.test, a.valid)))
                for s, sp, n in df.group_by(["sample", "split"]).len().iter_rows():
                    counts.setdefault(str(s), {}).setdefault(sp, 0)
                    counts[str(s)][sp] += n
                df.write_csv(fo, include_header=first)
                smoke = df.filter(pl.Series(rng.random(df.height) < a.smoke_fraction))
                smoke.write_csv(fs, include_header=first)
                first = False
                print(f"{sum(sum(v.values()) for v in counts.values())} rows kept, {len(dropped)} dropped", flush=True)
    finally:
        os.remove(tmp)

    frac = {s: {sp: round(n / sum(v.values()), 5) for sp, n in sorted(v.items())} for s, v in counts.items()}
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.abspath(__file__), "script_sha256": sha256(__file__),
        "input": os.path.abspath(a.input), "input_sha256": sha256(a.input),
        "configs": [os.path.abspath(c) for c in a.config],
        "seed": a.seed, "fractions": {"test": a.test, "valid": a.valid, "train": 1 - a.test - a.valid},
        "split_rule": "u = splitmix64(packed(run,subrun,cycle,evt) XOR seed) / 2^64; test if u<test, valid if u<test+valid",
        "max_rows": a.max_rows,
        "columns": keep + ["split"],
        "counts": counts, "achieved_fractions": frac,
        "dropped_nonfinite": {"n": len(dropped), "ids(run,subrun,cycle,evt,subevt,sample)": dropped},
        "outputs": {out_csv: sha256(out_csv), smoke_csv: sha256(smoke_csv)},
    }
    with open(os.path.join(a.out_dir, a.name + ".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"counts": counts, "achieved_fractions": frac, "dropped": len(dropped)}, indent=1))


if __name__ == "__main__":
    main()
