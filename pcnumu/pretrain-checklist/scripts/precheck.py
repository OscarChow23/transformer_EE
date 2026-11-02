#!/usr/bin/env python
"""Read-only precheck for a TransformerEE `train_wide.py` Slurm launch.

Rebuilds the effective config exactly as train_wide.main() would (using the
launcher's own kset/apply_profile/validate_and_autofill), predicts the model
dir hash, scans the data file the way the polars loader will see it, and checks
the sbatch resources. Prints a Markdown report; writes a JSON summary with --json.
Exit code 1 if any FAIL.

Run inside the transformer_ee conda env (needs polars, numpy, torch, wandb imports):
  python -I precheck.py --sbatch JOB.sbatch [--reference REF.json] [--max-rows N]
"""
import argparse
import contextlib
import copy
import glob
import hashlib
import io
import json
import lzma
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE_DIR = "/data/nova/chow/transformerEE-pcnumu-loop"   # Git clone, branch pcnumu-loop
RUNS_DIRS = ["/data/nova/chow/pcnumu-runs",              # loop run folders
             "/data/nova/chow/transformerEE/runs"]       # frozen r01-r05 (read-only)
OLD_CONFIG_DIR = "/data/nova/chow/transformerEE/configs"  # frozen r05 configs, for name clashes
GIT = "/data/chow/tools/git-env/bin/git"
# simulate() mirrors train_wide.main() of this file version (pcnumu-loop, upstream c327032).
VERIFIED_TRAIN_WIDE_SHA256 = "564b7c8fc9295f8a8db007c5f126e64a75d93340099834d0624bb6e36b1d189e"
ID_COLS = ["run", "subrun", "cycle", "evt", "subevt"]
NONFINITE = {"nan", "-nan", "inf", "-inf", "+inf", "infinity", "-infinity"}
REQUIRED = ["data_path", "save_path", "model_phys_name", "vector", "scalar", "target",
            "weight", "loss", "optimizer.name", "optimizer.kwargs.lr", "model.name",
            "model.epochs", "model.kwargs.nhead", "model.kwargs.num_layers",
            "model.kwargs.dropout", "seed", "max_num_prongs",
            "batch_size_train", "batch_size_valid", "batch_size_test", "num_workers",
            "early_stopping"]
# (config key, CLI flag that kset() fills it from when the JSON lacks it).
KSET_KEYS = [("data_path", "--data-path"), ("save_path", "--save-path"),
             ("dataframe_type", "--dataframe-type"), ("num_workers", "--num-workers"),
             ("model.epochs", "--epochs"), ("optimizer.name", "--optimizer"),
             ("optimizer.kwargs.lr", "--lr"), ("optimizer.kwargs.weight_decay", "--weight-decay"),
             ("model.kwargs.nhead", "--nhead"), ("model.kwargs.num_layers", "--num-layers"),
             ("model.kwargs.dropout", "--dropout"), ("model.kwargs.dim_feedforward", "--dim-ff")]


class Report:
    def __init__(self):
        self.items = []
        self.facts = {}

    def add(self, level, section, msg):
        self.items.append({"level": level, "section": section, "msg": msg})

    def n(self, level):
        return sum(i["level"] == level for i in self.items)

    def markdown(self):
        out = ["# Pre-training precheck", "",
               f"FAIL {self.n('FAIL')} · WARN {self.n('WARN')} · PASS {self.n('PASS')} · INFO {self.n('INFO')}", ""]
        sections = []
        for i in self.items:
            if i["section"] not in sections:
                sections.append(i["section"])
        order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
        for s in sections:
            out += [f"## {s}", ""]
            for i in sorted((i for i in self.items if i["section"] == s), key=lambda i: order[i["level"]]):
                out.append(f"- **{i['level']}** {i['msg']}")
            out.append("")
        return "\n".join(out)


def dget(d, dotted, default=None):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def dhas(d, dotted):
    sentinel = object()
    return dget(d, dotted, sentinel) is not sentinel


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------- sbatch

def expand(s, env):
    def rep(m):
        name = m.group(1) or m.group(2)
        return env.get(name, f"<{name}>")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", rep, s)


def parse_sbatch(path):
    text = Path(path).read_text()
    short = {"-p": "partition", "-c": "cpus-per-task", "-t": "time", "-o": "output",
             "-J": "job-name", "-e": "error", "-G": "gpus"}
    directives = {}
    for line in text.splitlines():
        m = re.match(r"#SBATCH\s+(--[\w-]+|-\w)(?:[=\s]+(\S+))?", line.strip())
        if m:
            key = m.group(1)
            directives[short.get(key, key.lstrip("-"))] = m.group(2)
    joined = re.sub(r"\\\s*\n", " ", text)
    env, cmd = {}, None
    for line in joined.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", s)
        if m and ":+" not in m.group(2):
            env[m.group(1)] = expand(m.group(2).strip().strip("\"'"), env)
        m = re.search(r"\bpython[0-9.]*\s+(\S*train_wide\.py)\s*(.*)$", s)
        if m:
            cmd = (m.group(1), expand(m.group(2), env))
    argv = shlex.split(cmd[1]) if cmd else None
    return directives, env, (cmd[0] if cmd else None), argv


def mem_gb(s):
    if not s:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)([KMGT]?)", s.upper())
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 2**20, "M": 1 / 1024, "G": 1, "T": 1024, "": 1 / 1024}[m.group(2)]


def time_hours(s):
    if not s:
        return None
    # Slurm: MM | MM:SS | HH:MM:SS | D-HH | D-HH:MM | D-HH:MM:SS
    parts = [int(x) for x in re.split(r"[-:]", s)]
    if "-" in s:
        d, h, m, sec = (parts + [0, 0, 0])[:4]
        return d * 24 + h + m / 60 + sec / 3600
    if len(parts) == 3:
        return parts[0] + parts[1] / 60 + parts[2] / 3600
    return parts[0] / 60 + (parts[1] / 3600 if len(parts) == 2 else 0)


# ---------------------------------------------------------------- effective config

def simulate(code_dir, argv, rep):
    """Mirror train_wide.main() without its side effects (no makedirs, no stdout redirect)."""
    sys.path.insert(0, code_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        import train_wide as tw  # noqa: E402
        from transformer_ee.utils.hash_json import hash_dict  # noqa: E402
    args = tw.build_argparser().parse_args(argv)
    with open(args.base_config, encoding="utf-8") as f:
        base = json.load(f)
    explicit = {a.split("=")[0] for a in argv if a.startswith("--")}
    prov = []
    for key, flag in KSET_KEYS:
        dest = flag.lstrip("-").replace("-", "_")
        cli_val = getattr(args, dest, None)
        if key == "optimizer.kwargs.weight_decay" and cli_val is None:
            continue
        if dhas(base, key):
            src = "json"
            jv = dget(base, key)
            same = (str(cli_val).lower() == str(jv).lower()) if isinstance(jv, str) else cli_val == jv
            if flag in explicit and not same:
                src = "json (CLI value IGNORED)"
        elif cli_val is None:
            src = "unset"
        else:
            src = "cli" if flag in explicit else "argparse DEFAULT"
        prov.append({"key": key, "flag": flag, "json": dget(base, key), "cli": cli_val, "source": src})

    cfg = copy.deepcopy(base)
    k = tw.kset
    # --- same order as train_wide.main() at VERIFIED_TRAIN_WIDE_SHA256 ---
    if "noise" not in base and args.enable_noise:
        rep.add("WARN", "Effective config", "`--enable-noise` with no `noise` key: launcher fills GENIE default "
                "variable names (Final_State_Particles_*), almost certainly wrong for NOvA. Put `noise` in the JSON.")
    try:
        tw.configure_noise(cfg, args)          # validates/merges the noise block; no-op without one
    except ValueError as exc:
        rep.add("FAIL", "Effective config", f"launcher rejects the noise configuration: {exc}")
    k(cfg, "save_path", args.save_path)
    if args.auto_unique_save_path:
        rep.add("FAIL", "Effective config", "`--auto-unique-save-path` appends a timestamped folder: the model dir "
                "cannot be predicted and the run would not land in its run folder. Remove it.")
    k(cfg, "data_path", args.data_path)
    k(cfg, "dataframe_type", args.dataframe_type)
    k(cfg, "num_workers", args.num_workers)
    k(cfg, "model.epochs", args.epochs)
    if args.early_stop_window is not None or args.early_stop_min_delta_pct is not None:
        early = cfg.get("early_stopping", {})
        early["enabled"] = True
        if args.early_stop_window is not None:
            early["window"] = int(args.early_stop_window)
        if args.early_stop_min_delta_pct is not None:
            early["min_delta_pct"] = float(args.early_stop_min_delta_pct)
        cfg["early_stopping"] = early
    if args.keep_last_n_epoch_checkpoints is not None:
        ck = cfg.get("checkpointing", {})
        ck["keep_last_n_epoch_checkpoints"] = int(args.keep_last_n_epoch_checkpoints)
        cfg["checkpointing"] = ck
    k(cfg, "optimizer.name", args.optimizer)
    k(cfg, "optimizer.kwargs.lr", args.lr)
    if cfg["optimizer"]["name"] == "sgd":
        k(cfg, "optimizer.kwargs.momentum", args.momentum)
    if args.weight_decay is not None:
        k(cfg, "optimizer.kwargs.weight_decay", args.weight_decay)
    mkw = tw.ensure_model_kwargs(cfg)
    k(mkw, "nhead", args.nhead)
    k(mkw, "num_layers", args.num_layers)
    k(mkw, "dropout", args.dropout)
    k(mkw, "dim_feedforward", args.dim_ff)
    if args.profile:
        tw.apply_profile(cfg, args.profile)
    if args.d_model is not None:
        mkw["d_model"] = int(args.d_model)
        mkw["auto_d_model"] = False
    if args.auto_d_model:
        mkw["auto_d_model"] = True
    if args.min_head_dim is not None:
        mkw["min_head_dim"] = int(args.min_head_dim)
    if args.base_multiple is not None:
        mkw["base_multiple"] = int(args.base_multiple)
    if args.scalar_as_token:
        mkw["scalar_as_token"] = True
    if args.use_new_head:
        mkw["use_new_head"] = True
    if args.norm_first:
        mkw["norm_first"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tw.validate_and_autofill(cfg)
    for line in buf.getvalue().splitlines():
        rep.add("WARN", "Effective config", f"launcher says: {line.strip()}")
    return args, base, cfg, prov, hash_dict(cfg)


def check_effective(args, base, cfg, prov, mhash, argv, rep, config_dir, runs_dir):
    """runs_dir: list of run trees scanned for name reuse."""
    S = "Effective config"
    for p in prov:
        if p["source"] == "argparse DEFAULT":
            rep.add("FAIL", S, f"`{p['key']}` absent from JSON → silently takes argparse default `{p['cli']}` "
                               f"(`{p['flag']}`). Put it in the JSON.")
        elif "IGNORED" in p["source"]:
            rep.add("FAIL", S, f"`{p['flag']} {p['cli']}` on the command line is IGNORED; JSON `{p['key']}` = "
                               f"`{p['json']}` wins. Make the sbatch and JSON agree.")
        elif p["source"] == "cli":
            rep.add("WARN", S, f"`{p['key']}` = `{p['cli']}` comes only from CLI `{p['flag']}` (not in JSON). "
                               "Prefer every value in the JSON.")
    if not dhas(cfg, "model.kwargs.dropout"):
        rep.add("WARN", S, "dropout not set anywhere → model default 0.1 (implicit).")
    missing = [k for k in REQUIRED if not dhas(base, k)]
    if missing:
        rep.add("WARN", S, "keys not explicit in the base JSON: " + ", ".join(f"`{m}`" for m in missing))
    else:
        rep.add("PASS", S, "all required keys explicit in the base JSON.")
    over = [a for a in ("--profile", "--d-model", "--auto-d-model", "--scalar-as-token", "--use-new-head",
                        "--norm-first", "--min-head-dim", "--base-multiple", "--early-stop-window",
                        "--early-stop-min-delta-pct", "--keep-last-n-epoch-checkpoints") if a in argv]
    if over:
        changed = {k: (dget(base, k), dget(cfg, k)) for k in flatten(cfg) if dget(base, k) != dget(cfg, k)
                   and not any(k == p["key"] and p["source"] != "json" for p in prov)}
        rep.add("INFO", S, f"CLI flags that DO override JSON: {', '.join(over)}. Keys changed vs JSON: " +
                (", ".join(f"`{k}` {a}→{b}" for k, (a, b) in changed.items()) or "none"))
    sc = cfg.get("split_column")
    if sc:
        load_py = open(os.path.join(rep.facts["code_dir"], "transformer_ee", "dataloader", "load.py")).read()
        if 'config.get("split_column")' not in load_py:
            rep.add("FAIL", S, f"config sets split_column={sc!r} but the checked-out load.py does not support it "
                               "(apply patches/0001-load-split-column.patch).")
        else:
            rep.add("PASS", S, f"split taken from column `{sc}` (loader patch present); test_size/valid_size unused.")
        for k in ("test_size", "valid_size"):
            if k in base:
                rep.add("WARN", S, f"`{k}` is in the JSON but ignored because split_column is set.")
    else:
        for k in ("test_size", "valid_size"):
            if k not in base:
                rep.add("FAIL", S, f"`{k}` missing and no split_column.")
    for k in ("test_size", "valid_size") if not sc else ():
        if dget(cfg, k) in (0, 0.0):
            rep.add("FAIL", S, f"`{k}` = 0 → empty split (load.py slices `[n - 0:]`).")
    model_dir = os.path.join(cfg["save_path"], "model_" + mhash)
    rep.facts.update(model_dir=model_dir, effective_config=cfg)
    rep.add("INFO", S, f"predicted model dir: `{model_dir}`")
    sp = cfg["save_path"]
    if os.path.isdir(model_dir):
        rep.add("FAIL", S, "predicted model dir ALREADY EXISTS → no resume logic; input.json kept, checkpoints overwritten.")
    elif os.path.isdir(sp):
        rep.add("WARN", S, f"save_path exists (train.log is appended): `{sp}` contains {sorted(os.listdir(sp))[:8]}")
    else:
        rep.add("PASS", S, "save_path is new.")
    # name collisions with other configs / existing runs
    clashes = []
    others = glob.glob(os.path.join(config_dir, "**", "*.json"), recursive=True)
    for d in runs_dir:  # loop run folders keep their resolved config as effective_config.json
        others += glob.glob(os.path.join(d, "**", "effective_config.json"), recursive=True)
    for f in others:
        if os.path.abspath(f) == os.path.abspath(args.base_config):
            continue
        try:
            o = json.load(open(f))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        for key in ("save_path", "model_phys_name"):
            if o.get(key) and o.get(key) == cfg.get(key):
                clashes.append(f"`{key}` also in {f}")
    for d in runs_dir:
        for f in glob.glob(os.path.join(d, "**", "input.json"), recursive=True):
            if os.path.dirname(os.path.abspath(f)).endswith("model_" + mhash):
                continue  # this exact config's own dir (reported above)
            try:
                o = json.load(open(f))
            except Exception:
                continue
            if o.get("model_phys_name") == cfg.get("model_phys_name"):
                clashes.append(f"`model_phys_name` used by existing run {f}")
    for c in clashes:
        rep.add("FAIL", S, "name reuse: " + c)
    if not clashes:
        rep.add("PASS", S, "save_path and model_phys_name not used by any other config or run.")


# ---------------------------------------------------------------- data scan

def open_text_header(path):
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rt") as f:
        return f.readline().rstrip("\n").split(",")


def materialize(path, tmpdir, max_rows):
    """Return a plain-CSV path polars can batch-read (decompressing/truncating to tmp if needed)."""
    if not path.endswith(".xz") and not max_rows:
        return path, None
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=tmpdir)
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rb") as src, os.fdopen(fd, "wb") as dst:
        if max_rows:
            # rows can contain quoted commas but not newlines, so line counting is safe
            for i, line in enumerate(src):
                if i > max_rows:
                    break
                dst.write(line)
        else:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    return tmp, tmp


def scan_data(cfg, rep, tmpdir, max_rows, overlap_files):
    import polars as pl
    S = "Data"
    path = cfg["data_path"]
    if not os.path.isfile(path):
        rep.add("FAIL", S, f"data_path does not exist: `{path}`")
        return
    if not (path.endswith(".csv") or path.endswith(".xz")):
        rep.add("FAIL", S, "loader only accepts .csv or .xz")
    header = open_text_header(path)
    vec, sca, tgt = cfg["vector"], cfg["scalar"], cfg["target"]
    miss = [c for c in vec + sca + tgt if c not in header]
    if miss:
        rep.add("FAIL", S, "columns missing from file: " + ", ".join(miss))
        return
    dup = [c for c in set(vec + sca) if (vec + sca).count(c) > 1]
    if dup:
        rep.add("FAIL", S, "feature listed twice: " + ", ".join(dup))
    leaky = [c for c in vec + sca if c.startswith(("png.truth", "true")) or c in ID_COLS + ["sample", "weight", "mode"]]
    if leaky:
        rep.add("FAIL", S, "truth/ID/bookkeeping columns used as features: " + ", ".join(leaky))
    rep.facts["n_vector"], rep.facts["n_scalar"], rep.facts["n_columns_file"] = len(vec), len(sca), len(header)
    rep.add("INFO", S, f"`{path}` ({os.path.getsize(path) / 1e9:.2f} GB on disk), {len(header)} columns; "
                       f"features {len(vec)} vector + {len(sca)} scalar; targets {tgt}.")

    vec_in_file = [c for c in header if c.startswith("png.")]
    csv_path, tmp = materialize(path, tmpdir, max_rows)
    try:
        # all columns as strings (infer_schema_length=0); null/NaN semantics match the loader's read
        reader = pl.read_csv_batched(csv_path, batch_size=200_000, infer_schema_length=0)
        rows = 0
        dropped_null, dropped_nan = {}, {}
        inf_scalar = {}
        stats = {c: {"n": 0, "nonfin": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf, "eq0": 0, "eqm5": 0, "eqm1": 0}
                 for c in vec + sca}
        tvals = {t: [] for t in tgt}
        nprong, len_mismatch = {}, 0
        samples, keys, split_counts = {}, [], {}
        has_ids = all(c in header for c in ID_COLS)
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]
            rows += df.height
            # loader drop_nan_rows: null in ANY column, or NaN in any float-typed (non png.*) column
            anybad = np.zeros(df.height, bool)
            for c in header:
                nul = df[c].is_null().to_numpy()
                if nul.any():
                    dropped_null[c] = dropped_null.get(c, 0) + int(nul.sum())
                    anybad |= nul
                if c not in vec_in_file:
                    low = df[c].str.to_lowercase()
                    isnan = low.is_in(["nan", "-nan"]).to_numpy()
                    isinf = low.is_in(["inf", "-inf", "+inf", "infinity", "-infinity"]).to_numpy()
                    if isnan.any():
                        dropped_nan[c] = dropped_nan.get(c, 0) + int(isnan.sum())
                        anybad |= isnan
                    if isinf.any():
                        inf_scalar[c] = inf_scalar.get(c, 0) + int(isinf.sum())
            df = df.filter(~pl.Series(anybad))
            if "sample" in header:
                for s, n in df.group_by("sample").len().iter_rows():
                    samples[s] = samples.get(s, 0) + n
            sc = cfg.get("split_column")
            if sc and sc in header:
                by = ["sample", sc] if "sample" in header else [sc]
                for r in df.group_by(by).len().iter_rows():
                    k_ = tuple(r[:-1])
                    split_counts[k_] = split_counts.get(k_, 0) + r[-1]
            if has_ids:
                keys.extend(df.select(ID_COLS).iter_rows())
            for t in tgt:
                tvals[t].append(df[t].cast(pl.Float64, strict=False).to_numpy())
            base_len = None
            for c in vec:
                s = df[c].fill_null("").replace("", "0").str.split(",")
                ln = s.list.len().to_numpy()
                if base_len is None:
                    base_len = ln
                    empty = (df[c].fill_null("") == "").to_numpy()
                    for k_, v_ in zip(*np.unique(np.where(empty, 0, ln), return_counts=True)):
                        nprong[int(k_)] = nprong.get(int(k_), 0) + int(v_)
                else:
                    len_mismatch += int((ln != base_len).sum())
                arr = s.explode().cast(pl.Float64, strict=False).to_numpy()
                acc(stats[c], arr)
            for c in sca:
                acc(stats[c], df[c].cast(pl.Float64, strict=False).to_numpy())
        rep.facts["rows_read"] = rows
        if max_rows:
            rep.add("WARN", S, f"scanned only the first {rows} rows (--max-rows); counts are NOT full-file.")
        kept = sum(nprong.values())
        rep.facts["rows_after_loader_drop"] = kept
        dropped = rows - kept
        if dropped:
            rep.add("WARN", S, f"loader `drop_nan_rows` will drop {dropped} rows ({dropped / rows:.4%}); null by column "
                               f"{dropped_null or '{}'}, NaN by column {dropped_nan or '{}'} (checks ALL columns, not just features).")
        else:
            rep.add("PASS", S, f"{rows} rows; loader drops none.")
        for c, n in inf_scalar.items():
            rep.add("FAIL" if c in sca + tgt else "WARN", S, f"`{c}` has {n} inf values (not dropped by loader).")
        if len_mismatch:
            rep.add("FAIL", S, f"{len_mismatch} row×feature vector-length mismatches among selected vector features.")
        mnp = cfg.get("max_num_prongs")
        over = sum(v for k_, v in nprong.items() if mnp is not None and k_ > mnp)
        rep.add("WARN" if over else "PASS", S, f"nprong distribution {dict(sorted(nprong.items()))}; "
                f"{over} rows exceed max_num_prongs={mnp} (truncation keeps first prongs in CSV order). "
                f"nprong=0 rows: {nprong.get(0, 0)} (become one all-zero prong).")
        sentinels = []
        for c in vec + sca:
            st = stats[c]
            if st["n"] == 0:
                continue
            nf = st["n"] - st["nonfin"]
            mean = st["sum"] / max(nf, 1)
            std = float(np.sqrt(max(st["sumsq"] / max(nf, 1) - mean ** 2, 0)))
            kind = "vector" if c in vec else "scalar"
            if st["nonfin"]:
                rep.add("FAIL", "Features", f"`{c}` ({kind}): {st['nonfin']} non-finite values survive the loader → "
                                            "NaN mean/std for the whole feature. Remove/replace upstream.")
            if std == 0:
                rep.add("WARN", "Features", f"`{c}` ({kind}) is constant ({st['min']:g}) → normalizes to 0, carries no information.")
            elif st["max"] > mean + 200 * std or st["min"] < mean - 200 * std:
                rep.add("WARN", "Features", f"`{c}` ({kind}) extreme tail (>200σ): range [{st['min']:.3g}, {st['max']:.3g}], "
                                            f"mean {mean:.3g}, std {std:.3g}.")
            sent = " ".join(f"{lab}:{st[k_] / st['n']:.0%}" for k_, lab in (("eq0", "0"), ("eqm5", "-5"), ("eqm1", "-1"))
                             if st[k_] / st["n"] > 0.01)
            if sent:
                sentinels.append(f"`{c}` {sent}")
        if sentinels:
            rep.add("INFO", "Features", "fraction of values equal to 0 / -5 / -1 (sentinel candidates; loader mean/std "
                                        "includes them): " + "; ".join(sentinels))
        # targets
        for t in tgt:
            a = np.concatenate(tvals[t]) if tvals[t] else np.array([])
            q = np.quantile(a, [0, .01, .5, .99, 1]).round(3).tolist() if a.size else []
            bad = int((~np.isfinite(a)).sum())
            nonpos = int((a <= 0).sum())
            rep.add("FAIL" if bad else "INFO", "Targets & weights", f"`{t}` quantiles [0,1,50,99,100]%: {q}; non-finite {bad}.")
            if nonpos and any("percentage" in n for n in dget(cfg, "loss.kwargs.base_loss_names", [])):
                rep.add("FAIL", "Targets & weights", f"`{t}` has {nonpos} values ≤ 0 with a percentage loss → inf/huge loss.")
        if cfg.get("weight", {}).get("name") == "FlatSpectraWeights" and tvals[tgt[0]]:
            check_weights(cfg, np.concatenate(tvals[tgt[0]]), rep)
        # split
        n = kept
        sc = cfg.get("split_column")
        if sc:
            if sc not in header:
                rep.add("FAIL", "Split", f"split_column `{sc}` is not a column of the file.")
            else:
                labels = {k_[-1] for k_ in split_counts}
                if not labels <= {"train", "valid", "test"}:
                    rep.add("FAIL", "Split", f"`{sc}` has labels {labels}; loader accepts only train/valid/test.")
                tot = {}
                for k_, v in split_counts.items():
                    tot[k_[-1]] = tot.get(k_[-1], 0) + v
                rep.facts["split"] = tot
                lines = []
                for smp in sorted({k_[0] for k_ in split_counts if len(k_) == 2}):
                    ns = {k_[1]: v for k_, v in split_counts.items() if len(k_) == 2 and k_[0] == smp}
                    t = sum(ns.values())
                    lines.append(f"sample {smp}: " + ", ".join(f"{lab} {ns.get(lab, 0)} ({ns.get(lab, 0) / t:.1%})"
                                                                for lab in ("train", "valid", "test")))
                rep.add("INFO", "Split", f"split from column `{sc}` (fixed upstream, file order kept): total " +
                        ", ".join(f"{lab} {tot.get(lab, 0)}" for lab in ("train", "valid", "test")) +
                        ("; " + "; ".join(lines) if lines else ""))
        ts, vs = (cfg.get("test_size"), cfg.get("valid_size")) if not sc else (0, 0)
        nt = int(n * ts) if isinstance(ts, float) else ts
        nv = int(n * vs) if isinstance(vs, float) else vs
        if not sc:
            rep.facts["split"] = {"train": n - nt - nv, "valid": nv, "test": nt}
            rep.add("INFO", "Split", f"loader split (row shuffle, seed {cfg.get('seed')}): train {n - nt - nv}, valid {nv}, "
                                     f"test {nt}. No event grouping or stratification.")
        if samples:
            tot = sum(samples.values())
            rep.facts["samples"] = samples
            rep.add("INFO", "Split", "sample mix (after loader drop): " +
                    ", ".join(f"sample {k_}: {v} ({v / tot:.1%})" for k_, v in sorted(samples.items())))
        if has_ids:
            uniq = len(set(keys))
            if uniq != len(keys):
                rep.add("WARN", "Split", f"{len(keys) - uniq} duplicate (run,subrun,cycle,evt,subevt) keys.")
            ev = len({k_[:4] for k_ in keys})
            if ev != len(keys):
                rep.add("WARN", "Split", f"{len(keys) - ev} rows share an event with another row → row split can leak "
                                         "an event's slices across train/test.")
            for other in overlap_files:
                ok = other_keys(other, tmpdir)
                inter = len(set(k_[:4] for k_ in keys) & ok)
                rep.add("WARN" if inter else "PASS", "Split",
                        f"event overlap with `{other}`: {inter} events.")
        else:
            rep.add("INFO", "Split", "no ID columns in file → category/event-level evaluation needs a separate ID-carrying file.")
        rep.facts["uncompressed_gb"] = uncompressed_size(path) / 1e9
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def uncompressed_size(path):
    if not path.endswith(".xz"):
        return os.path.getsize(path)
    import subprocess
    out = subprocess.run(["xz", "--robot", "--list", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("totals"):
            return int(line.split("\t")[4])
    with lzma.open(path, "rb") as f:  # fallback: stream through
        return sum(len(c) for c in iter(lambda: f.read(1 << 24), b""))


def acc(st, arr):
    st["n"] += arr.size
    fin = np.isfinite(arr)
    st["nonfin"] += int((~fin).sum())
    a = arr[fin]
    if a.size:
        st["sum"] += float(a.sum())
        st["sumsq"] += float((a ** 2).sum())
        st["min"] = min(st["min"], float(a.min()))
        st["max"] = max(st["max"], float(a.max()))
        st["eq0"] += int((a == 0).sum())
        st["eqm5"] += int((a == -5).sum())
        st["eqm1"] += int((a == -1).sum())


def other_keys(path, tmpdir):
    import polars as pl
    csv_path, tmp = materialize(path, tmpdir, None)
    try:
        df = pl.read_csv(csv_path, columns=ID_COLS[:4])
        return set(df.iter_rows())
    finally:
        if tmp:
            os.remove(tmp)


def check_weights(cfg, y, rep):
    S = "Targets & weights"
    kw = dict(cfg["weight"].get("kwargs", {}))
    bins, rng, alpha = kw.get("bins", 50), kw.get("range"), kw.get("alpha", 0)
    y = y[np.isfinite(y)]
    hist, edges = np.histogram(y, bins=bins, range=rng)
    a = max(hist.max() * alpha, 1)
    w = 1 / (hist + a)
    w = w / w.mean()
    w = np.clip(w, kw.get("minweight", -np.inf), kw.get("maxweight", np.inf))
    e = edges.copy()
    e[0], e[-1] = -np.inf, np.inf
    ev = w[np.clip(np.digitize(y, e) - 1, 0, len(w) - 1)]
    empty = float((hist == 0).mean())
    ess = ev.sum() ** 2 / (ev ** 2).sum()
    lo, hi = (rng if rng else (y.min(), y.max()))
    rep.add("INFO", S, f"FlatSpectraWeights on `{cfg['target'][0]}`: bins {bins}, range {list(rng) if rng else 'None → data min..max'} "
                       f"= [{lo:.3g}, {hi:.3g}] ({(hi - lo) / bins:.3g} per bin), alpha {alpha}. "
                       f"Empty bins {empty:.1%}; per-event mean weight {ev.mean():.3f}; "
                       f"max/min event weight {ev.max() / ev.min():.1f}; effective sample size {ess / y.size:.1%} of rows.")
    if rng is None and empty > 0.2:
        rep.add("WARN", S, f"{empty:.0%} of bins are empty because range is min..max of a long-tailed target; "
                           "bin-mean normalization then shrinks the per-event mean weight (effective LR change). "
                           "Consider an explicit `range`.")
    rep.facts["weights"] = {"empty_bin_frac": empty, "mean_event_weight": float(ev.mean()), "ess_frac": float(ess / y.size)}


# ---------------------------------------------------------------- resources / code

def check_resources(directives, env, cfg, rep, config_path_arg, base_config):
    S = "Slurm resources"
    part = directives.get("partition") or "normalrun (default)"
    mem = mem_gb(directives.get("mem"))
    cpus = int(directives.get("cpus-per-task") or 1)
    hours = time_hours(directives.get("time"))
    gpu = directives.get("gres") or directives.get("gpus")
    rep.facts["slurm"] = {"partition": part, "mem_gb": mem, "cpus": cpus, "time_h": hours, "gpu": gpu,
                          "output": directives.get("output"), "job_name": directives.get("job-name")}
    rep.add("INFO", S, f"partition {part}, gres {gpu}, cpus {cpus}, mem {directives.get('mem')}, time {directives.get('time')}, "
                       f"job-name {directives.get('job-name')}")
    if not gpu:
        rep.add("FAIL", S, "no GPU requested (--gres=gpu:1).")
    if part.startswith("test") and hours and hours > 2:
        rep.add("FAIL", S, "`test` partition is limited to 2 h.")
    nw = cfg.get("num_workers", 0)
    if nw >= cpus:
        rep.add("WARN", S, f"num_workers {nw} ≥ cpus-per-task {cpus}.")
    unc = rep.facts.get("uncompressed_gb")
    if mem and unc:
        ratio = mem / unc
        lvl = "WARN" if ratio < 15 else "PASS"
        rep.add(lvl, S, f"mem/uncompressed-data = {ratio:.0f}× ({mem:.0f} GB / {unc:.2f} GB). Precedent: v4 FHC ran fine with "
                        "32 GB on 0.65 GB uncompressed (≈49×); WARN below 15× is a heuristic. The .xz path holds "
                        "compressed + decompressed bytes + the parsed frame + per-worker item caches.")
    out = directives.get("output")
    if out:
        d = os.path.dirname(expand(out, env))
        if not os.path.isdir(d):
            rep.add("FAIL", S, f"--output directory does not exist: `{d}` → Slurm cannot write the log. mkdir it before sbatch.")
        if os.path.abspath(d) != os.path.abspath(cfg["save_path"]):
            rep.add("WARN", S, f"--output dir `{d}` ≠ effective save_path `{cfg['save_path']}` (train.log goes to save_path).")
    else:
        rep.add("WARN", S, "no --output: log goes to slurm-<id>.out in the submit dir.")
    for name, val in env.items():
        if name.upper() in ("DATA",) and val != cfg["data_path"]:
            rep.add("FAIL", S, f"sbatch `{name}={val}` ≠ effective data_path `{cfg['data_path']}` (CLI --data-path ignored when JSON has it).")
        if name.upper() in ("RUN_DIR",) and os.path.abspath(val) != os.path.abspath(cfg["save_path"]):
            rep.add("FAIL", S, f"sbatch `{name}={val}` ≠ effective save_path `{cfg['save_path']}`.")
    if config_path_arg and os.path.abspath(config_path_arg) != os.path.abspath(base_config):
        rep.add("FAIL", S, f"--config `{config_path_arg}` is not the sbatch --base-config `{base_config}`.")
    if env.get("WANDB_MODE") != "offline":
        rep.add("WARN", S, "WANDB_MODE is not offline → launcher defaults point at another group's W&B project.")


def check_code(code_dir, rep):
    S = "Code state"
    import subprocess  # pylint: disable=C0415
    tw = os.path.join(code_dir, "train_wide.py")
    sha = hashlib.sha256(open(tw, "rb").read()).hexdigest()
    git = GIT if os.path.exists(GIT) else "git"

    def run(*cmd):
        return subprocess.run([git, "-C", code_dir, *cmd], capture_output=True, text=True, check=True).stdout.rstrip()
    try:
        branch, commit = run("rev-parse", "--abbrev-ref", "HEAD"), run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        branch = commit = dirty = None
        rep.add("WARN", S, f"cannot read Git state of {code_dir}: {exc}")
    rep.facts["code"] = {"branch": branch, "commit": commit, "dirty": bool(dirty), "train_wide_sha256": sha}
    if commit:
        rep.add("INFO", S, f"{branch} @ {commit[:12]}")
        if dirty:
            rep.add("FAIL", S, "tracked files have uncommitted changes; the loop refuses to train from a dirty "
                               "checkout:\n" + dirty)
        else:
            rep.add("PASS", S, "checkout clean (tracked files).")
    if sha != VERIFIED_TRAIN_WIDE_SHA256:
        rep.add("WARN", S, "train_wide.py differs from the version this precheck's merge logic was verified against "
                           "→ re-verify `simulate()` before trusting the effective-config section.")
    else:
        rep.add("PASS", S, "train_wide.py matches the verified version.")


def diff_reference(cfg, ref_path, rep):
    S = "Diff vs reference"
    ref = json.load(open(ref_path))
    rep.add("INFO", S, f"reference: `{ref_path}`")
    a, b = flatten(ref), flatten(cfg)
    for k in sorted(set(a) | set(b)):
        if k in ("vector", "scalar"):
            continue
        if a.get(k, "<absent>") != b.get(k, "<absent>"):
            rep.add("INFO", S, f"`{k}`: {a.get(k, '<absent>')} → {b.get(k, '<absent>')}")
    for k in ("vector", "scalar"):
        add = [c for c in cfg.get(k, []) if c not in ref.get(k, [])]
        rem = [c for c in ref.get(k, []) if c not in cfg.get(k, [])]
        if add or rem:
            rep.add("INFO", S, f"{k}: +{add} −{rem}")
        elif cfg.get(k) != ref.get(k):
            rep.add("INFO", S, f"{k}: same set, different order")
        else:
            rep.add("INFO", S, f"{k}: identical ({len(cfg.get(k, []))})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sbatch", required=True)
    p.add_argument("--config", help="JSON you intend to use; checked against the sbatch --base-config")
    p.add_argument("--reference", help="reference config or a run's model_*/input.json to diff against")
    p.add_argument("--code-dir", default=CODE_DIR)
    p.add_argument("--max-rows", type=int, default=None, help="scan only the first N rows (quick look)")
    p.add_argument("--overlap-with", action="append", default=[], help="other CSV(.xz) to check event overlap")
    p.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--json", help="write report items + facts here")
    a = p.parse_args()

    rep = Report()
    rep.facts["code_dir"] = a.code_dir
    directives, env, script, argv = parse_sbatch(a.sbatch)
    if argv is None:
        rep.add("FAIL", "Launch command", "no `python ... train_wide.py` command found in the sbatch.")
        print(rep.markdown())
        sys.exit(1)
    rep.add("INFO", "Launch command", f"`{script} {' '.join(argv)}`")
    check_code(a.code_dir, rep)
    args, base, cfg, prov, mhash = simulate(a.code_dir, argv, rep)
    check_effective(args, base, cfg, prov, mhash, argv, rep, OLD_CONFIG_DIR, RUNS_DIRS)
    if not a.skip_data:
        scan_data(cfg, rep, a.tmpdir, a.max_rows, a.overlap_with)
    check_resources(directives, env, cfg, rep, a.config, args.base_config)
    if a.reference:
        diff_reference(cfg, a.reference, rep)
    print(rep.markdown())
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"items": rep.items, "facts": rep.facts, "provenance": prov}, f, indent=2, default=str)
    sys.exit(1 if rep.n("FAIL") else 0)


if __name__ == "__main__":
    main()
