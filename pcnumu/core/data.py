"""Inputs: aligned test predictions, model folders, train.log losses, and provenance hashes.

Every evaluation reads predictions the same way. result.csv (written by training, in test-row
file order) has no event IDs, so it is aligned to the dataset's test rows by position, and the
alignment is verified row by row on the true values. Any mismatch stops the program.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import polars as pl

ID_COLUMNS = ["run", "subrun", "cycle", "evt", "subevt"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE = os.path.dirname(os.path.abspath(__file__))
GIT = "/data/chow/tools/git-env/bin/git"


# ---------- model folders ----------
def find_model_dir(save_path):
    """Newest save_path/model_*/ that has a result.csv. Prefer a scorecard's recorded model_dir."""
    dirs = [d for d in glob.glob(os.path.join(save_path, "model_*")) if os.path.exists(os.path.join(d, "result.csv"))]
    if not dirs:
        sys.exit(f"no model_*/result.csv under {save_path}")
    return max(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "result.csv")))


def scorecard_of(run_dir):
    """A run's saved scorecard.json, which names the model folder it scored."""
    return json.load(open(os.path.join(run_dir, "scorecard", "scorecard.json")))


# ---------- predictions ----------
def load_test_predictions(cfg, model_dir):
    """Predictions aligned with the data file's test rows.

    Returns (sample labels, {target: true}, {target: pred}, fingerprint). Truths come from the
    source CSV as Float64; result.csv's float32 truths are used only to check the alignment.
    The fingerprint is the sha256 of the test rows' event IDs and sample labels in file order.
    """
    sc = cfg.get("split_column")
    if not sc:
        sys.exit("config has no split_column; the frozen test set needs the split file")
    targets = cfg["target"]
    res = pl.read_csv(os.path.join(model_dir, "result.csv"))
    test = (pl.scan_csv(cfg["data_path"], infer_schema_length=0)
            .filter(pl.col(sc) == "test")
            .select(ID_COLUMNS + ["sample"] + targets)
            .collect())
    if test.height != res.height:
        sys.exit(f"row mismatch: {res.height} predictions vs {test.height} test rows in {cfg['data_path']}")
    smp = test["sample"].to_numpy().astype(str)
    if not set(smp) <= {"1", "2", "3"}:
        sys.exit("unexpected sample labels")
    true = {t: test[t].cast(pl.Float64).to_numpy() for t in targets}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in targets}
    for t in targets:  # result.csv stores float32 truths: they must match the file row by row
        rt = res[f"true_{t}"].to_numpy().astype(np.float64)
        bad = int(np.sum(~np.isclose(rt, true[t], rtol=1e-5, atol=1e-6)))
        if bad:
            sys.exit(f"{bad} rows where result.csv true_{t} != test-row {t}: predictions are not aligned")
        if not np.all(np.isfinite(true[t]) & (true[t] > 0)):
            sys.exit(f"nonpositive or nonfinite {t}; cannot compute relative error")
    fingerprint = hashlib.sha256(test.select(ID_COLUMNS + ["sample"]).write_csv().encode()).hexdigest()
    return smp, true, pred, fingerprint


def load_result_csv(path, targets):
    """All rows of a result.csv on its own stored truths (older runs without a split column)."""
    res = pl.read_csv(path)
    true = {t: res[f"true_{t}"].to_numpy().astype(np.float64) for t in targets}
    pred = {t: res[f"pred_{t}"].to_numpy().astype(np.float64) for t in targets}
    return true, pred


def scan_test_ids(data_path, split_column):
    """(rows, unique event IDs, sample labels) of the dataset's test rows, in file order."""
    test = (pl.scan_csv(data_path, infer_schema_length=0).filter(pl.col(split_column) == "test")
            .select(ID_COLUMNS + ["sample"]).collect())
    return test.height, test.select(ID_COLUMNS).n_unique(), test["sample"].to_numpy().astype(str)


def parse_losses(log_path):
    """Per-epoch (epoch, train_loss, valid_loss) from the LAST training session in train.log.

    train.log is opened in append mode, so a re-launch into the same folder appends a second
    session (r04 appended to r03's log). Sessions start at a "Start Time:" line.
    """
    pat = re.compile(r"^Epoch: (\d+), train_loss: (\S+), valid_loss: (\S+)\s*$")
    sessions = [[]]
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("Start Time:"):
                sessions.append([])
            m = pat.match(line)
            if m:
                sessions[-1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    n_sessions = sum(1 for s in sessions if s)
    return (sessions[-1] if sessions[-1] else []), n_sessions


# ---------- provenance ----------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version():
    """Commit and dirty flag of the checkout this code runs from (None if not a checkout)."""
    git = GIT if os.path.exists(GIT) else "git"
    try:
        commit = subprocess.run([git, "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run([git, "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def code_hashes(*scripts):
    """sha256 of the given command scripts and of every pcnumu/core module they rely on."""
    files = list(scripts) + sorted(glob.glob(os.path.join(CORE, "*.py")))
    return {os.path.relpath(os.path.abspath(f), REPO): sha256_file(f) for f in files}
