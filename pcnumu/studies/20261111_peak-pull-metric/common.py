"""Shared helpers for the peak-pull metric study (11 Nov 2026). CPU only, read-only on run folders.

Model directories come from each run's scorecard/scorecard.json (never from a glob) and the
test-set fingerprint is checked against the scorecard, as pull_dist.py does.

    python -I common.py        # builds cache/pred_<horn>.npz and cache/feat_<horn>.npz (once)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/data/nova/chow/transformerEE-pcnumu-loop"
R = "/data/nova/chow/pcnumu-runs"
CACHE = os.path.join(HERE, "cache")
FIGS = os.path.join(HERE, "figures")
TRAIN_NPZ = R + "/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{}.npz"  # y = trueE, s = sample, train rows

RUNS = {
    "r05": "r05_v3data-allsamples_v3features",
    "r06": "r06_flat-alpha001",
    "r07": "r07_s1-subsample",
    "r07b": "r07b_s1-subsample-repeat",
    "r07c": "r07c_s1-subsample-repeat",
    "r07d": "r07d_s1-subsample-repeat",
    "r08": "r08_per-sample-flat",
    "r09": "r09_per-sample-flat-r05shares",
}
R07 = ["r07", "r07b", "r07c", "r07d"]
MAIN = ["r05", "r06", "r08", "r09"]
HORNS = ["fhc", "rhc"]
SAMPLES = ["1", "2", "3"]
# frozen r05 trueE resolution per sample (fractions), from the leaderboard / handoff section 3
SIGMA = {"fhc": {"1": 0.0675, "2": 0.101, "3": 0.1518}, "rhc": {"1": 0.061, "2": 0.092, "3": 0.1724}}
EDGES = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
COL = {"r05": "#2a78d6", "r06": "#eb6834", "r08": "#1baf7a", "r09": "#eda100", "r07": "#8a8a85"}  # dataviz reference palette slots 1-4, grey for the r07 repeats


def run_dir(run, horn):
    return f"{R}/{RUNS[run]}/{horn}"


def _load_one(run, horn):
    sys.path.insert(0, REPO + "/pcnumu/eval")
    import scorecard as sc  # noqa: E402

    card = json.load(open(f"{run_dir(run, horn)}/scorecard/scorecard.json"))
    md = card["model_dir"]
    if "smoke" in md:
        sys.exit(f"smoke model refused: {md}")
    cfg = json.load(open(os.path.join(md, "input.json")))
    smp, true, pred, fp = sc.load_test_predictions(cfg, md)
    if fp != card["test_set"]["fingerprint_sha256"]:
        sys.exit(f"{run} {horn}: test-set fingerprint differs from the scorecard's")
    return dict(smp=smp, t=true["trueE"], p=pred["trueE"], tl=true["trueLepE"], pl=pred["trueLepE"],
                fp=fp, model_dir=md, data_path=cfg["data_path"], split_column=cfg["split_column"])


def build_cache():
    import polars as pl

    os.makedirs(CACHE, exist_ok=True)
    for horn in HORNS:
        out = {}
        ref = None
        for run in RUNS:
            d = _load_one(run, horn)
            if ref is None:
                ref = d
                out.update(smp=d["smp"], t=d["t"], tl=d["tl"], fp=d["fp"])
            # all runs must share the test rows, or event-by-event comparisons are meaningless
            assert d["fp"] == ref["fp"] and np.array_equal(d["t"], ref["t"]) and np.array_equal(d["smp"], ref["smp"])
            out[f"p_{run}"] = d["p"]
            out[f"pl_{run}"] = d["pl"]
            out[f"dir_{run}"] = d["model_dir"]
            print(horn, run, d["model_dir"], d["fp"][:12], len(d["p"]), flush=True)
        np.savez(os.path.join(CACHE, f"pred_{horn}.npz"), **out)
        # a few event-level inputs of the same test rows, same row order (uncertainty proxies)
        cols = ["calE", "remPngCalE", "nHit", "orphCalE", "trueHadE", "mode", "png.len", "png.calE", "trueE"]
        test = (pl.scan_csv(ref["data_path"], infer_schema_length=0)
                .filter(pl.col(ref["split_column"]) == "test").select(cols).collect())
        assert test.height == len(ref["t"])
        assert np.allclose(test["trueE"].cast(pl.Float64).to_numpy(), ref["t"])
        feat = {c: test[c].cast(pl.Float64).to_numpy() for c in cols[:6]}
        for c in ("png.len", "png.calE"):  # comma-separated per-prong lists; keep the first (leading prong) and the count
            parts = [s.split(",") if s else [] for s in test[c].to_list()]
            feat[c + "0"] = np.array([float(x[0]) if x and x[0] else np.nan for x in parts])
            feat["nprong"] = np.array([len(x) for x in parts], dtype=float)
        np.savez(os.path.join(CACHE, f"feat_{horn}.npz"), **feat)
        print(horn, "features cached", test.height, flush=True)


_PRED = {}


def load(horn):
    """dict with smp, t (true trueE), p_<run> (pred trueE), dir_<run>, fp; all runs on the same test rows."""
    if horn not in _PRED:
        d = np.load(os.path.join(CACHE, f"pred_{horn}.npz"))
        _PRED[horn] = {k: d[k] for k in d.files}
    return _PRED[horn]


def sample_tp(horn, run, s):
    d = load(horn)
    m = d["smp"] == s
    return d["t"][m], d[f"p_{run}"][m]


def model_dirs_md():
    rows = []
    for horn in HORNS:
        d = load(horn)
        for run in RUNS:
            rows.append(f"| {run} | {horn.upper()} | `{d['dir_' + run]}` |")
    return "\n".join(rows)


# ---------- training spectra as each run weighted them ----------
def train_rows(horn):
    d = np.load(TRAIN_NPZ.format(horn))
    return d["y"].astype(np.float64), d["s"].astype(int)


_W = {}


def train_weights(horn, run):
    """(y, s, w): train-row trueE, sample, and the loss weight that run applied.

    r05/r06: pooled FlatSpectraWeights (alpha 0.05 / 0.01, 500 bins). r07*: r06's weighter refitted after
    dropping half of S1; the real drop is a hash of the event id (energy-blind), reproduced here by a fixed
    random half, which gives the same weights up to counting noise. r08/r09: PerSampleFlatWeights from the
    trial spec (r09's per-sample share constants do not change the shape within a sample).
    """
    key = (horn, run[:3])
    if key in _W:
        return _W[key]
    sys.path.insert(0, REPO)
    from transformer_ee.utils.weights import FlatSpectraWeights, PerSampleFlatWeights  # noqa: E402

    y, s = train_rows(horn)
    if run in ("r05", "r06"):
        w = FlatSpectraWeights(y, bins=500, alpha={"r05": 0.05, "r06": 0.01}[run]).getweight(y)
    elif run.startswith("r07"):
        keep = (s != 1) | (np.random.default_rng(20261104).random(y.size) < 0.5)
        y, s = y[keep], s[keep]
        w = FlatSpectraWeights(y, bins=500, alpha=0.01).getweight(y)
    else:
        cfg = json.load(open(f"{run_dir(run, horn)}/effective_config.json"))
        assert cfg["weight"]["name"] == "PerSampleFlatWeights"
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            w = PerSampleFlatWeights(y, s, **cfg["weight"]["kwargs"]).getweight(y, s)
    _W[key] = (y, s, np.asarray(w, dtype=np.float64))
    return _W[key]


FINE = np.arange(0.0, 40.0001, 0.05)
XC = 0.5 * (FINE[1:] + FINE[:-1])


def prior_hist(horn, run, s, weighted=True):
    """Weighted training trueE density of sample s as run trained it (events per 0.05 GeV, 0-40 GeV)."""
    y, smp, w = train_weights(horn, run)
    m = smp == int(s)
    h, _ = np.histogram(y[m], FINE, weights=w[m] if weighted else None)
    return h


def bayes_table(prior, sig, mape=True, grid=None):
    """Toy estimator: for an input that points to energy m (Gaussian, relative width sig), the estimate is the
    median of prior(E) x Gauss(E; m, sig*m) [x 1/E if mape: MAPE's optimum is the 1/true-weighted median].
    Returns (grid, estimate)."""
    grid = np.arange(0.2, 14, 0.02) if grid is None else grid
    est = np.empty_like(grid)
    pr = prior / XC if mape else prior
    for i, m in enumerate(grid):
        post = pr * np.exp(-0.5 * ((XC - m) / (sig * m)) ** 2)
        c = np.cumsum(post)
        est[i] = XC[np.searchsorted(c, c[-1] / 2)] if c[-1] > 0 else m
    return grid, est


def log_slope(prior, sig, t):
    """d ln(prior) / d ln E at energies t, after smoothing the density per unit ln E with a Gaussian of width
    sig in ln E (what an event of relative uncertainty sig 'sees'). This is the direction the prior pulls:
    positive = towards higher energy. MAPE's 1/E factor turns a density per GeV into a density per ln E times
    1/E^0, so the relevant density is prior(E) per GeV (see report)."""
    u = np.log(XC)
    ug = np.arange(np.log(0.3), np.log(30), 0.01)
    dens = np.interp(ug, u, prior, left=0, right=0)
    k = np.exp(-0.5 * (np.arange(-4 * sig, 4 * sig + 1e-9, 0.01) / sig) ** 2)
    sm = np.convolve(dens, k / k.sum(), mode="same")
    ls = np.gradient(np.log(np.maximum(sm, 1e-12)), ug)
    return np.interp(np.log(t), ug, ls)




# ---------- toy estimator under a run's effective prior ----------
_Z = np.linspace(-4, 4, 161)
_ZW = np.exp(-0.5 * _Z ** 2)
_ZW /= _ZW.sum()
_TAB = {}


def flat_prior():
    return np.ones_like(XC)


def toy_table(horn, run, s, sig, mape=True):
    """(grid, est) for the toy estimator under run's weighted training spectrum of sample s; run=None: flat prior."""
    key = (horn, None if run is None else run[:3], str(s), round(sig, 5), mape)
    if key not in _TAB:
        pr = flat_prior() if run is None else prior_hist(horn, run, s)
        _TAB[key] = bayes_table(pr, sig, mape)
    return _TAB[key]


def toy_predict(t, z, horn, run, s, sig, mape=True):
    """Toy prediction for true energies t with unit-normal draws z: the input points to m = t(1 + sig z)."""
    g, e = toy_table(horn, run, s, sig, mape)
    return np.interp(np.clip(t * (1 + sig * z), g[0], g[-1]), g, e)


def predicted_pull(horn, run, s, sig, kind="mean"):
    """Callable B(t): the toy's pull at true energy t, = ln(estimate under the run's prior) minus
    ln(estimate under a flat prior), averaged over the input blur (kind='mean') or at the central input
    (kind='median'). Natural sign (positive = pushed up). Depends only on the trial spec, not on the model."""
    tg = np.arange(0.5, 9.0, 0.025)
    g, e = toy_table(horn, run, s, sig)
    g0, e0 = toy_table(horn, None, s, sig)
    if kind == "median":
        b = np.log(np.interp(tg, g, e) / np.interp(tg, g0, e0))
    else:
        m = np.clip(tg[:, None] * (1 + sig * _Z[None, :]), g[0], g[-1])
        b = (np.log(np.interp(m, g, e) / np.interp(m, g0, e0)) * _ZW[None, :]).sum(1)
    return lambda t: np.interp(t, tg, b)


def weighter(horn, run):
    """Callable (trueE, sample int array) -> the loss weight that run would give those rows."""
    sys.path.insert(0, REPO)
    from transformer_ee.utils.weights import FlatSpectraWeights, PerSampleFlatWeights  # noqa: E402

    y, s = train_rows(horn)
    if run in ("r05", "r06"):
        w = FlatSpectraWeights(y, bins=500, alpha={"r05": 0.05, "r06": 0.01}[run])
        return lambda x, smp: w.getweight(x)
    if run.startswith("r07"):
        keep = (s != 1) | (np.random.default_rng(20261104).random(y.size) < 0.5)
        w = FlatSpectraWeights(y[keep], bins=500, alpha=0.01)
        return lambda x, smp: w.getweight(x)
    import contextlib
    import io

    cfg = json.load(open(f"{run_dir(run, horn)}/effective_config.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        w = PerSampleFlatWeights(y, s, **cfg["weight"]["kwargs"])
    return lambda x, smp: w.getweight(x, smp)


if __name__ == "__main__":
    build_cache()
