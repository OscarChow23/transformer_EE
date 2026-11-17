"""Metric definitions: the one place where bias, resolution, windows, the ellipse and peak pull are defined.

Relative error r = (pred - true) / true.
  bias        = median(r)
  resolution  = (q84 - q16) / 2 of r (numpy.quantile, linear interpolation; not a FWHM)
  band_XX     = share of events with |r| < XX
Headline window: true trueE in [0.5, 6] GeV (inclusive at both ends, as in every saved scorecard).
Ellipse fraction: share of events with (100 r_trueE / a)^2 + (100 r_trueLepE / b)^2 <= 1, with
frozen axes per test set from ellipse_frozen.json.
"""
import json
import os

import numpy as np

SAMPLES = [("1", "Sample 1 (contained)"), ("2", "Sample 2 (peripheral, µ contained)"),
           ("3", "Sample 3 (peripheral, µ uncontained)"), ("all", "All samples")]
WINDOW = (0.5, 6.0)  # GeV, headline window on true trueE, inclusive
BANDS = (0.05, 0.10, 0.15)
MIN_PER_BIN = 50  # binned quantiles need at least this many events
ELLIPSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ellipse_frozen.json")


def relative_error(pred, true):
    return (pred - true) / true


def sample_mask(smp, k):
    """Events of sample k ("1", "2", "3"), or every event for "all"."""
    return (smp == k) if k != "all" else np.ones(smp.size, bool)


def in_headline_window(t):
    return (t >= WINDOW[0]) & (t <= WINDOW[1])


def stats(r):
    """n, bias, resolution, RMS and tolerance-band fractions of relative errors r. Empty -> NaN."""
    if r.size == 0:
        return dict(n=0, bias=np.nan, res=np.nan, rms=np.nan, f5=np.nan, f10=np.nan, f15=np.nan)
    q16, q50, q84 = np.quantile(r, [0.16, 0.5, 0.84])
    a = np.abs(r)
    return dict(n=int(r.size), bias=q50, res=(q84 - q16) / 2, rms=float(np.sqrt(np.mean(r ** 2))),
                f5=float(np.mean(a < 0.05)), f10=float(np.mean(a < 0.10)), f15=float(np.mean(a < 0.15)))


def binned_bias_resolution(x, r, edges, min_n=MIN_PER_BIN):
    """Bias and resolution of r in bins of x (np.digitize bins); NaN where a bin has < min_n events."""
    idx = np.digitize(x, edges) - 1
    bias, res = [], []
    for i in range(len(edges) - 1):
        ri = r[idx == i]
        if ri.size >= min_n:
            q16, q50, q84 = np.quantile(ri, [0.16, 0.5, 0.84])
            bias.append(q50)
            res.append((q84 - q16) / 2)
        else:
            bias.append(np.nan)
            res.append(np.nan)
    return np.array(bias), np.array(res)


# ---------- ellipse ----------
def inside_ellipse(x, y, a, b):
    """Per-event membership of (x, y) in the axis-aligned ellipse; same units for x, a and y, b."""
    return x * x / (a * a) + y * y / (b * b) <= 1.0


def root_axis(v, nbins):
    """Range as graph_eval/eval_model.C builds h2_energy_theta's x axis: data min/max,
    clamp_res_range to [-200, 200], then pad 5% of max(1, width)."""
    lo, hi = max(float(v.min()), -200.0), min(float(v.max()), 200.0)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * max(1.0, abs(hi - lo))
    return lo - pad, hi + pad, nbins


def root_findbin(v, lo, hi, nbins):
    """TAxis::FindBin for a fixed-width axis: 0 underflow, 1..nbins, nbins+1 overflow."""
    b = 1 + np.floor(nbins * (v - lo) / (hi - lo)).astype(np.int64)
    b[v < lo] = 0
    b[v >= hi] = nbins + 1
    return b


def ellipse_fraction(x, y, a, b, nbins=200):
    """Unbinned fraction (the headline) plus the binned eval_model.C cross-check.

    The binned version ports fractionInsideEllipse(h2, 0, 0, a, b, 0) on a 200x200 TH2D: bin
    centres are tested and the denominator is the histogram's in-range content, so entries
    beyond about ±210% are not counted. The y axis gets the same range rule as x.
    """
    xa, ya = root_axis(x, nbins), root_axis(y, nbins)
    bx, by = root_findbin(x, *xa), root_findbin(y, *ya)
    ok = (bx >= 1) & (bx <= nbins) & (by >= 1) & (by <= nbins)
    cx = xa[0] + (bx[ok] - 0.5) * (xa[1] - xa[0]) / nbins
    cy = ya[0] + (by[ok] - 0.5) * (ya[1] - ya[0]) / nbins
    inside = int(np.sum(inside_ellipse(cx, cy, a, b)))
    total = int(ok.sum())
    return dict(fraction=inside / total if total else None, n_in_histogram=total,
                n_out_of_range=int(x.size - total), x_range=[xa[0], xa[1]], y_range=[ya[0], ya[1]],
                unbinned_fraction=float(np.mean(inside_ellipse(x, y, a, b))) if x.size else None)


def ellipse_axes(fingerprint):
    """Frozen axes for this test set, or the provisional fallback if it is not listed."""
    spec = json.load(open(ELLIPSE_FILE))
    for name, e in spec["frozen"].items():
        for horn, fp in e["test_sets"].items():
            if fp == fingerprint:
                return dict(a_percent=e["a_percent"], b_percent=e["b_percent"], provisional=False,
                            frozen_as=f"{name}/{horn}", definition=e["definition"])
    fb = spec["provisional_fallback"]
    return dict(a_percent=fb["a_percent"], b_percent=fb["b_percent"], provisional=True,
                frozen_as=None, definition="unbinned")


# ---------- error maps (graph_eval/eval_model.C graph_resolution_stat "view_0to6_") ----------
ERRMAP = dict(bins=60, xmin=0.0, xmax=6.0)


def errmap_bins(x, y):
    """graph_resolution_stat binning: bin = int((x - xmin) / width) for xmin <= x <= xmax,
    kept if 0 <= bin < bins (so x == xmax is dropped in a fixed-range view)."""
    nb, lo, hi = ERRMAP["bins"], ERRMAP["xmin"], ERRMAP["xmax"]
    keep = (x >= lo) & (x <= hi)
    b = ((x[keep] - lo) / ((hi - lo) / nb)).astype(np.int64)
    ok = (b >= 0) & (b < nb)
    return b[ok], y[keep][ok]


# ---------- peak pull (version 1, adopted 2026-11-04) ----------
PEAK_VERSION = 1
PEAK_LOW, PEAK_HIGH = (1.0, 1.5), (3.0, 4.0)
PEAK_EDGES = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
PEAK_DEFINITIONS = (
    "A model that learned the ~2 GeV training peak as a prior pulls uncertain events towards it: events below "
    "the peak are overestimated, events above it underestimated. On trueE, per sample, on the frozen test rows: "
    "peak_pull = 100 * (median r for true 1.0-1.5 GeV - median r for true 3.0-4.0 GeV), 0 = no pull, positive = "
    "pulled towards the peak (points of relative error); pull_slope = slope of log(median pred) vs log(bin centre), "
    "0.25 GeV bins over 1-5 GeV, 1 = no pull, below 1 = estimates squeezed towards the middle; bias_curve = median r "
    "(%) in the same bins. The pull slope also rises if the model overestimates the tail, so read it together with "
    "peak_pull and the bias curve. Both measures miss behaviour above 4 GeV.")


def median_r(t, p, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.median(p[m] / t[m] - 1.0)) if m.any() else float("nan")


def peak_measures(t, p):
    """Peak pull, pull slope and the binned bias curve for one sample's true/pred trueE."""
    centres = 0.5 * (PEAK_EDGES[1:] + PEAK_EDGES[:-1])
    meds, bias, ns = [], [], []
    for lo, hi in zip(PEAK_EDGES[:-1], PEAK_EDGES[1:]):
        m = (t >= lo) & (t < hi)
        ns.append(int(m.sum()))
        meds.append(float(np.median(p[m])) if m.any() else float("nan"))
        bias.append(100 * median_r(t, p, lo, hi))
    ok = np.isfinite(meds) & (np.array(ns) >= MIN_PER_BIN)
    slope = float(np.polyfit(np.log(centres[ok]), np.log(np.array(meds)[ok]), 1)[0]) if ok.sum() >= 3 else None
    return {
        "peak_pull": round(100 * (median_r(t, p, *PEAK_LOW) - median_r(t, p, *PEAK_HIGH)), 3),
        "pull_slope": None if slope is None else round(slope, 4),
        "n_low": int(((t >= PEAK_LOW[0]) & (t < PEAK_LOW[1])).sum()),
        "n_high": int(((t >= PEAK_HIGH[0]) & (t < PEAK_HIGH[1])).sum()),
        "bias_curve": [{"lo": float(a), "hi": float(b), "n": n, "median_r_pct": round(x, 3)}
                       for a, b, n, x in zip(PEAK_EDGES[:-1], PEAK_EDGES[1:], ns, bias)],
    }


# ---------- bootstrap helpers ----------
def interval(reps):
    """Central 68% of the defined bootstrap replicates; undefined (NaN) replicates are counted, not used."""
    reps = np.asarray(reps, float)
    ok = np.isfinite(reps)
    if not ok.any():
        return dict(lo68=None, hi68=None, n_valid_reps=0, n_undefined_reps=int(reps.size))
    lo, hi = np.quantile(reps[ok], [0.16, 0.84])
    return dict(lo68=float(lo), hi68=float(hi), n_valid_reps=int(ok.sum()), n_undefined_reps=int((~ok).sum()))


def wilson68(k, n):
    """Wilson score interval at z = 1 (about 68%), conditional on n. None if n == 0."""
    if n == 0:
        return None
    z, ph = 1.0, k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]
