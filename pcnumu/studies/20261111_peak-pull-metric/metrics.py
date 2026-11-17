"""Candidate peak-pull measures, all as plain functions of (true, pred) arrays of ONE sample.

Sign convention everywhere: positive = towards the ~2 GeV peak of the raw spectrum
(overestimate on the low side, underestimate on the high side). Units: points (100 x), except where noted.
rho = ln(pred/true); r = pred/true - 1.
"""
import numpy as np

LOW, HIGH, MODE = (1.0, 1.5), (3.0, 4.0), (1.75, 2.25)
EDGES = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
CLIP = 1.0  # |rho| is clipped at 1 (a factor e) in means, so a handful of wild predictions cannot set the answer


def _win(t, w):
    return (t >= w[0]) & (t < w[1])


def rho(t, p):
    return np.clip(np.log(np.maximum(p, 1e-6) / t), -CLIP, CLIP)


# ---------- v1 (current) ----------
def v1_peak_pull(t, p):
    r = p / t - 1
    return 100 * (np.median(r[_win(t, LOW)]) - np.median(r[_win(t, HIGH)]))


def v1_pull_slope(t, p):
    meds, ok = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = _win(t, (lo, hi))
        ok.append(m.sum() >= 50)
        meds.append(np.median(p[m]) if m.any() else np.nan)
    ok = np.array(ok)
    return float(np.polyfit(np.log(CENTRES[ok]), np.log(np.array(meds)[ok]), 1)[0])


# ---------- per-side pieces (never subtracted) ----------
def side(t, p, w, sign, tau, tau_log=None):
    """All per-side candidates in one window. sign=+1 low side, -1 high side."""
    m = _win(t, w)
    r = p[m] / t[m] - 1
    x = rho(t[m], p[m])
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    out = {
        "median": 100 * sign * np.median(r),                      # v1 piece
        "mean_log": 100 * sign * x.mean(),                        # threshold-free, sees the tail
        "tail_asym": 100 * sign * ((q84 - q50) - (q50 - q16)),    # robust skew of rho
        "n": int(m.sum()),
    }
    for name, th in tau.items():                                  # proposal: net fraction in r space
        out[f"nf_r_{name}"] = 100 * ((sign * r > th).mean() - (sign * r < -th).mean())
        out[f"nf_log_{name}"] = 100 * ((sign * x > th).mean() - (sign * x < -th).mean())  # same, symmetric in ln
    return out


def all_sides(t, p, sig):
    tau = {"sig": sig, "10": 0.10, "20": 0.20}
    lo, hi, mo = side(t, p, LOW, +1, tau), side(t, p, HIGH, -1, tau), side(t, p, MODE, +1, tau)
    out = {}
    for k in lo:
        if k == "n":
            continue
        out[f"{k}_low"], out[f"{k}_high"] = lo[k], hi[k]
        # the same statistic at the mode (no net pull expected there) gives the run's overall offset and the
        # intrinsic asymmetry of its errors; subtracting it per side removes a global scale shift without
        # mixing the two sides. On the high side the reference enters with the opposite sign.
        out[f"{k}_low_ref"], out[f"{k}_high_ref"] = lo[k] - mo[k], hi[k] + mo[k]
    out["level_peak"] = mo["mean_log"]
    # net fraction after removing the run's level at the peak: rho minus the median rho of the mode window,
    # then count beyond +-sig towards minus away (the proposal, made blind to a global scale shift)
    c = np.median(rho(t[_win(t, MODE)], p[_win(t, MODE)]))
    for name, w, sign in (("low", LOW, 1), ("high", HIGH, -1)):
        x = sign * (rho(t[_win(t, w)], p[_win(t, w)]) - c)
        out[f"nfc_sig_{name}"] = 100 * ((x > sig).mean() - (x < -sig).mean())
    return out


# ---------- curves ----------
def curves(t, p, tau):
    """Per 0.25 GeV bin: n, q16/q50/q84 of r (%), mean rho (%), net fraction beyond tau in ln space (%),
    signed towards 2 GeV for the net fraction only."""
    rows = []
    for lo, hi, c in zip(EDGES[:-1], EDGES[1:], CENTRES):
        m = _win(t, (lo, hi))
        if m.sum() < 20:
            rows.append((c, int(m.sum())) + (np.nan,) * 5)
            continue
        r = p[m] / t[m] - 1
        x = rho(t[m], p[m])
        s = 1 if c < 2.0 else -1
        q = np.percentile(r, [16, 50, 84]) * 100
        rows.append((c, int(m.sum()), q[0], q[1], q[2], 100 * x.mean(),
                     100 * ((s * x > tau).mean() - (s * x < -tau).mean())))
    return np.array(rows, dtype=float)


# ---------- shape fits ----------
def log_fit(t, p):
    """Quadratic fit of mean rho per bin vs ln(E/2 GeV): slope-1 (shrinkage, <0 = squeezed) and curvature."""
    y = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = _win(t, (lo, hi))
        y.append(rho(t[m], p[m]).mean() if m.sum() >= 50 else np.nan)
    y = np.array(y)
    ok = np.isfinite(y)
    c2, c1, c0 = np.polyfit(np.log(CENTRES[ok] / 2.0), y[ok], 2)
    return {"shrink_slope": -100 * c1, "curvature": 100 * c2}  # shrink_slope > 0 = squeezed towards 2 GeV


def width_ratio(t, p):
    """sd(ln pred) / sd(ln true) for events with true 1-5 GeV. Pull narrows the predicted spectrum (<1);
    plain resolution widens it (>1)."""
    m = _win(t, (1.0, 5.0))
    return float(np.std(np.log(np.maximum(p[m], 1e-6))) / np.std(np.log(t[m])))


# ---------- projection on the pull predicted from the run's weighted training spectrum ----------
def bin_means(t, x):
    out = np.full(len(CENTRES), np.nan)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = _win(t, (lo, hi))
        if m.sum() >= 20:
            out[i] = x[m].mean()
    return out


def squeeze(t, p):
    """Local response slope deficit per side: 100 x (1 - d ln(pred) / d ln(true)), from a straight-line fit of
    the per-bin mean rho against ln(bin centre). Low side: bins 1.0-2.0 GeV; high side: bins 2.5-4.5 GeV.
    0 = estimates follow the truth one-to-one; 20 = a 10% step in true energy moves the estimate by only 8%.
    Blind to a global scale shift. Positive = squeezed (on the low side that is towards the peak)."""
    y = bin_means(t, rho(t, p))
    out = {}
    for name, (a, b) in (("low", (1.0, 2.0)), ("high", (2.5, 4.5))):
        m = (CENTRES > a) & (CENTRES < b) & np.isfinite(y)
        out[f"squeeze_{name}"] = -100 * np.polyfit(np.log(CENTRES[m]), y[m], 1)[0] if m.sum() >= 3 else np.nan
    return out


def projection(t, p, B, tilt=False):
    """Least-squares fit, event by event, of rho_i = a + c * B(t_i) [+ d * ln(t_i/2)] for true 1-5 GeV.
    B(t) = pull predicted for this run by the toy posterior estimator under the run's weighted training
    spectrum (mean ln(est/true) at true energy t; a callable). c = 1: the run is pulled exactly as the toy
    says; 0: not at all. Returns c and the fitted pull c*B averaged over the low and high windows (points,
    signed towards 2 GeV)."""
    m = _win(t, (1.0, 5.0))
    x, b = rho(t[m], p[m]), B(t[m])
    cols = [np.ones(m.sum()), b] + ([np.log(t[m] / 2.0)] if tilt else [])
    A = np.stack(cols, 1)
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    c = coef[1]
    return {"c": float(c),
            "fit_low": 100 * c * B(t[_win(t, LOW)]).mean(),
            "fit_high": -100 * c * B(t[_win(t, HIGH)]).mean()}


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def reco_calibration(t, p, w, edges=EDGES):
    """Self-consistency in bins of the PREDICTION: weighted median of ln(pred/true), weights = the run's
    training weight / true (MAPE's optimum is the 1/true-weighted median under the training weights).
    A network that is the best MAPE estimator for its own training spectrum gives 0 in every bin, however
    strongly it is pulled. Non-zero = an error that is not explained by the training prior (scale, tilt,
    under-fitting). Returns (centres, values in %, n)."""
    out, ns = [], []
    x = np.log(np.maximum(p, 1e-6) / t)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        ns.append(int(m.sum()))
        out.append(100 * wmedian(x[m], (w[m] / t[m])) if m.sum() >= 50 else np.nan)
    return 0.5 * (edges[1:] + edges[:-1]), np.array(out), np.array(ns)


def reco_defect(t, p, w):
    """One number from reco_calibration: n-weighted rms over prediction bins 1-5 GeV (points)."""
    _, v, n = reco_calibration(t, p, w)
    ok = np.isfinite(v)
    return float(np.sqrt(np.sum(n[ok] * v[ok] ** 2) / n[ok].sum()))


def scalar_metrics(t, p, sig, B=None, w=None):
    out = {"v1_peak_pull": v1_peak_pull(t, p), "v1_pull_slope": v1_pull_slope(t, p),
           "width_ratio": width_ratio(t, p)}
    out.update(all_sides(t, p, sig))
    out.update(log_fit(t, p))
    out.update(squeeze(t, p))
    if B is not None:
        pr = projection(t, p, B)
        out.update(proj_c=pr["c"], proj_low=pr["fit_low"], proj_high=pr["fit_high"])
        pr = projection(t, p, B, tilt=True)
        out.update(projT_c=pr["c"], projT_low=pr["fit_low"], projT_high=pr["fit_high"])
    if w is not None:
        out["reco_defect"] = reco_defect(t, p, w)
    return out
