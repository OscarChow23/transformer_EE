"""Regional comparison: windows, migration counts and paired bootstrap for models on the same test events.

Windows are on true trueE, lower-inclusive and upper-exclusive (None = no upper edge).
  resolution / bias      stats() of r_nu in each true-energy window
  ellipse fraction       per-event membership of (100 r_nu, 100 r_mu) in the frozen ellipse
  migration into D       D = reconstructed trueE in [1, 4); for each true source S:
                         m_S = #(S and D), rate = m_S / N_S, contribution = m_S / N_D
N_D counts every reconstructed-D event of any true energy, so it depends on the model.
Values are fractions; displays multiply by 100. Undefined values are NaN, never 0.
"""
import numpy as np

from pcnumu.core.metrics import WINDOW, interval, stats, wilson68

INF = float("inf")
CUTS = dict(
    units="GeV",
    convention="lower-inclusive, upper-exclusive; null upper edge = no upper limit",
    selection_variable="true trueE (source CSV, Float64); reconstructed = pred_trueE",
    resolution_windows=[[1.0, 3.0], [3.0, 6.0], [0.5, 1.0]],  # plot order: the third bar is [0.5, 1)
    bias_windows=[[1.0, 1.5], [1.5, 2.5], [2.5, 4.0]],
    ellipse_windows=[[1.0, 3.0], [3.0, 6.0]],
    migration_sources=[[4.0, 6.0], [6.0, None]],
    overflow_breakdown=[[6.0, 10.0], [10.0, 20.0], [20.0, None]],
    migration_all=[4.0, None],
    reco_destination=[1.0, 4.0],
    legacy_window=dict(lo=WINDOW[0], hi=WINDOW[1], upper_inclusive=True),
)
LOW_COUNT = 10  # migration numerators below this are flagged as bootstrap-limited
LEGACY_ATOL = 1e-12
PAIRED_METRICS = ("bias", "resolution", "ellipse_fraction", "migration_rate", "contribution")


def edges(w):
    return float(w[0]), (INF if w[1] is None else float(w[1]))


def wkey(w):
    lo, hi = edges(w)
    return f"{lo:g}-{'inf' if hi == INF else f'{hi:g}'}"


def in_window(x, w):
    lo, hi = edges(w)
    return (x >= lo) & (x < hi)


def stat_windows(cuts=CUTS):
    out = []
    for w in cuts["resolution_windows"] + cuts["bias_windows"]:
        if w not in out:
            out.append(w)
    return out


def source_windows(cuts=CUTS):
    return cuts["migration_sources"] + cuts["overflow_breakdown"] + [cuts["migration_all"]]


def model_metrics(t, r, inside, p, cuts=CUTS):
    """Flat {metric|window: scalar} for one model. t, r, inside, p are aligned event arrays
    (true trueE, r_nu, ellipse membership, pred_trueE)."""
    m = {}
    for w in stat_windows(cuts):
        s = stats(r[in_window(t, w)])
        k = wkey(w)
        m[f"n|{k}"] = s["n"]
        m[f"bias|{k}"] = float(s["bias"])
        m[f"resolution|{k}"] = float(s["res"])
    for w in cuts["ellipse_windows"]:
        sel = in_window(t, w)
        n, k = int(sel.sum()), wkey(w)
        m[f"ellipse_n|{k}"] = n
        m[f"ellipse_inside|{k}"] = int(inside[sel].sum())
        m[f"ellipse_fraction|{k}"] = float(inside[sel].mean()) if n else np.nan
    dest = in_window(p, cuts["reco_destination"])
    n_d = int(dest.sum())
    m[f"N_D|{wkey(cuts['reco_destination'])}"] = n_d
    lo_d, hi_d = edges(cuts["reco_destination"])
    for w in source_windows(cuts):
        src, k = in_window(t, w), wkey(w)
        n_s, m_s = int(src.sum()), int((src & dest).sum())
        m[f"N_S|{k}"] = n_s
        m[f"m_S|{k}"] = m_s
        m[f"below_D|{k}"] = int((src & (p < lo_d)).sum())
        m[f"above_D|{k}"] = int((src & (p >= hi_d)).sum())
        m[f"migration_rate|{k}"] = m_s / n_s if n_s else np.nan
        m[f"contribution|{k}"] = m_s / n_d if n_d else np.nan
    return m


def paired_bootstrap(t, events, n_boot, seed, cuts=CUTS, progress=None):
    """Replicates of model_metrics for every model, resampling the same events for all models.

    events = {label: dict(r=..., inside=..., p=...)} aligned with t. Returns {label: {key: array}}.
    """
    rng = np.random.default_rng(seed)
    keys = None
    reps = {}
    for i in range(n_boot):
        idx = rng.integers(0, t.size, t.size)
        tb = t[idx]
        for label, e in events.items():
            mb = model_metrics(tb, e["r"][idx], e["inside"][idx], e["p"][idx], cuts)
            if keys is None:
                keys = list(mb)
            if label not in reps:
                reps[label] = {k: np.empty(n_boot) for k in keys}
            for k in keys:
                reps[label][k][i] = mb[k]
        if progress and (i + 1) % 100 == 0:
            progress(i + 1)
    return reps


def jnum(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if np.isfinite(v) else None


def summarise(point, reps, reference, cuts=CUTS):
    """Point values with 68% intervals per model, plus paired differences against the reference model."""
    dk = wkey(cuts["reco_destination"])
    results, paired = {}, {}
    for label in point:
        results[label], paired[label] = {}, {}
        for k, v in point[label].items():
            entry = dict(value=jnum(v), **interval(reps[label][k]))
            metric, win = k.split("|")
            if metric in ("migration_rate", "contribution"):
                m_s = point[label][f"m_S|{win}"]
                n = point[label][f"N_S|{win}"] if metric == "migration_rate" else point[label][f"N_D|{dk}"]
                entry.update(numerator=m_s, denominator=n, binomial68_wilson=wilson68(m_s, n),
                             binomial_note=None if metric == "migration_rate" else "conditional on observed N_D",
                             bootstrap_limited=bool(m_s < LOW_COUNT or m_s == n))
            results[label][k] = entry
            if label != reference and metric in PAIRED_METRICS:
                d = point[label][k] - point[reference][k]
                paired[label][k] = dict(value=jnum(d), **interval(reps[label][k] - reps[reference][k]))
    return results, paired


def count_checks(point, t, cuts=CUTS):
    """Pilot check 3: count identities that must hold exactly for every model."""
    out = []
    dk = wkey(cuts["reco_destination"])
    src_keys = [wkey(w) for w in cuts["migration_sources"]]
    brk_keys = [wkey(w) for w in cuts["overflow_breakdown"]]
    all_k, over_k = wkey(cuts["migration_all"]), wkey(cuts["migration_sources"][1])
    bias_keys = [wkey(w) for w in cuts["bias_windows"]]
    n_1to4 = int(in_window(t, [cuts["bias_windows"][0][0], cuts["bias_windows"][-1][1]]).sum())
    for label, m in point.items():
        bad = []
        for w in source_windows(cuts):
            k = wkey(w)
            if not (0 <= m[f"m_S|{k}"] <= m[f"N_S|{k}"] and m[f"m_S|{k}"] <= m[f"N_D|{dk}"]):
                bad.append(f"bounds {k}")
            if m[f"m_S|{k}"] + m[f"below_D|{k}"] + m[f"above_D|{k}"] != m[f"N_S|{k}"]:
                bad.append(f"below+in+above != N_S for {k}")
        for q in ("m_S", "N_S"):
            if sum(m[f"{q}|{k}"] for k in src_keys) != m[f"{q}|{all_k}"]:
                bad.append(f"sources do not sum to >=4 for {q}")
            if sum(m[f"{q}|{k}"] for k in brk_keys) != m[f"{q}|{over_k}"]:
                bad.append(f"overflow breakdown does not sum for {q}")
        if m[f"N_D|{dk}"] and not np.isclose(sum(m[f"contribution|{k}"] for k in src_keys),
                                             m[f"contribution|{all_k}"], rtol=0, atol=1e-12):
            bad.append("contributions do not add to the >=4 contribution")
        if m[f"N_D|{dk}"] and not np.isclose(sum(m[f"contribution|{k}"] for k in brk_keys),
                                             m[f"contribution|{over_k}"], rtol=0, atol=1e-12):
            bad.append("overflow breakdown contributions do not add")
        if sum(m[f"n|{k}"] for k in bias_keys) != n_1to4:
            bad.append("bias windows do not partition [1, 4)")
        out.append(dict(name=f"3 counts {label}", passed=not bad, detail=bad or "ok"))
    labels = list(point)
    common = [k for k in point[labels[0]] if k.split("|")[0] in ("n", "ellipse_n", "N_S")]
    diff = [k for k in common if any(point[x][k] != point[labels[0]][k] for x in labels[1:])]
    out.append(dict(name="3 true-source and window counts match across models", passed=not diff, detail=diff or "ok"))
    return out
