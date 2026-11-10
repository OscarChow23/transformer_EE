"""
Calculate the weights for the loss function.
"""

import numpy as np


class NullWeights:
    """
    A null weighter.
    """

    def getweight(self, _):
        """
        Return 1 for all weights.
        """
        return 1.0


class FlatSpectraWeights:
    """
    Initialize the weights with an array, then return the weights (as array) for a given value x.
    The weighter will calculate the weight based on the following function using the histogram count:
        w = 1 / (count + alpha * max (count))
    In addition, the weighter will normalize the weights to have a mean of 1 and clip the weights to a given range.
    """

    def __init__(  # pylint: disable=W0622
        self, array, bins=50, range=None, maxweight=np.inf, minweight=-np.inf, alpha=0
    ) -> None:
        self.hist = np.histogram(array, bins=bins, range=range)
        _alpha = max(self.hist[0]) * alpha
        if _alpha < 1:
            _alpha = 1
        self.weights = 1 / (
            self.hist[0] + _alpha
        )  # Add _alpha > 0 to avoid division by zero
        self.weights = self.weights / np.mean(self.weights)
        self.bins = self.hist[1]
        self.weights[self.weights > maxweight] = maxweight
        self.weights[self.weights < minweight] = minweight
        self.bins[0] = -np.inf
        self.bins[-1] = np.inf

    def getweight(self, x):
        return self.weights[np.digitize(x, self.bins) - 1]


def _sample_labels(sample, allowed):
    """Integer sample labels; rejects bool, fractional, non-finite, missing and unconfigured labels."""
    flat = np.atleast_1d(np.asarray(sample, dtype=object)).ravel()
    out = np.empty(flat.size, dtype=np.int64)
    for i, v in enumerate(flat):
        if v is None or isinstance(v, (bool, np.bool_)):
            raise ValueError(f"bad sample label {v!r}")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"bad sample label {v!r}") from None
        if not np.isfinite(f) or f != int(f) or int(f) not in allowed:
            raise ValueError(f"bad sample label {v!r}; expected one of {sorted(allowed)}")
        out[i] = int(f)
    return out


class PerSampleFlatWeights:
    """
    Flatten each sample's own trueE spectrum over [lo, hi), easing back over [hi, hi+ease), then
    scale so each sample's weights average 1 on the rows it was fitted on (train). Per sample:
        w_bin = 1 / (count + max(alpha * max(count), 1))   on 0.1 GeV bins over [lo, hi + ease)
        w     = w_bin * norm                               norm: the flat part [lo, hi) averages 1
        w     = w ** (1 - (x - hi) / ease)                 in [hi, hi + ease): log-linear back to 1
        w     = 1                                          below lo and from hi + ease
        w     = w * scale                                  scale: the sample's weights average 1
    Optional "share" (all samples or none): each sample's scale is then multiplied by a constant so its
    weights make up that fraction of the total train weight (shares normalised to sum 1; overall mean stays 1).
    The shape within a sample is unchanged. Fitted once; getweight applies the stored map, so valid/test
    rows are never refitted.
    `per_sample` = {"1": {"lo": .., "hi": .., "ease": .., "alpha": ..[, "share": ..]}, ...} (int or str keys).
    """

    needs_sample = True
    KEYS = {"lo", "hi", "ease", "alpha"}

    def __init__(self, array, sample, per_sample, width=0.1):
        params = {}
        for k, p in per_sample.items():
            if set(p) - {"share"} != self.KEYS:
                raise ValueError(f"sample {k}: expected keys {sorted(self.KEYS)} (+ optional share), got {sorted(p)}")
            key = int(_sample_labels([k], {int(float(k))})[0])  # bool/fractional keys rejected like labels
            if key in params:
                raise ValueError(f"sample key {k!r} given twice (int and str keys normalise to the same sample)")
            q = {n: float(p[n]) for n in p}
            if not all(np.isfinite(v) for v in q.values()) or not (q["hi"] > q["lo"] and q["ease"] > 0 and q["alpha"] >= 0):
                raise ValueError(f"sample {k}: need finite values, hi > lo, ease > 0, alpha >= 0; got {p}")
            params[key] = q
        if not (np.isfinite(width) and width > 0):
            raise ValueError(f"bad bin width {width}")
        with_share = [k for k, p in params.items() if "share" in p]
        if with_share and (len(with_share) != len(params) or any(params[k]["share"] <= 0 for k in params)):
            raise ValueError("share must be given, and positive, for every sample or for none")
        y = np.asarray(array, dtype=np.float64)
        s = _sample_labels(sample, set(params))
        self.params, self.edges, self.wbin, self.scale = params, {}, {}, {}
        for k, p in params.items():
            yk = y[s == k]
            if not yk.size:
                raise ValueError(f"sample {k} is configured but has no rows to fit on")
            e = np.arange(p["lo"], p["hi"] + p["ease"] + 1e-9, width)
            c, _ = np.histogram(yk, e)
            wb = 1.0 / (c + max(p["alpha"] * c.max(), 1))
            core = (yk >= p["lo"]) & (yk < p["hi"])
            if not core.any():
                raise ValueError(f"sample {k}: no rows in [{p['lo']}, {p['hi']})")
            idx = np.clip(np.digitize(yk[core], e) - 1, 0, len(wb) - 1)
            self.edges[k], self.wbin[k], self.scale[k] = e, wb * (core.sum() / wb[idx].sum()), 1.0
            self.scale[k] = yk.size / self._apply(k, yk).sum()
        if with_share:
            total = sum(p["share"] for p in params.values())
            for k, p in params.items():
                self.scale[k] *= (p["share"] / total) * s.size / np.count_nonzero(s == k)
        self.summary(y, s)

    def _apply(self, k, x):
        p = self.params[k]
        w = self.wbin[k][np.clip(np.digitize(x, self.edges[k]) - 1, 0, len(self.wbin[k]) - 1)]
        tz = (x >= p["hi"]) & (x < p["hi"] + p["ease"])
        w[tz] = w[tz] ** (1 - (x[tz] - p["hi"]) / p["ease"])
        w[(x >= p["hi"] + p["ease"]) | (x < p["lo"])] = 1.0
        return w * self.scale[k]

    def getweight(self, x, sample):
        """Weight(s) for trueE x of the given sample(s); scalar in, scalar out."""
        xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if not np.isfinite(xa).all():
            raise ValueError("non-finite trueE has no defined weight")
        s = _sample_labels(sample, set(self.params))
        if s.size != xa.size:
            raise ValueError(f"{xa.size} values but {s.size} sample labels")
        w = np.empty_like(xa)
        for k in self.params:
            m = s == k
            if m.any():
                w[m] = self._apply(k, xa[m])
        return float(w[0]) if np.ndim(x) == 0 else w

    def summary(self, y, s):
        """Printed at fit time (lands in train.log) so a live job can be checked against the preview."""
        total = self.getweight(y, s).sum()
        for k, p in sorted(self.params.items()):
            yk = y[s == k]
            w = self.getweight(yk, np.full(yk.size, k))
            peak = w[(yk > 1.8) & (yk < 2.2)]
            ess = w.sum() ** 2 / (w ** 2).sum() / w.size
            print(f"PerSampleFlatWeights sample {k}: {p} rows {yk.size} mean {w.mean():.4f} "
                  f"max/peak {w.max() / peak.mean() if peak.size else float('nan'):.1f} ESS {ess:.1%} "
                  f"share of weight {w.sum() / total:.2%}")


def create_weighter(config: dict, array, sample=None):
    """
    Create a weighter from a config.
    Args:
        "config":
        {
            "weight":
            {
                name: the name of weighter.
                kwargs: the arguments of weighter.
                    For FlatSpectraWeights, kwargs should be:
                    {
                        "bins": the number of bins for the histogram.
                        "range": the range of the histogram. eg. (0, 5)
                        ...
                    }
            ...
            }
        }
    """
    if "weight" not in config:
        return NullWeights()

    _kwgs = config["weight"].get("kwargs", {})
    if config["weight"]["name"] == "FlatSpectraWeights":
        weighter = FlatSpectraWeights(array, **_kwgs)
    elif config["weight"]["name"] == "PerSampleFlatWeights":
        if sample is None:
            raise ValueError("PerSampleFlatWeights needs each row's sample label (polars loader only)")
        weighter = PerSampleFlatWeights(array, sample, **_kwgs)
    else:
        raise ValueError(f"Unsupported weighter: {config['weight']['name']}")
    return weighter
