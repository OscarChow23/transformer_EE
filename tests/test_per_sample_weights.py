"""PerSampleFlatWeights (r08): weights equal the agreed recipe, are fitted on train only, and fail loudly on bad input.

The spec is `reference()` below, copied from the r08 preview
(pcnumu/studies/20261110_r08-flattening-preview/plot_regimes.py: weights()).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformer_ee.utils.weights import PerSampleFlatWeights, create_weighter  # noqa: E402

FHC = {"1": {"lo": 1.0, "hi": 5.0, "ease": 3.0, "alpha": 0.05},
       "2": {"lo": 1.0, "hi": 5.0, "ease": 3.0, "alpha": 0.02},
       "3": {"lo": 1.0, "hi": 5.0, "ease": 3.0, "alpha": 0.02}}
CACHE = "/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{}.npz"


def reference(y, lo, hi, taper, alpha, width=0.1):
    """The agreed recipe for one sample's training rows (spec; do not 'fix' it here)."""
    e = np.arange(lo, hi + taper + 1e-9, width)
    c, _ = np.histogram(y, e)
    wb = 1.0 / (c + max(alpha * c.max(), 1))
    w = wb[np.clip(np.digitize(y, e) - 1, 0, len(wb) - 1)]
    core = (y >= lo) & (y < hi)
    w = w * (core.sum() / w[core].sum())
    tz = (y >= hi) & (y < hi + taper)
    f = (y[tz] - hi) / taper
    w[tz] = w[tz] ** (1 - f)
    w[(y >= hi + taper) | (y < lo)] = 1.0
    return w * (y.size / w.sum())


def toy(n=60000, seed=0):
    """Peaked spectra with long tails, one per sample (unskippable stand-in for the real cache)."""
    rng = np.random.default_rng(seed)
    s = rng.choice([1, 2, 3], n, p=[0.7, 0.08, 0.22])
    y = np.where(s == 1, rng.gamma(9, 0.22, n), rng.gamma(4, 0.6, n)) + 0.3
    return y, s


def expected(y, s, params):
    out = np.empty_like(y)
    for k, p in params.items():
        m = s == int(k)
        out[m] = reference(y[m], p["lo"], p["hi"], p["ease"], p["alpha"])
    return out


class MatchesSpec(unittest.TestCase):
    def test_toy_data(self):
        y, s = toy()
        w = PerSampleFlatWeights(y, s, FHC).getweight(y, s)
        np.testing.assert_allclose(w, expected(y, s, FHC), rtol=1e-12, atol=0)

    def test_cached_training_rows(self):
        for horn, lo1 in (("fhc", 1.0), ("rhc", 0.9)):
            if not Path(CACHE.format(horn)).exists():
                self.skipTest("cache absent (the toy test still covers the spec)")
            d = np.load(CACHE.format(horn))
            y, s = d["y"], d["s"]
            params = {k: dict(v) for k, v in FHC.items()}
            params["1"]["lo"] = lo1
            w = PerSampleFlatWeights(y, s, params).getweight(y, s)
            np.testing.assert_allclose(w, expected(y, s, params), rtol=1e-12, atol=0)


class Behaviour(unittest.TestCase):
    def setUp(self):
        self.y, self.s = toy()
        self.wt = PerSampleFlatWeights(self.y, self.s, FHC)
        self.w = self.wt.getweight(self.y, self.s)

    def test_mean_one_per_sample(self):
        for k in (1, 2, 3):
            self.assertAlmostEqual(self.w[self.s == k].mean(), 1.0, places=12)

    def test_subset_and_scalar_equal_full_fit(self):
        sub = np.arange(0, self.y.size, 7)
        np.testing.assert_array_equal(self.wt.getweight(self.y[sub], self.s[sub]), self.w[sub])
        for i in sub[:200]:
            self.assertEqual(self.wt.getweight(float(self.y[i]), int(self.s[i])), self.w[i])

    def test_other_rows_use_the_fitted_map(self):
        """Valid-like rows from a different spectrum get the train map, not a refit."""
        yv, sv = toy(5000, seed=1)
        yv = yv * 1.3
        wv = self.wt.getweight(yv, sv)
        ref = PerSampleFlatWeights(self.y, self.s, FHC)  # same fit
        np.testing.assert_array_equal(wv, ref.getweight(yv, sv))
        self.assertFalse(np.allclose(wv, PerSampleFlatWeights(yv, sv, FHC).getweight(yv, sv)))

    def test_shape(self):
        for k in (1, 2, 3):
            scale = self.wt.scale[k]
            p = FHC[str(k)]
            out = np.array([p["lo"] - 0.2, p["hi"] + p["ease"], 40.0])
            np.testing.assert_allclose(self.wt.getweight(out, np.full(3, k)), scale, rtol=1e-12)
            # the ease-off ends where "outside" starts: no step at hi + ease
            end = self.wt.getweight(np.array([p["hi"] + p["ease"] - 1e-7]), np.array([k]))[0]
            self.assertAlmostEqual(end / scale, 1.0, places=5)

    def test_label_forms(self):
        y = self.y[:5]
        base = self.wt.getweight(y, np.array([1, 2, 3, 1, 2]))
        for labels in (np.array([1.0, 2.0, 3.0, 1.0, 2.0]), np.array(["1", "2", "3", "1", "2"]), [1, 2, 3, 1, 2]):
            np.testing.assert_array_equal(self.wt.getweight(y, labels), base)

    def test_bad_labels(self):
        y = self.y[:2]
        for bad in ([1, 4], [1.5, 2], [True, False], [1, np.nan], ["1", "S2"], [1, None]):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.wt.getweight(y, np.array(bad, dtype=object) if None in bad else np.array(bad))

    def test_bad_fits(self):
        y, s = self.y, self.s
        with self.assertRaises(ValueError):  # configured sample with no rows
            PerSampleFlatWeights(y[s != 2], s[s != 2], FHC)
        with self.assertRaises(ValueError):  # rows from an unconfigured sample
            PerSampleFlatWeights(y, s, {"1": FHC["1"], "2": FHC["2"]})
        with self.assertRaises(ValueError):  # empty core
            PerSampleFlatWeights(y, s, {**FHC, "3": {"lo": 50.0, "hi": 60.0, "ease": 3.0, "alpha": 0.02}})
        with self.assertRaises(ValueError):  # missing / misspelt parameter
            PerSampleFlatWeights(y, s, {**FHC, "3": {"lo": 1.0, "hi": 5.0, "alpha": 0.02}})

    def test_int_and_str_config_keys_agree(self):
        as_int = {int(k): v for k, v in FHC.items()}
        np.testing.assert_array_equal(PerSampleFlatWeights(self.y, self.s, as_int).getweight(self.y, self.s), self.w)

    def test_create_weighter(self):
        cfg = {"weight": {"name": "PerSampleFlatWeights", "kwargs": {"per_sample": FHC}}}
        np.testing.assert_array_equal(create_weighter(cfg, self.y, sample=self.s).getweight(self.y, self.s), self.w)
        with self.assertRaises(ValueError):
            create_weighter(cfg, self.y)  # pandas path / any caller without samples
        self.assertTrue(create_weighter(cfg, self.y, sample=self.s).needs_sample)

    def test_hand_computed(self):
        """Independent oracle: weights worked out by hand for a tiny sample (lo 1, hi 2, ease 1, alpha 0)."""
        y = np.array([1.05] * 10 + [1.55] * 5 + [2.5] * 2 + [0.5, 4.0])
        wt = PerSampleFlatWeights(y, np.ones(y.size, int), {"1": {"lo": 1.0, "hi": 2.0, "ease": 1.0, "alpha": 0.0}})
        # bins of 0.1 over [1, 3): counts 10 (1.0-1.1), 5 (1.5-1.6), 2 (2.5-2.6); w_bin = 1/(count + 1)
        norm = 15 / (10 / 11 + 5 / 6)                    # the 15 core rows average 1
        b0, b5, b15, empty = norm / 11, norm / 6, norm / 3, norm / 1
        ease = b15 ** (1 - 0.5)                           # 2.5 GeV is halfway through the ease-off
        scale = 19 / (10 * b0 + 5 * b5 + 2 * ease + 1 + 1)
        np.testing.assert_allclose(wt.getweight(y, np.ones(y.size, int)),
                                   scale * np.array([b0] * 10 + [b5] * 5 + [ease] * 2 + [1, 1]), rtol=1e-12)
        # rows it was not fitted on: an empty core bin, an occupied bin, the ease-off, outside (below lo, at hi+ease).
        # Bin edges come from arange (as in the preview), so a value exactly on a nominal edge may fall either side.
        q = np.array([1.25, 1.55, 2.25, 0.99, 3.0])
        want = scale * np.array([empty, b5, empty ** 0.75, 1, 1])
        np.testing.assert_allclose(wt.getweight(q, np.ones(q.size, int)), want, rtol=1e-12)

    def test_bad_params_and_inputs(self):
        y, s = self.y, self.s
        for bad in ({"share": float("nan")}, {"alpha": float("nan")}, {"alpha": -0.1}, {"ease": 0.0}, {"hi": 1.0}):
            params = {k: {**v, "share": 0.3} for k, v in FHC.items()} if "share" in bad else dict(FHC)
            params["2"] = {**params["2"], **bad}
            with self.assertRaises(ValueError, msg=str(bad)):
                PerSampleFlatWeights(y, s, params)
        with self.assertRaises(ValueError):  # 1 and "1" are the same sample
            PerSampleFlatWeights(y, s, {**FHC, 1: FHC["1"]})
        with self.assertRaises(ValueError):
            PerSampleFlatWeights(y, s, {True: FHC["1"], "2": FHC["2"], "3": FHC["3"]})
        with self.assertRaises(ValueError):
            self.wt.getweight(np.array([1.5, np.nan]), np.array([1, 1]))

    def test_share(self):
        """share fixes each sample's fraction of the total weight; the shape within a sample is unchanged."""
        shares = {"1": 0.5467, "2": 0.0494, "3": 0.4039}
        params = {k: {**v, "share": shares[k]} for k, v in FHC.items()}
        w2 = PerSampleFlatWeights(self.y, self.s, params).getweight(self.y, self.s)
        for k in (1, 2, 3):
            m = self.s == k
            self.assertAlmostEqual(w2[m].sum() / w2.sum(), shares[str(k)] / sum(shares.values()), places=12)
            ratio = w2[m] / self.w[m]
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
        self.assertAlmostEqual(w2.mean(), 1.0, places=12)
        with self.assertRaises(ValueError):  # share for some samples only
            PerSampleFlatWeights(self.y, self.s, {**FHC, "1": {**FHC["1"], "share": 0.5}})
        with self.assertRaises(ValueError):
            PerSampleFlatWeights(self.y, self.s, {k: {**v, "share": 0.0} for k, v in FHC.items()})


if __name__ == "__main__":
    unittest.main()
