"""PROPOSAL: unit tests that pin peak_metrics version 2. Synthetic events only, so they run anywhere:

    python -I test_peak_metrics_v2.py -v

Each test is one of the injection cases of the 11 Nov 2026 study (report.md), reduced to a closed form:
a spectrum with a peak near 2 GeV, lognormal errors, and a pull written as a known shift in ln(pred).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peak_metrics_v2 import peak_measures_v2  # noqa: E402

SIG = 0.15


def events(n=400000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.concatenate([rng.gamma(9.0, 0.24, int(0.8 * n)), rng.uniform(1.0, 5.0, n - int(0.8 * n))])
    t = t[(t > 0.8) & (t < 5.5)]
    return t, rng


def pull(t, size):
    """Known pull in ln(pred): +size at 1.25 GeV, 0 at 2 GeV, -size/2 at 3.5 GeV (piecewise linear in E)."""
    return np.interp(t, [1.0, 1.25, 2.0, 3.5, 5.0], [size, size, 0.0, -size / 2, -size / 2])


class PeakMetricsV2(unittest.TestCase):
    def setUp(self):
        self.t, self.rng = events()
        self.eps = SIG * self.rng.standard_normal(self.t.size)
        self.score = np.abs(self.rng.standard_normal(self.t.size))  # unrelated to the error unless a test says so

    def run_m(self, ln_shift=0.0, eps=None, score=None):
        p = self.t * np.exp((self.eps if eps is None else eps) + ln_shift)
        return peak_measures_v2(self.t, p, SIG, self.score if score is None else score)

    def test_a_no_pull_reads_zero(self):
        m = self.run_m()
        for k in ("low_vs_peak", "peak_vs_high", "low_vs_peak_median", "peak_vs_high_median"):
            self.assertLess(abs(m[k]), 0.5, k)
        self.assertLess(abs(m["net_towards_low"]), 2.0)
        self.assertLess(abs(m["net_towards_high"]), 2.0)

    def test_b_known_pull_is_recovered_and_rises(self):
        last = -1.0
        for size in (0.02, 0.05, 0.10):
            m = self.run_m(pull(self.t, size))
            lo = (self.t >= 1.0) & (self.t < 1.5)
            pk = (self.t >= 1.75) & (self.t < 2.25)
            want = 100 * (pull(self.t, size)[lo].mean() - pull(self.t, size)[pk].mean())
            self.assertAlmostEqual(m["low_vs_peak"], want, delta=0.5)
            self.assertGreater(m["low_vs_peak"], last)
            self.assertGreater(m["peak_vs_high"], 0)
            last = m["low_vs_peak"]

    def test_c_pull_in_a_subset_mean_sees_it_median_does_not(self):
        sub = self.rng.random(self.t.size) < 0.3
        # the pulled 30% are also the badly measured ones (errors 3x wider), as for real uncertain events
        m = self.run_m(np.where(sub, pull(self.t, 0.30), 0.0), eps=np.where(sub, 3.0, 1.0) * self.eps,
                       score=np.where(sub, 2.0, 1.0) + 0.01 * self.score)
        lo = (self.t >= 1.0) & (self.t < 1.5)
        want = 100 * 0.3 * pull(self.t, 0.30)[lo].mean()
        self.assertAlmostEqual(m["low_vs_peak"], want, delta=1.0)        # mean: the full average pull (about 8)
        self.assertLess(m["low_vs_peak_median"], 0.6 * m["low_vs_peak"])  # median: under half of it
        self.assertGreater(m["uncertain_vs_certain"]["low"], 20)          # the pulled events are the uncertain ones

    def test_e_scale_shift_is_not_pull(self):
        a, b = self.run_m(), self.run_m(np.log(1.03))
        for k in ("low_vs_peak", "peak_vs_high", "net_towards_low", "net_towards_high"):
            self.assertAlmostEqual(a[k], b[k], delta=0.3, msg=k)
        self.assertAlmostEqual(b["level_peak"] - a["level_peak"], 100 * np.log(1.03), delta=0.05)

    def test_f_broadening_is_not_pull(self):
        m = self.run_m(eps=1.5 * self.eps)
        self.assertLess(abs(m["low_vs_peak"]), 0.6)
        self.assertLess(abs(m["peak_vs_high"]), 0.6)

    def test_d_tilt_IS_read_as_pull_documented_limit(self):
        """A plain energy-dependent bias cannot be told from pull by a bias curve. Pinned so nobody assumes otherwise."""
        tilt = np.log(1 - 0.05 * (self.t - 1) / 3)
        m = self.run_m(tilt)
        self.assertGreater(m["peak_vs_high"], 1.5)
        u = m["uncertain_vs_certain"]  # ... but the uncertainty gradient does not move
        self.assertLess(abs(u["high"] - u["peak"]), 1.0)

    def test_g_high_side_overestimates_do_not_touch_the_low_side(self):
        base = self.run_m(pull(self.t, 0.08))
        hit = (self.t > 2.5) & (self.rng.random(self.t.size) < 0.15)
        m = self.run_m(pull(self.t, 0.08) + np.where(hit, np.log(1.25), 0.0))
        self.assertAlmostEqual(m["low_vs_peak"], base["low_vs_peak"], delta=0.05)
        self.assertLess(m["peak_vs_high"], base["peak_vs_high"] - 2.0)   # shows on the high side only

    def test_v1_identity(self):
        p = self.t * np.exp(self.eps + pull(self.t, 0.05))
        r = p / self.t - 1
        v1 = 100 * (np.median(r[(self.t >= 1.0) & (self.t < 1.5)]) - np.median(r[(self.t >= 3.0) & (self.t < 4.0)]))
        m = peak_measures_v2(self.t, p, SIG)
        self.assertAlmostEqual(m["low_vs_peak_median"] + m["peak_vs_high_median"], v1, delta=0.002)
        self.assertIsNone(m["uncertain_vs_certain"])


if __name__ == "__main__":
    unittest.main()
