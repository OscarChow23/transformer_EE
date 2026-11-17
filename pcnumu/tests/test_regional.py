"""Synthetic checks for pcnumu.core.regional: window edges, denominators, ellipse boundary, empty windows.

    python -I pcnumu/tests/test_regional.py      (or pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pcnumu.core.metrics import interval, wilson68  # noqa: E402
from pcnumu.core.regional import CUTS, in_window, model_metrics, wkey  # noqa: E402


def test_window_edges_lower_inclusive_upper_exclusive():
    x = np.array([0.4999, 0.5, 0.9999, 1.0, 1.5, 2.5, 3.0, 4.0, 6.0, 60.0])
    assert in_window(x, [0.5, 1.0]).tolist() == [False, True, True] + [False] * 7
    assert in_window(x, [3.0, 6.0]).tolist() == [False] * 6 + [True, True, False, False]
    assert in_window(x, [6.0, None]).tolist() == [False] * 8 + [True, True]  # 6 is overflow


def test_boundaries_each_land_in_exactly_one_partition_cell():
    bias = CUTS["bias_windows"]
    for v in (1.0, 1.5, 2.5, 3.9999):
        assert sum(bool(in_window(np.array([v]), w)[0]) for w in bias) == 1
    assert not any(in_window(np.array([4.0]), w)[0] for w in bias)
    src = CUTS["migration_sources"]
    for v in (4.0, 5.9999, 6.0, 1e3):
        assert sum(bool(in_window(np.array([v]), w)[0]) for w in src) == 1
    for v in (6.0, 10.0, 20.0, 1e3):
        assert sum(bool(in_window(np.array([v]), w)[0]) for w in CUTS["overflow_breakdown"]) == 1


def _metrics(t, p, inside=None):
    t, p = np.asarray(t, float), np.asarray(p, float)
    r = (p - t) / t
    inside = np.zeros(t.size, bool) if inside is None else np.asarray(inside)
    return model_metrics(t, r, inside, p, CUTS)


def test_migration_denominators():
    # true 2 is in no source but lands in D; 5->3 migrates, 5->5 stays above, 7->3.99 migrates, 25->0.5 falls below
    m = _metrics([2.0, 5.0, 5.0, 7.0, 25.0], [2.0, 3.0, 5.0, 3.99, 0.5])
    assert m["N_D|1-4"] == 3  # pred 2, 3, 3.99: N_D counts every true energy, including the true-2 event
    assert (m["N_S|4-6"], m["m_S|4-6"], m["above_D|4-6"]) == (2, 1, 1)
    assert (m["N_S|6-inf"], m["m_S|6-inf"], m["below_D|6-inf"]) == (2, 1, 1)
    assert m["migration_rate|4-6"] == 0.5
    assert np.isclose(m["contribution|4-6"], 1 / 3)
    assert np.isclose(m["contribution|4-6"] + m["contribution|6-inf"], m["contribution|4-inf"])
    assert m["N_S|20-inf"] == 1 and m["m_S|20-inf"] == 0


def test_empty_windows_are_undefined_not_zero():
    m = _metrics([2.0], [2.0])
    assert m["n|0.5-1"] == 0 and np.isnan(m["resolution|0.5-1"]) and np.isnan(m["bias|0.5-1"])
    assert m["ellipse_n|3-6"] == 0 and np.isnan(m["ellipse_fraction|3-6"])
    assert m["N_S|4-6"] == 0 and np.isnan(m["migration_rate|4-6"])
    m = _metrics([5.0], [10.0])  # nothing reconstructed in D
    assert m["N_D|1-4"] == 0 and np.isnan(m["contribution|4-6"])


def test_ellipse_fraction_uses_event_denominator_and_closed_boundary():
    a = b = 22.0
    x = np.array([22.0, 0.0, 30.0, 500.0])  # on the circle, centre, outside, far outside (beyond histogram range)
    y = np.array([0.0, 22.0, 0.0, 0.0])
    inside = x * x / (a * a) + y * y / (b * b) <= 1.0
    assert inside.tolist() == [True, True, False, False]
    m = _metrics([2.0] * 4, [2.0] * 4, inside)
    assert m["ellipse_n|1-3"] == 4 and m["ellipse_fraction|1-3"] == 0.5


def test_interval_counts_undefined_replicates():
    e = interval([np.nan, 1.0, 2.0, 3.0, np.nan])
    assert e["n_valid_reps"] == 3 and e["n_undefined_reps"] == 2
    assert interval([np.nan])["lo68"] is None


def test_wilson_interval():
    assert wilson68(0, 0) is None
    lo, hi = wilson68(0, 100)
    assert lo == 0.0 or abs(lo) < 1e-15
    assert hi > 0  # never a zero-width interval at m = 0
    lo, hi = wilson68(50, 100)
    assert lo < 0.5 < hi


def test_window_keys():
    assert wkey([6.0, None]) == "6-inf" and wkey([0.5, 1.0]) == "0.5-1"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
