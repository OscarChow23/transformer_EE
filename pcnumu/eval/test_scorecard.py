"""Unit checks for scorecard.py's ports of the ROOT macros and its train.log parser.

    python pcnumu/eval/test_scorecard.py      (or pytest)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorecard import _root_axis, _root_findbin, ellipse_axes, ellipse_fraction, errmap_bins, parse_losses  # noqa: E402


def test_findbin_matches_taxis_edges():
    v = np.array([-1.0, 0.0, 0.0999, 0.1, 9.99, 10.0, 11.0])
    # 100 bins on [0, 10): x == xmax is overflow, x < xmin underflow
    assert _root_findbin(v, 0.0, 10.0, 100).tolist() == [0, 1, 1, 2, 100, 101, 101]


def test_ellipse_equals_literal_bin_loop():
    """Same result as transcribing fractionInsideEllipse's loop over a filled TH2D."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 12, 20000), [350.0, -500.0]])  # two entries beyond ±200
    y = np.concatenate([rng.normal(0, 8, 20000), [0.0, 0.0]])
    a, b, nb = 10.0, 10.0, 200
    got = ellipse_fraction(x, y, a, b, nb)

    (xl, xh, _), (yl, yh, _) = _root_axis(x, nb), _root_axis(y, nb)
    h = np.zeros((nb + 2, nb + 2))
    np.add.at(h, (_root_findbin(x, xl, xh, nb), _root_findbin(y, yl, yh, nb)), 1.0)
    total = inside = 0.0
    for ix in range(1, nb + 1):          # in-range bins only, as the C loop
        for iy in range(1, nb + 1):
            cx = xl + (ix - 0.5) * (xh - xl) / nb
            cy = yl + (iy - 0.5) * (yh - yl) / nb
            total += h[ix, iy]
            if cx * cx / (a * a) + cy * cy / (b * b) <= 1.0:
                inside += h[ix, iy]
    assert got["n_in_histogram"] == total == 20000
    assert got["n_out_of_range"] == 2
    assert abs(got["fraction"] - inside / total) < 1e-12


def test_parse_losses_uses_last_session():
    text = ("Start Time: a\nEpoch: 0, train_loss: 0.5, valid_loss: 0.4\nEpoch: 1, batch: 3 Loss: 0.1\n"
            "Epoch: 1, train_loss: 0.3, valid_loss: 0.35\n"
            "Start Time: b\nEpoch: 0, train_loss: 0.9, valid_loss: 0.8\n")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text)
    try:
        losses, n = parse_losses(f.name)
    finally:
        os.unlink(f.name)
    assert n == 2 and losses == [(0, 0.9, 0.8)]


def test_errmap_drops_xmax_like_fixed_range_view():
    b, y = errmap_bins(np.array([-0.1, 0.0, 0.05, 5.99, 6.0, 7.0]), np.arange(6.0))
    assert b.tolist() == [0, 0, 59] and y.tolist() == [1.0, 2.0, 3.0]


def test_ellipse_axes_frozen_only_for_listed_test_sets():
    fhc = "20be926adaf557259fb74d05552e3337fc94c04fd2e813b6f29850afc9b43118"
    got = ellipse_axes(fhc)
    assert (got["a_percent"], got["b_percent"], got["provisional"], got["frozen_as"]) == (22.0, 22.0, False, "training_v3/fhc")
    other = ellipse_axes("0" * 64)
    assert other["provisional"] is True and other["frozen_as"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
