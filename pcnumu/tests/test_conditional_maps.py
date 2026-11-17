"""Overflow/edge regression check for eval_by_sample.conditional_maps.

    python -I pcnumu/tests/test_conditional_maps.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")]
from eval_by_sample import conditional_maps  # noqa: E402
from pcnumu.core import style  # noqa: E402  (importable once eval_by_sample has set up the path)


def test_conditional_maps_edges_and_overflow():
    style.use()
    with tempfile.TemporaryDirectory() as out:
        x = np.tile([1., 1., 10., 11.], 3)
        y = np.tile([1., 20., 10., 11.], 3)
        conditional_maps(x, y, np.repeat(["1", "2", "3"], 4), out)
        with np.load(Path(out) / "sample3_pred_vs_true_trueE_counts.npz") as data:
            assert data["slice_counts"].sum() == 3
            assert data["counts"].sum() == 2
            assert data["n_x_outside"] == 1 and data["n_y_outside"] == 1
            assert np.isclose(data["conditional_probability"].sum(), 1.5)


if __name__ == "__main__":
    test_conditional_maps_edges_and_overflow()
    print("ok test_conditional_maps_edges_and_overflow")
