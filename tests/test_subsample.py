"""The loader's `subsample` cut drops only the hashed S1 train/valid rows and never touches test."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pcnumu/eval"))
from pcnumu.prep.make_split_csv import assign_split  # noqa: E402
from r07_s1_subsample_preview import packed_key, unit_hash  # noqa: E402  (independent copy used for the preview)
from transformer_ee.dataloader.load import get_train_valid_test_dataloader  # noqa: E402

SUB = {"sample": 1, "keep": 0.5, "seed": 20261104, "splits": ["train", "valid"]}


def frame(n=20000):
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"run": rng.integers(1, 10**6, n), "subrun": rng.integers(0, 64, n),
                       "cycle": rng.integers(0, 1000, n), "evt": rng.integers(0, 10**5, n),
                       "sample": rng.integers(1, 4, n), "trueE": rng.random(n)})
    return df.with_columns(pl.Series("split", assign_split(df, 20261029, 0.20, 0.04)))


def load(df, **cfg):
    seen = []
    config = {"dataframe_type": "polars", "data_path": "x", "batch_size_train": 1, "batch_size_valid": 1,
              "batch_size_test": 1, **cfg}
    with patch("transformer_ee.dataloader.pl_dataset.get_polars_df_from_file", return_value=df), \
            patch("transformer_ee.dataloader.pl_dataset.Normalized_Polars_Dataset_with_cache",
                  side_effect=lambda c, d, **k: seen.append(d) or MagicMock()), \
            patch("torch.utils.data.DataLoader"):
        get_train_valid_test_dataloader(config)
    return seen  # train, valid, test frames


class SubsampleTests(unittest.TestCase):
    def test_cut(self):
        df = frame()
        train, valid, test = load(df, split_column="split", subsample=SUB)
        self.assertTrue(test.equals(df.filter(pl.col("split") == "test")))
        ids = ["run", "subrun", "cycle", "evt"]
        u = unit_hash(packed_key(*(df[c].cast(pl.UInt64).to_numpy() for c in ids)), SUB["seed"])
        expect = df.filter(pl.Series((df["sample"].to_numpy() != 1) | (u < 0.5)))
        for name, got in (("train", train), ("valid", valid)):
            self.assertTrue(got.equals(expect.filter(pl.col("split") == name)), name)
        s1 = (df["sample"] == 1) & (df["split"] == "train")
        self.assertAlmostEqual((train["sample"] == 1).sum() / s1.sum(), 0.5, delta=0.03)

    def test_guards(self):
        with self.assertRaises(ValueError):
            load(frame(100), split_column="split", subsample={**SUB, "splits": ["test"]})
        with self.assertRaises(ValueError):
            load(frame(100), subsample=SUB)


if __name__ == "__main__":
    unittest.main()
