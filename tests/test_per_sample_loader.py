"""PerSampleFlatWeights through the real polars loaders (num_workers=2): every collated weight in train and
valid batches equals the train-fitted recipe for that row's (trueE, sample); valid is never refitted."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from transformer_ee.dataloader.load import get_train_valid_test_dataloader  # noqa: E402
from test_per_sample_weights import FHC, reference  # noqa: E402

SMOKE = ("/data/nova/chow/transformerEE/datasets/mprod6_1/training_v3/derived/"
         "dataset_transformer_ee_fd_fhc_nonswap_training_v3_split_v3features_smoke.csv")
R05 = ("/data/nova/chow/transformerEE/runs/pc-numu/fhc/pcnumu_fhc_r05_v3data-allsamples_v3features_20261029/"
       "model_6db92fccf53484d786f0c76dbc5a2ed9/input.json")


class LoaderWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        df = pl.read_csv(SMOKE, infer_schema_length=0).sample(n=6000, seed=20261110)
        path = Path(cls.tmp.name) / "slice.csv"
        df.write_csv(path)
        cfg = json.load(open(R05))
        cfg.update(data_path=str(path), num_workers=2, batch_size_train=97, batch_size_valid=31, batch_size_test=50,
                   weight={"name": "PerSampleFlatWeights", "kwargs": {"per_sample": FHC}})
        cls.cfg = json.loads(json.dumps(cfg))  # str keys, as from input.json
        torch.manual_seed(1)
        cls.train, cls.valid, _, _ = get_train_valid_test_dataloader(cls.cfg)
        tr = df.filter(pl.col("split") == "train")
        y, s = tr["trueE"].cast(pl.Float64).to_numpy(), tr["sample"].cast(pl.Int64).to_numpy()
        cls.ref = {}  # independent reference: per-sample fit on the train rows, applied by lookup
        for k, p in FHC.items():
            m = s == int(k)
            cls.ref[int(k)] = (y[m], reference(y[m], p["lo"], p["hi"], p["ease"], p["alpha"]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def expected(self, y32, s):
        """Reference weight for a train row, found by its float32 trueE (the batch tensor is float32)."""
        ys, ws = self.ref[s]
        hit = np.flatnonzero(ys.astype(np.float32) == np.float32(y32))
        self.assertTrue(hit.size, f"trueE {y32} not among sample {s} train rows")
        self.assertTrue(np.allclose(ws[hit], ws[hit[0]]), "ambiguous float32 match with different weights")
        return float(ws[hit[0]])

    def batches(self, loader, n):
        out = []
        for i, b in enumerate(loader):
            if i == n:
                break
            out.append(b)
        return out

    def sample_of(self, scalar):
        names = self.cfg["scalar"]
        cols = [names.index(f"sample{k}") for k in (1, 2, 3)]
        return scalar[:, cols].argmax(1).numpy() + 1  # normalised one-hot: the set flag is the only positive one

    def test_train_batches(self):
        n = 0
        for vec, sca, mask, tgt, w in self.batches(self.train, 6):
            self.assertEqual(tuple(w.shape), (tgt.shape[0], 1))
            for y, s, wi in zip(tgt[:, 0].numpy(), self.sample_of(sca), w[:, 0].numpy()):
                self.assertAlmostEqual(float(wi), self.expected(y, s), delta=1e-6 * max(1.0, float(wi)))
                n += 1
        self.assertGreater(n, 400)

    def test_valid_uses_train_fit(self):
        wt = self.train.dataset.weighter
        self.assertIs(self.valid.dataset.weighter, wt)
        n = 0
        for vec, sca, mask, tgt, w in self.batches(self.valid, 8):
            y = tgt[:, 0].numpy().astype(np.float64)
            s = self.sample_of(sca)
            np.testing.assert_allclose(w[:, 0].numpy(), wt.getweight(y, s), rtol=1e-6)
            n += len(y)
        self.assertGreater(n, 100)

    def test_repeat_reads_identical(self):
        a = [b[4] for b in self.batches(self.valid, 3)]
        b = [b[4] for b in self.batches(self.valid, 3)]
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
