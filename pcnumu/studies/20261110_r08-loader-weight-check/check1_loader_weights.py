"""r08 pre-GO check 1: the weighter the loader builds (from the loop-resolved effective config, on the loader's own
train population after drop_nan_rows) equals the agreed recipe, per sample; reports strength numbers vs the preview.
Reference parameters are typed here independently of the trial spec."""
import sys, numpy as np, polars as pl
CODE = "/data/nova/chow/transformerEE-pcnumu-loop"
sys.path[:0] = [CODE, CODE + "/pcnumu/loop", CODE + "/tests"]
import loop
from transformer_ee.dataloader.pl_dataset import drop_nan_rows
from transformer_ee.dataloader.pl_dataset import get_polars_df_from_file
from transformer_ee.utils.weights import create_weighter, FlatSpectraWeights
from test_per_sample_weights import reference
REF = {"fhc": {1: (1.0, 0.05), 2: (1.0, 0.02), 3: (1.0, 0.02)}, "rhc": {1: (0.9, 0.05), 2: (1.0, 0.02), 3: (1.0, 0.02)}}
cfgs = loop.resolve("r08_per-sample-flat")
for horn in ("fhc", "rhc"):
    cfg = cfgs[horn]
    df = get_polars_df_from_file(cfg["data_path"])
    tr = drop_nan_rows(df.filter(pl.col(cfg["split_column"]) == "train"))
    y, s = tr[cfg["target"][0]].to_numpy().astype(np.float64), tr["sample"].to_numpy()
    print(f"== {horn.upper()}: {df.height} rows in file, {tr.height} train rows after drop_nan_rows")
    del df
    w = create_weighter(cfg, y, sample=s).getweight(y, s)
    w05 = FlatSpectraWeights(y, bins=500, alpha=0.05).getweight(y)
    for k, (lo, a) in REF[horn].items():
        m = s == k
        r = reference(y[m], lo, 5.0, 3.0, a)
        wk, yk = w[m], y[m]
        ess = wk.sum() ** 2 / (wk ** 2).sum() / wk.size
        print(f"  S{k}: rows {m.sum()}, max|Δw| vs recipe {np.abs(wk - r).max():.2e}, mean {wk.mean():.6f}, "
              f"max ÷ 2 GeV {wk.max() / wk[(yk > 1.8) & (yk < 2.2)].mean():.1f}, ESS {ess:.1%}; "
              f"weight share of all train: r08 {wk.sum() / w.sum():.1%} vs r05 {w05[m].sum() / w05.sum():.1%} (rows {m.mean():.1%})")
    print(f"  combined ESS r08 {w.sum()**2/(w**2).sum()/w.size:.1%} (r05 {w05.sum()**2/(w05**2).sum()/w05.size:.1%}); "
          f"mean event weight r08 {w.mean():.3f} vs r05 {w05.mean():.3f}")
