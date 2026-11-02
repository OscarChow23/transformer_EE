# R05 quick neutrino-energy diagnostics — 30 October 2026

Use the existing `eval_by_sample.py`; no ROOT, inference, export or training is required.
Dependencies: numpy, polars, matplotlib, installed in
`/common/users/chow/miniforge3/envs/transformer_ee/bin/python`.

```bash
cd /data/nova/chow/transformerEE
PYTHON=/common/users/chow/miniforge3/envs/transformer_ee/bin/python
for horn in fhc rhc; do
  run="/data/nova/chow/transformerEE/runs/pc-numu/$horn/pcnumu_${horn}_r05_v3data-allsamples_v3features_20261029"
  case "$horn" in
    fhc) model="$run/model_6db92fccf53484d786f0c76dbc5a2ed9" ;;
    rhc) model="$run/model_6d64aa3fca02cf0641e52337a4d80ca9" ;;
  esac
  "$PYTHON" eval/eval_by_sample.py --config "$model/input.json" --model-dir "$model" \
    --conditional-maps-only --out-dir "$run/diagnostics_trueE_NEW_UNIQUE_NAME"
done
```

Existing output directories are refused. The reviewed outputs are
`<run>/diagnostics_trueE_20261030_v2/`; earlier first-pass directories are preserved.
Each horn has six PNGs, six PDFs, six count NPZs, slice_statistics.csv and provenance.json.
Samples 1/2/3 = standard contained / peripheral muon contained / peripheral muon uncontained.
All rows in the derived CSV were checked: flags are exclusive and agree with sample.
Predictions align row by row on both trueE and trueLepE to the file's test rows.
The loader preserves order and uses shuffle=False for test. eval() loads best_model.zip.

The split is a deterministic event hash of run/subrun/cycle/evt with seed 20261029,
thresholds 0.20 and 0.24: nominal test/valid/train 20/4/76%, NOT a second 4% of the
remaining 80%. Actual counts are in the derived dataset manifest. Config seed=0 is
not the event-hash split seed. Both saved configs have dropout=.3, FlatSpectraWeights
alpha=.05/bins=500, 50 vector and 9 scalar inputs. This is R05, not Oct22 R04.
FHC job110 completed Oct29, best epoch21/loss .02661387436, last24.
RHC job111 completed Oct30, best epoch15/loss .03093204461, last19.

## Definitions and display

Unweighted held-out diagnostic events. x bins: 0–10 GeV, width .25 GeV. Energy y:
0–10 GeV, width .125 GeV. Relative error r=(pred-true)/true, y -1 to 1, width .02.
Colours = count in (x,y) / ALL finite positive-truth events in that x slice, including
vertical overflow in the denominator. Thus visible slice sums may be below one.
No log axes. Empty cells masked. Last bin includes final edge, matching numpy histogram.
Raw 2D counts, full slice counts, probabilities, edges, quantiles and excluded/overflow
counts are saved in NPZ; per-slice counts/overflow/quantiles are also in CSV.
Quantiles q16/median/q84 use all raw y values (no y clipping), numpy default linear
interpolation, require >=50 events per slice. q84 above display is clipped visually,
not excluded numerically. Bias=median(r); central spread=(q84-q16)/2, not RMS or
Gaussian sigma. Existing eval_by_sample summaries use trueE 0.5–6 GeV and all energy.

## GPVM reconciliation

GPVM authoritative older-dropout comparison path:
`/exp/nova/app/users/ochow/transformer_EE-wide/graph_eval/`.
Its eval_model.C SHA256 a173c1bbb2dc80797f9138b291bc03cde7d7e93da92c5d02a78dd88fa98e0ee5
has newer raw-count 2D/quantile plots but no category handling. Local legacy macro
SHA256 d6c9d1366df0d3a628effebd94f46705306ac8b9bb0275b5f5cbdd84bfafe222 differs.
The GPU R05 category plot source is eval/eval_by_sample.py, identified by the Oct29
handoff and actual completed job outputs. Extend that source rather than replacing
the divergent ROOT macro. Mathematical residual sign and q16/q84 conventions agree;
this is not byte-for-byte ROOT reproduction: conditional normalization, bins,
minimum counts and numpy interpolation are explicitly specified here.
ROOT itself and GPVM-only files are unnecessary for these quick plots. Full legacy
ROOT parity would need the current GPVM eval_model.C, run_eval_all.sh and
export_root_canvases.C; compare_dropout.py is the Oct22 dropout comparison, not R05.

## Findings to discuss

For trueE .5–6 GeV, median neutrino relative error / central spread:
FHC S1 -2.00%/6.75%, S2 -1.17%/10.13%, S3 -.69%/15.18%.
RHC S1 -1.09%/6.12%, S2 -1.10%/9.20%, S3 +.98%/17.24%.
S3 full-energy central spread is 18.03% FHC and 20.30% RHC. The window excludes
28,877 of 125,420 FHC S3 events and 42,220 of 175,669 RHC S3 events.
Both S3 curves show positive low-energy bias and a widening central band at higher
energy. FHC at 1.75–2 GeV has median +.5%, spread13.5%; at 8–8.25 median -6.3%,
spread32.8%. RHC at 2–2.25 median+.8%, spread15.2%; at 8–8.25 median-3.1%, spread29.9%.
These observations do not establish peak learning or a cause; compare future models
on the same held-out events and avoid drawing performance conclusions from global
sample-weighted summaries alone. No new training or grid jobs were launched.
