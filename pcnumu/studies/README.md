# pcnumu/studies: one-off investigation scripts

One folder per investigation, `YYYYMMDD_topic/`, with the scripts that answered one question once (a
preview, a scan, a metric investigation). They are the record of what was run and are not maintained;
anything that will be rerun on future trials belongs in `pcnumu/eval/` with its logic in `pcnumu/core/`.

| Folder | Question | Outputs |
|---|---|---|
| `20261030_r05-diagnostics` | r05 conditional diagnostics and the GPVM reconciliation (README only) | frozen `transformerEE/runs/pc-numu/<horn>/…r05…/diagnostics_trueE_20261030_v2/` |
| `20261102_ellipse-calibration` | calibrating the frozen 22% ellipse on r05 | `pcnumu-runs/r05_…/ellipse_calibration_training_v3/` |
| `20261103_flattening-scans` | FlatSpectraWeights alpha/bins, per-sample and taper scans before r06/r08 | `pcnumu-runs/r06_…/flattening_scan/`, `flattening_preview/` |
| `20261103_r05-edge-and-tail` | r05 migration at the window edge and the high-energy tail | `pcnumu-runs/r06_…/flattening_scan/edge_and_tail/` |
| `20261104_r07-s1-subsample` | previewing r07 (half of S1) and its alpha choice | `pcnumu-runs/r07_…/preview/`, `flattening_scan/` |
| `20261110_feature-importance` | input correlations and permutation importance (r06 FHC) | `pcnumu-analysis/studies/20261109_feature-correlation-fhc/`, `20261110_feature-importance/` |
| `20261110_r08-flattening-preview` | r08 per-sample flattening shape options | `pcnumu-analysis/studies/20261110_r08-flattening-preview/` |
| `20261110_r08-loader-weight-check` | checking the r08 loader's weights against the preview | `pcnumu-runs/r08_…/check1_loader_weights.txt` |
| `20261111_peak-pull-metric` | what the peak-pull metric measures, and v2 proposals | `pcnumu-analysis/studies/20261111_peak-pull-metric/` |

New studies write their outputs to `/data/nova/chow/pcnumu-analysis/studies/` under the same folder name.
Each output folder records the script, commit and command that made it. Folders with an `IMPORTED.md` were
copied here on 2026-11-17 from where they were first written; the originals stay next to their outputs.
