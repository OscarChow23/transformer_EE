# pcnumu/studies: one-off investigation scripts

One folder per investigation, `YYYYMMDD_topic/`, with the scripts that answered one question once (a
preview, a scan, a metric investigation). They are the record of what was run and are not maintained;
anything that will be rerun on future trials belongs in `pcnumu/eval/` with its logic in `pcnumu/core/`.

| Folder | Question |
|---|---|
| `20261030_r05-diagnostics` | r05 conditional diagnostics and the GPVM reconciliation (README only) |
| `20261102_ellipse-calibration` | calibrating the frozen 22% ellipse on r05 |
| `20261103_flattening-scans` | FlatSpectraWeights alpha/bins, per-sample and taper scans before r06/r08 |
| `20261103_r05-edge-and-tail` | r05 migration at the window edge and the high-energy tail |
| `20261104_r07-s1-subsample` | previewing r07 (half of S1) and its alpha choice |
| `20261110_feature-importance` | input correlations and permutation importance (r06 FHC) |
| `20261110_r08-flattening-preview` | r08 per-sample flattening shape options |
| `20261110_r08-loader-weight-check` | checking the r08 loader's weights against the preview |
| `20261111_peak-pull-metric` | what the peak-pull metric measures, and v2 proposals |

Outputs: `/data/nova/chow/pcnumu-analysis/studies/` (or the trial folder they were made for). Each
output folder records the script, commit and command that made it. Folders with an `IMPORTED.md` were
copied here on 2026-11-17 from where they were first written; the originals stay next to their outputs.
