# pcnumu/eval — evaluation commands

Thin command-line programs over the shared library `pcnumu/core/`. Each reads saved predictions
(nothing is re-inferred), checks the alignment with the dataset's test rows, and writes to a fresh
output folder.

| Command | When | Writes |
|---|---|---|
| `scorecard.py RUN_DIR` | after every training run | `RUN_DIR/scorecard/`: `scorecard.json` (+ `.sha256`, `.md`), `peak_metrics.json`, loss curves, error maps, ellipse planes |
| `scorecard.py RUN_DIR --peak-only` | to refresh peak metrics of an existing scorecard | `RUN_DIR/scorecard/peak_metrics.json` |
| `eval_by_sample.py --config CFG` | per-sample diagnostics of one run | summary tables and plots per sample, incl. above 6 GeV |
| `eval_by_sample.py --config CFG --flattening-only DIR` | at the start of every training job (job template) | training spectrum before/after the flattening weights |
| `compare.py runs --run ... --out-dir DIR` | quick like-for-like comparison of any runs | summary table, error distributions, bias/resolution vs energy |
| `compare.py regional --horn H --sample S --out-dir DIR` | paired regional comparison of r05/r06/r08/r09; S = 1, 2, 3 or 23 (S2+S3 pooled) | `DIR/<horn>/s<sample>/summary.json`, counts, checks |
| `compare.py render DIR/fhc/sS [DIR/rhc/sS]` | after `regional` | five bar plots and `comparison.md` from `summary.json` |
| `leaderboard.py [--md PATH]` | after each new scorecard or regional comparison | `pcnumu-runs/leaderboard_<horn>.csv` and the model-comparison note (per-sample regional tables from `--regional`) |
| `figindex.py add/index` | when a figure is worth keeping | figure records in `/data/nova/chow/pcnumu-analysis/figures/` |

Every command prints its full usage with `--help`. Definitions are in `pcnumu/core/metrics.py`
and `pcnumu/core/regional.py`.
