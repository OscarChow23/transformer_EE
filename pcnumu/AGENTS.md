# Rules for agents working in pcnumu/

Read `pcnumu/README.md` first. These rules apply to every agent (Claude, Codex, others) writing
code, plots or results for this project.

## Where things go

- **Code is in git, outputs never are.** Write every script that produces something kept inside this
  repository. Outputs go to `/data/nova/chow/pcnumu-analysis/` (studies, figures, talks, plans) or a
  trial's folder in `/data/nova/chow/pcnumu-runs/`. Never write a script into an output folder.
- **Reusable tool or one-off?** If it will be rerun on future trials, it belongs in `eval/` (a command)
  with its logic in `core/` and a test in `tests/`. If it answers one question once, it goes in
  `studies/YYYYMMDD_topic/`. Talk figures go in `plots/talks/YYYYMMDD_name/`.
- **Never re-implement a metric.** Import from `core/metrics.py` / `core/regional.py`. Load predictions
  with `core/data.load_test_predictions` (positional alignment, verified row by row). If a definition
  is missing, add it to `core/` with a test, not to your script.
- **Every output folder records how it was made:** script path, git commit and dirty flag, exact
  command, input hashes and test-set fingerprint. Refuse to overwrite an existing output folder.
- The training library (`transformer_ee/`, `train_wide.py`) must never import from `pcnumu/`.

## Plots: one house style

Every figure (diagnostics, studies, talks, thesis) uses `pcnumu/core/style.py`:

```python
from pcnumu.core import style
style.use()                                  # once, before making figures
fig, ax = style.figure()
ax.plot(x, y, **style.line("3"), label=style.SAMPLE_STYLE["3"]["label"])
style.labels(ax, style.E_TRUE, "Resolution [%]")
style.tag(ax, "FHC · S3 · test set")
style.save(fig, out_dir, "name", caption="...")
```

- Do not set matplotlib rcParams, fonts or colours locally. Use the fixed vocabulary:
  `style.SAMPLE_STYLE` (S1/S2/S3/all), `style.SERIES` (runs or models in order), `style.BARS`
  (grouped/stacked bars), `style.E_TRUE` / `E_RECO` / `FRAC` for axis titles. If something is
  missing, add it to `style.py` and say so.
- No figure titles or text boxes on the plot. Axis titles with units, a legend, at most one short
  in-frame tag (horn, sample, population). Explanations go in the caption (`style.save(..., caption=)`).
- Captions say what is plotted (x, y, normalisation, binning), which events (horn, sample, split, N),
  which models (run IDs), how uncertainties were computed, and the one claim the plot supports.
  Statistical error bars on test events do not include training-seed variation: say so.
- Label what a histogram is: raw counts, weighted training contribution, or unit-normalised shape.

## Results

- Numbers come from a model's saved scorecard `model_dir`, never from globbing run folders or from
  smoke runs. Compare models only on the same test events (equal test-set fingerprints) within a horn.
- Training is launched only by Oscar through `pcnumu/loop/loop.py submit` after a signed GO record.
