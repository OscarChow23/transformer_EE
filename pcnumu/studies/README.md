# pcnumu/studies: one-off investigation scripts

One folder per investigation, named `YYYYMMDD_topic/`. A study answers one question once
(a preview, a scan, a metric investigation). Scripts that will be rerun on every new trial
belong in `pcnumu/eval/` instead, with tests.

- Write study scripts here, in git, from the start. Outputs go to
  `/data/nova/chow/pcnumu-analysis/studies/<same or similar name>/`, never into this repo.
- Each output folder records how it was made (`README.md` or `provenance.json`: script,
  commit, exact command).
- Folders with an `IMPORTED.md` were copied here on 2026-11-17 from where they were
  originally written; the originals stay next to their outputs.
