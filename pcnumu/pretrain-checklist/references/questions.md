# Question bank

Ask in this order, with ≤4 questions per AskUserQuestion call. **S** marks a question that
is also asked in smoke mode. Fill `{…}` from the precheck report and facts JSON. Option
labels are suggestions: keep 2–4 options, and put the honest "something is off" option in
every question. The tool adds "Other" for free text automatically.

A ✱ marks a **pick-the-right-description** question. Build three plausible descriptions
with only one correct (from the precheck), and do not mark which. If Oscar picks a wrong
one, that is the most useful outcome of the checklist: stop and talk it through.

---

## A. Purpose and comparison

**A1 S. Purpose.** "What question does this run answer?" Options are drawn from the
context, e.g. "Effect of training_v3 data at fixed features vs {ref}", "Effect of new
features at fixed data", "Hyperparameter variant ({param})", "Smoke test of pipeline
only". Record the answer verbatim. Every later question is judged against it.

**A2 ✱. What changes vs the reference.** Show the diff count. The options are three
summaries of the diff vs `{reference}`, only one correct. For example, (a) "data file +
save names only; features, model, weights identical", (b) "data file + dropout 0.3",
(c) "data file + 16 new vector features". Why: if more than one thing changes, the
comparison cannot attribute the effect.

**A3. Decision rule.** "Which result would change what we do next?" Free text via Other,
or options like "Adopt if held-out resolution in 0.5–6 GeV improves for every
category", "Exploratory: no decision rule". Fixing this before seeing results avoids
post-hoc reading.

## B. Data and split

**B1 S. Data file.** Show `{data_path}`, size on disk / uncompressed, rows, columns,
producing script + seed + manifest (if known). "Is this the intended file and version?"

**B2. Category mix and spectrum.** Show sample counts/fractions and `trueE` quantiles
(and fraction outside 0.5–6 GeV if relevant). "Is this the mix you intend the model to
learn from?" Options: "Yes, intended", "Mix needs changing", "Not sure what the mix
should be". Why: the loss averages over events, so the dominant category dominates
the fit.

**B3. Dropped rows.** Show loader-dropped rows (null/NaN in any of the file's columns),
rows dropped upstream (manifest), and non-finite feature values. "Acceptable?"

**B4 ✱. Where the evaluation numbers come from.** Options include (a) "the loader's
internal test split ({n_test} rows, random rows, no IDs)", (b) "a separate held-out
file {path} with IDs, evaluated by inference afterwards", (c) "the validation split".
Also show the `--overlap-with` result if a reference model is evaluated on the same set.

## C. Features

**C1 S. Feature set.** "{n_vector} vector + {n_scalar} scalar; vs reference: +{added}
−{removed}." "Confirm this is the list?"

**C2. Sentinels and validity.** Show the 0 / −5 / −1 fractions for the affected features
and how they are handled (kept in-band, explicit validity flag, transformed). "Agree
with this handling?" Why: mean/std normalization includes sentinels, so a feature with
36% zeros has its scale set partly by "invalid".

**C3. Problem features.** List constant, extreme-tail and non-finite features from the
precheck. For each: keep / clip / drop. Constant features (e.g. RHC `coarseTiming`,
`lowGain`) are harmless but carry no information. Keeping them can be right for
comparability.

**C4. Inputs are reco-only.** Show that the precheck leakage check passed, plus the
decision on reco-energy scalars (`numuRecoE` etc.). "Confirm there is no truth or
bookkeeping info in the inputs, and agree with the reco-energy decision?"

**C5. Prong cap.** "max_num_prongs={n}; {k} events exceed it; truncation keeps the
first prongs in CSV order. OK?"

## D. Targets, loss, weights

**D1. Targets and loss.** "{targets}, loss {names} with coefficients {c}." Point out
anything special. For example, MAPE weights low-energy events more and blows up as
targets → 0 (min target {min}).

**D2 ✱. What the weighting does.** Show bins, range, alpha, empty-bin fraction, per-event
mean weight, max/min weight ratio and effective sample size. Options are three
descriptions, one correct, e.g. (a) "Flattens trueE between {lo} and {hi}; {x}% of bins
empty; average event weight {m}, so the effective learning rate is about {m}× the
unweighted one", (b) "Flattens only within 0.5–6 GeV", (c) "Weights are mean 1 over
events". Then: "Keep, or change range/bins/alpha/maxweight?" Offer to plot the
weighted spectrum per category.

## E. Model and optimization

**E1. Architecture (effective).** nhead, num_layers, d_model, dim_feedforward,
scalar_as_token, use_new_head, auto_d_model, plus which of them were set by
`--profile`/`--d-model`. "Matches what you expect?"

**E2. Training settings.** dropout, optimizer, lr, weight decay, batch sizes, epochs,
early stopping (window, min delta). For each, say whether it equals the reference.

**E3. Noise/augmentation.** none / coherent / particle. State what the checked-out code
supports: local code is coherent-only; particle needs upstream `c3270327`.

**E4. Reproducibility.** "There is no torch seeding, so init and batch order differ
per run. Differences smaller than run-to-run spread are not meaningful. Single run OK,
or repeat N times / add seeding first?"

## F. Effective config and bookkeeping

**F1 S. Single source of truth.** Show the provenance lines: any value from CLI only,
CLI ignored, argparse default, or the num_layers bug. "Confirm every intended value is in
the JSON and nothing comes from defaults?" (A FAIL here must be fixed, not waived.)

**F2 S. Names and paths.** save_path, model_phys_name, job-name, `--output`, W&B
project/id/mode, predicted `model_<md5>` dir. All must be new and follow the naming
convention. "OK?"

**F3. Code version.** Branch @ commit, train_wide.py hash status, known relevant
differences vs upstream. "Run on this code?"

## G. Resources and failure plan

**G1 S. Slurm request.** partition, time, mem (and mem/data ratio), cpus vs num_workers,
GPU. Also mention smoke-test status for full mode and what else is queued
(`squeue -u $USER`), since there is only one GPU. "OK to request this?"

**G2. If it fails or times out.** There is no resume logic. The same config reuses the
same model dir; `last_model.zip` and rolling epoch checkpoints exist. "Plan if it
dies: rerun from scratch with a new save_path / accept partial?"

## H. Evaluation and recording

**H1. Evaluation plan.** Which set, which metrics (bias, resolution, tails for each
target; per category; 0.5–6 GeV window; |rel. err.| < 5/10/15%), and which baseline
models are evaluated on the same set. "Agree?"

**H2. Recording.** Where the run gets registered (runs.md / findings.md via the
permitted channel; training-register.md owned by the laptop) and what evidence is kept.
"OK?"

## Final

**GO S.** Restate the job in one line (job name, data, change vs reference, expected
time). "GO to submit exactly this sbatch (sha256 {short}) now?" Options: "GO",
"Not yet".
