---
name: pretrain-checklist
description: Hard pre-launch gate for TransformerEE trials on spacuda001. One GO per trial, signed against the trial spec and covering both horns. Resolves the trial spec into each run's effective config (loop.py prepare), runs the read-only precheck on every run (effective config as train_wide.py will build it, predicted model dir, data scan as the loader sees it, weights, resources), walks Oscar through a question checklist focused on what differs from the baseline, and writes the GO record that `loop.py submit` requires. Oscar submits; Claude never does. Use this whenever a TransformerEE trial, smoke test or re-run is about to be launched or "sent off", whenever a trial spec (pcnumu/trials/*/trial.yaml) has just been written or edited, or when Oscar asks to check, review or sign off a training config, even if he does not say "checklist".
---

# Pre-training checklist (hard gate, one GO per trial)

A run on the single RTX 8000 costs most of a day, and it is hard to interpret if any setting
was not what we believed. Past runs show the failure modes are rarely crashes:
- CLI flags silently ignored because the JSON wins;
- keys silently taking argparse defaults;
- names reused, so logs land in another run's directory;
- NaNs in vector strings poisoning normalization;
- flattening weights behaving differently than assumed.

The goal is that **Oscar and Claude hold the same, correct picture of what the trial will do**
before it is submitted, and that this agreement is written down next to the runs.

Terms follow the glossary (`transformeree-adaptation/CONTEXT.md` in the second brain):
- a **trial** (rNN) is one hypothesis;
- a **run** is one horn of a trial (rNN-fhc, rNN-rhc);
- the **trial spec** is the short file that gets signed;
- the **effective config** is what a run actually trains with.

## Where things live

| what | where |
|---|---|
| code (Git clone, branch `pcnumu-loop`) | `/data/nova/chow/transformerEE-pcnumu-loop` (`$CODE` below) |
| trial spec (in Git) | `$CODE/pcnumu/trials/<trial>/trial.yaml` |
| trial folder (outputs, never in Git) | `/data/nova/chow/pcnumu-runs/<trial>/` |
| run folder = `save_path` | `/data/nova/chow/pcnumu-runs/<trial>/<horn>/` |
| GO record | `/data/nova/chow/pcnumu-runs/<trial>/go-record.md` |
| r01–r05 (frozen, read-only) | `/data/nova/chow/transformerEE/` |

## The gate

Oscar submits with `python pcnumu/loop/loop.py submit <trial>`. It refuses unless all of these hold:
- `go-record.md` holds a `GO given by Oscar` line;
- the hashes in the record match the trial spec (both the Git copy and the run-folder copy) and
  every horn's `effective_config.json` and `train.sbatch`;
- the spec still resolves to exactly those files;
- the spec is committed;
- the checkout is clean (tracked files) and at the commit named in the record.

Each job repeats the same check as its first step (`loop.py job-start`) and writes
`provenance.json` (commit, dirty flag, hashes, job id). It stops before the GPU is touched if
anything changed while it was queued.

The PreToolUse hook (`scripts/gate_hook.py`, wired in `~/.claude/settings.json`) stops Claude from:
- running `loop.py submit` or `loop.py job-start`;
- running `sbatch`/`srun` of any script that runs `train_wide.py`;
- running `train_wide.py` directly.

**Never write the GO line before Oscar has said GO.** If Oscar says "just launch it", explain in
one line what is still open. The gate is his rule, so he can override it, but only in explicit
words, and the override goes in the record.

## Workflow

Run Python inside the env:

```bash
source /common/users/chow/miniforge3/etc/profile.d/conda.sh; conda activate transformer_ee; cd $CODE
```

### 1. The trial spec
Write or review `pcnumu/trials/<trial>/trial.yaml`. It contains:
- `trial` (equal to the folder name, `rNN_<short-description>`);
- `baseline`;
- a one-line `hypothesis`;
- `horns`;
- **only** the settings that differ, under `overrides` (dotted keys that must exist in the baseline);
- optionally `horn_overrides`, `add` (intentionally new keys) and `resources` (mem, time, cpus, partition).

`save_path` and `model_phys_name` are set by the loop. Commit the spec. GO is refused for an
uncommitted spec, so commit after every edit.

### 2. Resolve and prepare
```bash
python -I pcnumu/loop/loop.py resolve <trial>    # diff vs baseline per horn, predicted model dirs; writes nothing
python -I pcnumu/loop/loop.py prepare <trial>    # writes the trial folder (refuses once a GO record exists)
```
`resolve` lists every changed setting and flags changes to `data_path`, `split_column` or
`target`, because those change what is measured. The test set must stay the same for every
trial on a dataset version.

### 3. Run the precheck on every run
For each horn:
```bash
python -I -W ignore pcnumu/pretrain-checklist/scripts/precheck.py \
  --sbatch /data/nova/chow/pcnumu-runs/<trial>/<horn>/train.sbatch \
  --reference <baseline run's effective config or model_*/input.json> \
  --tmpdir <scratchpad> --json <scratchpad>/precheck_<horn>.json > <scratchpad>/precheck_<horn>.md
```
- It is read-only. Its only write is a temporary decompressed copy in `--tmpdir`, deleted at the end.
- A full scan of a 6–8 GB training_v3 file takes several minutes and about 14 GB of RAM, so run it
  in the background. Full mode needs a full scan; `--max-rows N` is only for a first look.
- `simulate()` mirrors `train_wide.py` main() at `VERIFIED_TRAIN_WIDE_SHA256`. If the code-state
  section WARNs that `train_wide.py` changed, re-verify `simulate()` before trusting it.
- Expected FAIL before submission: none. A dirty checkout is a FAIL.

Read the reports yourself first and propose a fix for every FAIL. Do not edit without Oscar's OK.
After any fix, commit, re-prepare and re-run the precheck.

### 4. State your understanding
Before asking anything, write a short "what this trial will do" summary (≤12 lines):
- the hypothesis;
- what changes vs the baseline (from `resolve`, not from the YAML as written);
- data and how many events of each sample it trains on;
- the weighting;
- the effective model and optimizer settings;
- where the outputs go;
- the expected time per run;
- how it will be evaluated (scorecard, compared with the baseline on the same test set).

This is the half of "both understand" that Oscar can correct.

### 5. Ask the checklist
Use `references/questions.md`. Ask with AskUserQuestion, at most 4 questions per call. Each
question carries its concrete value so that it stands on its own. Spend the time on:
- the settings that differ from the baseline;
- the WARNs;
- the choices (sample mix, weighting, features, comparison baseline).

For settings unchanged from a signed baseline, a one-line confirmation per section is enough.
Ask about both horns together, and name a horn only where they differ. "Not sure" means stop
and resolve. Skip a question only if it does not apply, and record "n/a: why".

### 6. Write the GO record (one per trial)
```bash
python -I pcnumu/loop/loop.py go-header <trial>   # exact hash lines + current commit
```
Write `/data/nova/chow/pcnumu-runs/<trial>/go-record.md`. It starts with those lines unchanged,
followed by:
```
## Precheck summary
per horn: FAIL/WARN counts; each WARN and how it was understood or waived (with reason)
## Trial summary (agreed)
<the step-4 summary, as corrected>
## Answers
<id> <question (short)>: <answer> [notes]
## Decision
GO given by Oscar at <time>   (or: NO-GO / overrides: ...)
```
Save each precheck report as `precheck_<horn>.md` in the trial folder. Then run
`loop.py check-go <trial>`: it must say "ready to submit".

### 7. Oscar submits
Give Oscar the command. Do not run it:
```bash
cd /data/nova/chow/transformerEE-pcnumu-loop && python -I pcnumu/loop/loop.py submit <trial>
```
After the jobs start, check each run's `provenance.json` (commit, `dirty: false`), compare the
real `model_*/` with the predicted model dir, and look at `train.log` for finite first-epoch
losses. Report any mismatch right away: it means our shared picture was wrong.

## Notes
- The checklist is for understanding, not ceremony.
- Do not invent values. If something cannot be checked on this host, say so and record "unverified".
- The laptop owns `training-register.md`. Remote evidence goes to the second-brain
  sub-project per its AGENTS.md.
