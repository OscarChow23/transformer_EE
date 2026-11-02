---
name: pretrain-checklist
description: Hard pre-launch gate for TransformerEE training on spacuda001. Runs a read-only precheck on the sbatch + JSON (effective config as train_wide.py will build it, predicted model dir, data scan as the loader sees it, weights, resources), then walks Oscar through a question checklist so both sides understand every configuration choice, and writes a signed record before anything is submitted. Use this whenever a TransformerEE / train_wide.py training, smoke test, or re-run is about to be launched or "sent off", whenever an sbatch or training config JSON has just been written or edited and the next step would be submitting it, or when Oscar asks to check, review, or sign off a training config, even if he does not say "checklist".
---

# Pre-training checklist (hard gate)

A training run on the single RTX 8000 costs most of a day and is hard to interpret if
any setting was not what we believed. Past runs show the failure modes are rarely crashes:
CLI flags silently ignored because the JSON wins, keys silently taking argparse defaults,
names reused so logs land in another run's directory, NaNs in vector strings poisoning
normalization, flattening weights behaving differently than assumed. The goal here is
that **Oscar and Claude hold the same, correct picture of what the job will do** before it
is submitted, and that this agreement is written down next to the job.

## The gate

Do not run `sbatch`, `srun`, or `python train_wide.py` for any training job (including
smoke tests) until all of these hold:

1. The precheck has been run on the exact sbatch + JSON being submitted, with no
   unresolved FAIL. A FAIL is resolved by fixing and re-running, or by Oscar explicitly
   waiving it with a reason that goes in the record.
2. Every checklist question for the mode (full or smoke) has an answer from Oscar.
   "Not sure" or "change needed" means stop, resolve, and re-ask; it is not an answer.
3. Oscar has given an explicit final GO for this specific record.
4. The sha256 of the sbatch and JSON still match the ones in the record. Any edit after
   sign-off invalidates the record. Re-run the precheck and re-ask the affected questions.

If Oscar says "just launch it", explain in one line what is still open and offer the
smoke-mode checklist. The gate is his own rule, so he can override it, but only in
explicit words, and that override goes in the record.

A PreToolUse hook (`scripts/gate_hook.py`, wired in `~/.claude/settings.json`) enforces
this mechanically. It denies:
- `sbatch` of any script that runs `train_wide.py`, unless a record in `job_scripts/checklists/`
  contains the lines `sbatch: <abs path>  sha256 <hex>` and `config: <abs path>  sha256 <hex>`
  with both hashes current, plus a line starting `GO given by Oscar`;
- direct `python train_wide.py`.

So write those lines exactly as in the template below. Never write the GO line before
Oscar has said GO.

## Workflow

### 1. Identify the job
Find the sbatch and the JSON it passes as `--base-config`, plus the reference run it should
be compared with (usually the previous run's `model_*/input.json`, which is the effective
config of that run). Ask if any of these is ambiguous. Choose the mode:
- **smoke**: `test` partition, ≤2 h, few epochs or a subset file. Only questions marked S.
- **full**: everything else.

### 2. Run the precheck
```bash
source /common/users/chow/miniforge3/etc/profile.d/conda.sh; conda activate transformer_ee
python -I -W ignore ~/.claude/skills/pretrain-checklist/scripts/precheck.py \
  --sbatch JOB.sbatch --config CONFIG.json --reference REF_input.json \
  --tmpdir <scratchpad> --json <scratchpad>/precheck.json > <scratchpad>/precheck.md
```
- It is read-only. The only thing it writes is a temporary decompressed copy in `--tmpdir`, deleted at the end.
- A full scan of a 7.6 GB uncompressed v3 file takes ~3 min, ~14 GB RAM and 7.6 GB of
  `--tmpdir` space, so run it in the background.
  `--max-rows N` is fine for a first look, but the record needs a full scan for full mode,
  because rare problems (e.g. 17 `nan` prongs in 2.8 M rows) only show up there.
- Add `--overlap-with OTHER.csv.xz` when the reference model was trained on a different
  file, to check whether its training events sit in our evaluation set.
- What it checks:
  - It rebuilds the effective config through `train_wide.py`'s own helpers and predicts
    `save_path/model_<md5>`. This was verified to reproduce an existing run's model dir.
    If `train_wide.py` changed (code-state WARN), re-verify `simulate()` against the
    launcher's `main()` before trusting it.
  - For every value it reports where it comes from: JSON, CLI, argparse default, or
    launcher bug.
  - It emulates the loader's `drop_nan_rows`, then checks per-feature non-finite,
    constant, tail and sentinel values, nprong vs `max_num_prongs`, target ranges vs loss,
    the FlatSpectraWeights histogram, split sizes, sample mix, name reuse and Slurm
    resources.

Read the report yourself first. For every FAIL, propose the concrete fix. Do not edit
GPU workspace files without Oscar's OK, then re-run.

### 3. State your understanding
Before asking anything, write Oscar a short "what this job will do" summary in plain
terms (≤12 lines):
- data and how many events of each category it trains on
- what changes vs the reference
- effective model/optimizer settings
- the weighting
- where outputs go
- how long it should take
- how it will be evaluated

Every value comes from the precheck, not from the JSON as written. This is the half of
"both understand" that Oscar can correct.

### 4. Ask the checklist
Use `references/questions.md`. Ask with AskUserQuestion, at most 4 questions per call,
grouped by section, in order. Each question carries the concrete value it is about (e.g.
"Flattening: 500 bins over 0.43–68 GeV, 61% empty, mean event weight 0.33. Keep?") so that
it stands on its own. A few questions are deliberately not yes/no: they ask Oscar to pick
which description is correct. This catches a mismatch that a "Confirmed" click would hide.
Skip a question only if it truly does not apply, and note "n/a: why" in the record.

When an answer changes the config, make the edit (with OK), re-run the precheck, and
re-ask only the questions whose values changed.

### 5. Write the record
Write `/data/nova/chow/transformerEE/job_scripts/checklists/<sbatch-stem>_<YYYYMMDD-HHMM>.md`:
```
# Pre-training checklist: <sbatch-stem>   (mode: full|smoke)
date, operator: Oscar + Claude session
sbatch: <path>  sha256 <...>
config: <path>  sha256 <...>
reference: <path>
code: <branch @ commit>, train_wide.py sha256 <...>
predicted model dir: <...>
## Precheck summary
FAIL/WARN counts; each FAIL and how it was resolved or waived (with reason)
## Run summary (agreed)
<the step-3 summary, as corrected>
## Answers
<id> <question (short)>: <answer> [notes]
## Decision
GO given by Oscar at <time>   (or: NO-GO / overrides: ...)
## Launch
job id, submit time, post-launch check result
```
Save the precheck markdown alongside it as `<same-name>.precheck.md`.

### 6. Launch and verify
Only after GO:
- re-hash the sbatch and JSON against the record
- `mkdir -p` the `--output` directory
- submit

Append the job id to the record. Once the job has started, compare the real
`model_*/input.json` with the predicted model dir and effective config. Also check
`train.log` for finite first-epoch losses. Report any mismatch to Oscar right away; it
means our shared picture was wrong.

## Notes
- The checklist is for understanding, not ceremony. If a question is answered by a
  PASS line that Oscar has seen, a one-line confirmation is enough. Spend the time on
  the WARNs and on the choices (sample mix, weighting, features, comparison baseline).
- Do not invent values. If something cannot be checked on this host (e.g. second-brain
  notes, upstream repo state), say so in the question and record "unverified".
- The laptop owns `training-register.md`. Remote evidence goes to the second-brain
  sub-project per its AGENTS.md. The checklist record lives on the GPU host next to the
  job scripts.
