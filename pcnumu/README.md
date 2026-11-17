# pcnumu/ — PC numu energy estimation on top of TransformerEE

Code for the NOvA peripheral/contained numu (PC numu) energy-estimation study: data preparation,
a gated experiment loop for training trials, evaluation of trained models, and the scripts behind
studies and talk figures. It sits on the TransformerEE library (`transformer_ee/`, `train_wide.py`,
upstream `joshuabarrow221/transformer_EE`) and keeps everything study-specific in this one folder.

**Dependency rule:** `pcnumu/` imports from `transformer_ee/`; the library never imports from
`pcnumu/`. Library changes are separate commits so they can be proposed upstream on their own.

## Layout

| Folder | What it is | Rerun? |
|---|---|---|
| `core/` | **Shared library**: loading and aligning test predictions, the metric definitions, the regional comparison, the house plot style, shared plots, writers. See [`core/__init__.py`](core/__init__.py). | imported |
| `eval/` | **Commands** run on every trained model: `scorecard.py`, `eval_by_sample.py`, `compare.py` (runs / regional / render), `leaderboard.py`, `figindex.py`. See [`eval/README.md`](eval/README.md). | yes |
| `tests/` | Tests for `core/` and the commands. | yes |
| `prep/` | Derived dataset with a hashed per-event `split` column (`make_split_csv.py`) and its batch job. | per dataset |
| `loop/` | Experiment loop: trial spec → effective config → GO record → batch submission (`loop.py`, job template). | per trial |
| `trials/` | One `trial.yaml` per trial (`rNN_<description>`): hypothesis, baseline, settings, status. | — |
| `pretrain-checklist/` | The pre-launch gate: checklist skill, `precheck.py`, and the hook that blocks unsigned submissions. | per trial |
| `studies/YYYYMMDD_topic/` | One-off investigation scripts, kept as the record of what was run. Not maintained. | no |
| `plots/talks/YYYYMMDD_name/` | Scripts that made one talk's figures (record). | no |

Run outputs never go in this repository. Trials are written to `/data/nova/chow/pcnumu-runs/`
(one folder per trial, `fhc/` and `rhc/` inside); everything made *about* runs (studies, figures,
talks, plans) to `/data/nova/chow/pcnumu-analysis/`. The map is `/data/nova/chow/README.md`.

## Running

Python: `/data/nova/chow/miniforge3/envs/transformer_ee/bin/python -I` (isolated mode; every command
adds the repository root to its import path itself).

```bash
python -I pcnumu/eval/scorecard.py RUN_DIR                       # after training: scorecard + peak metrics
python -I pcnumu/eval/leaderboard.py --md model-comparison.md    # table of every scored trial
python -I pcnumu/eval/compare.py regional --horn fhc --sample 3 --out-dir DIR
python -I pcnumu/eval/compare.py render DIR/fhc/s3
python -I pcnumu/tests/test_metrics.py                           # tests: each file runs standalone
```

## Conventions

- **Metrics** are defined once, in `core/metrics.py` (and `core/regional.py` for the regional
  comparison). Commands and studies import them; they never re-implement a definition.
- **Plots** use the house style, `core/style.py`, and its fixed colours. See `AGENTS.md`.
- **Provenance:** every output records the code commit and dirty flag, file hashes and the test-set
  fingerprint. Outputs refuse to overwrite an existing folder.
- **Training** is launched only through `loop/loop.py submit` with a signed GO record.

## Installing the skill and gate

The files in `pretrain-checklist/` are copies. The active ones are read from
`~/.claude/skills/pretrain-checklist/`, and the hook from `~/.claude/settings.json`.
`settings-hook.json` is the `hooks` block from that file, kept as a record.
