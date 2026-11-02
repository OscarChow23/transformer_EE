# pcnumu/: PC numu additions on top of upstream

Everything here was written for the PC numu recovery work and is not part of
`joshuabarrow221/transformer_EE`. It lives under one directory so upstream files
stay untouched and `wide_model_test` merges cleanly. Code changes to upstream
files are separate commits (`load.py`, `train.py`).

| directory | contents |
|---|---|
| `eval/` | `eval_by_sample.py` (per-sample test-set evaluation), `compare_runs.py` (two-run comparison), `test_conditional_maps.py`, `README_diagnostics.md` |
| `prep/` | `make_split_csv.py` (derived CSV with a hashed per-event `split` column), its sbatch, `merge_csv.sh` |
| `pretrain-checklist/` | the Claude Code skill (`SKILL.md`, `scripts/precheck.py`, `references/questions.md`) and the sbatch gate (`scripts/gate_hook.py`, `settings-hook.json`) |

## Imported unchanged from the old working tree

These files were copied byte for byte from `/data/nova/chow/transformerEE`
(frozen) and still contain its absolute paths. They have not been re-pointed at
this clone.

- `pretrain-checklist/scripts/precheck.py`: `BASE` and `--code-dir` point at the old tree.
  It also replicates the old launcher's `num_layers = nhead` fill (`kset(mkw,
  "num_layers", args.nhead)`), which this branch fixes (`args.num_layers`, upstream
  commit `144a149`). Re-verify before relying on it for a JSON without `model.kwargs.num_layers`.
- `pretrain-checklist/scripts/gate_hook.py`: `CHECKLIST_DIR` defaults to the old tree's
  `job_scripts/checklists`, which is read-only once frozen. Set `PRETRAIN_CHECKLIST_DIR`
  or change the default.
- `eval/eval_by_sample.py`: `--code-dir` defaults to the old tree's `code/`.
- `prep/*.sbatch`: paths into the old tree's `datasets/` and `configs/`.

## Installing the skill and gate

The files in `pretrain-checklist/` are copies. The active ones are read from
`~/.claude/skills/pretrain-checklist/`, and the hook from `~/.claude/settings.json`.
`settings-hook.json` is the `hooks` block from that file, kept as a record.
