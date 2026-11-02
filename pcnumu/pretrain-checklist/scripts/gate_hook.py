#!/usr/bin/env python3
"""PreToolUse(Bash) gate for TransformerEE training launches by Claude.

Oscar submits training himself (`python pcnumu/loop/loop.py submit <trial>` in his own terminal,
which this hook never sees). So for a Claude session this hook denies:
  - `loop.py submit` and `loop.py job-start` (job-start belongs to the Slurm job; running it by
    hand writes provenance.json and blocks the real job);
  - `sbatch`/`srun` of any script that runs train_wide.py;
  - direct `python ... train_wide.py` (except inside a precheck command).
Everything else passes untouched, including sbatch of non-training scripts. Fails open (allows)
only on its own internal errors, and says so in a systemMessage.
"""
import json
import os
import re
import shlex
import sys

TRAIN_RE = re.compile(r"\bpython[0-9.]*\s+\S*train_wide\.py\b")
LOOP_RE = re.compile(r"\bloop\.py\s+(submit|job-start)\b")
VALUE_OPTS = {"-p", "-J", "-o", "-e", "-t", "-c", "-D", "-d", "--partition", "--job-name", "--output", "--error",
              "--time", "--dependency", "--chdir", "--mem", "--gres"}


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


def batch_scripts(cmd, cwd):
    """Paths of scripts passed to sbatch or srun anywhere in the command line."""
    out = []
    for seg in re.split(r"&&|\|\||;|\|", cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        for i, t in enumerate(toks):
            if os.path.basename(t) not in ("sbatch", "srun"):
                continue
            j = i + 1
            while j < len(toks) and toks[j].startswith("-"):
                if "=" not in toks[j] and toks[j] in VALUE_OPTS:
                    j += 1
                j += 1
            if j < len(toks):
                out.append(os.path.abspath(os.path.join(cwd, toks[j])))
    return out


def main():
    try:
        data = json.load(sys.stdin)
        cmd = data.get("tool_input", {}).get("command", "") or ""
        cwd = data.get("cwd") or os.getcwd()
        if not re.search(r"\b(sbatch|srun)\b|train_wide\.py|loop\.py", cmd):
            return
        m = LOOP_RE.search(cmd)
        if m:
            deny(f"`loop.py {m.group(1)}` is not for Claude: Oscar submits trials himself, and job-start runs "
                 "inside the Slurm job. Give Oscar the command instead.")
        if TRAIN_RE.search(cmd) and "precheck.py" not in cmd:
            deny("Direct train_wide.py launches are not allowed. Trials run through "
                 "`loop.py prepare` + the pretrain-checklist GO + Oscar's `loop.py submit`.")
        for script in batch_scripts(cmd, cwd):
            if os.path.isfile(script) and TRAIN_RE.search(re.sub(r"\\\s*\n", " ", open(script).read())):
                deny(f"{script} runs train_wide.py. Oscar submits training himself "
                     "(`python pcnumu/loop/loop.py submit <trial>`); nothing submits on his behalf.")
    except SystemExit:
        raise
    except Exception as e:  # never wedge the shell on a hook bug, but make it visible
        print(json.dumps({"systemMessage": f"pretrain gate hook error (allowed): {e!r}"}))


if __name__ == "__main__":
    main()
