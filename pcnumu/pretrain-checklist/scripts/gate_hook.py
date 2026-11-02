#!/usr/bin/env python3
"""PreToolUse(Bash) gate for TransformerEE training launches.

Blocks `sbatch X.sbatch` when X runs train_wide.py, unless a pretrain-checklist record in
CHECKLIST_DIR names that sbatch with its current sha256, its config with the config's
current sha256, and contains a GO line. Blocks direct `srun`/`python train_wide.py`.
Everything else passes untouched. Fails open (allows) only on its own internal errors,
and says so in a systemMessage.
"""
import glob
import hashlib
import json
import os
import re
import shlex
import sys

CHECKLIST_DIR = os.environ.get("PRETRAIN_CHECKLIST_DIR", "/data/nova/chow/transformerEE/job_scripts/checklists")
TRAIN_RE = re.compile(r"\bpython[0-9.]*\s+\S*train_wide\.py\b")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


def sbatch_scripts(cmd, cwd):
    """Paths of batch scripts passed to sbatch anywhere in the command line."""
    out = []
    for seg in re.split(r"&&|\|\||;|\|", cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        for i, t in enumerate(toks):
            if os.path.basename(t) != "sbatch":
                continue
            j = i + 1
            while j < len(toks) and toks[j].startswith("-"):
                # options with a separate value (e.g. -p normalrun); --opt=val needs no skip
                if "=" not in toks[j] and toks[j] in ("-p", "-J", "-o", "-e", "-t", "-c", "-D", "--partition",
                                                     "--job-name", "--output", "--error", "--time",
                                                     "--dependency", "-d", "--chdir", "--mem", "--gres"):
                    j += 1
                j += 1
            if j < len(toks):
                out.append(os.path.abspath(os.path.join(cwd, toks[j])))
    return out


def check_record(script):
    s_sha = sha(script)
    for rec in sorted(glob.glob(os.path.join(CHECKLIST_DIR, "*.md")), reverse=True):
        if rec.endswith(".precheck.md"):
            continue
        text = open(rec).read()
        m = re.search(r"^sbatch:\s*(\S+)\s+sha256\s+([0-9a-f]{64})", text, re.M)
        if not m or os.path.abspath(m.group(1)) != script:
            continue
        if m.group(2) != s_sha:
            return f"record {os.path.basename(rec)} signed a different version of {script} (edited after sign-off)."
        c = re.search(r"^config:\s*(\S+)\s+sha256\s+([0-9a-f]{64})", text, re.M)
        if not c:
            return f"record {os.path.basename(rec)} has no `config: <path>  sha256 <hex>` line."
        if not os.path.exists(c.group(1)) or sha(c.group(1)) != c.group(2):
            return f"config {c.group(1)} changed (or is missing) since record {os.path.basename(rec)} was signed."
        if not re.search(r"^GO given by Oscar", text, re.M):
            return f"record {os.path.basename(rec)} has no GO line yet."
        return None
    return f"no pretrain-checklist record in {CHECKLIST_DIR} for {script}."


def main():
    try:
        data = json.load(sys.stdin)
        cmd = data.get("tool_input", {}).get("command", "") or ""
        cwd = data.get("cwd") or os.getcwd()
        if not re.search(r"\b(sbatch|srun)\b|train_wide\.py", cmd):
            return
        if TRAIN_RE.search(cmd) and "precheck.py" not in cmd:
            deny("Direct train_wide.py launches bypass the pre-training checklist. Launch through an sbatch "
                 "script that has a signed pretrain-checklist record (use the pretrain-checklist skill).")
        for script in sbatch_scripts(cmd, cwd):
            if not os.path.isfile(script):
                continue
            if not TRAIN_RE.search(re.sub(r"\\\s*\n", " ", open(script).read())):
                continue  # not a training job (e.g. preprocessing): allowed
            problem = check_record(script)
            if problem:
                deny(f"Pre-training gate: {problem} Run the pretrain-checklist skill and get Oscar's GO first.")
    except SystemExit:
        raise
    except Exception as e:  # never wedge the shell on a hook bug, but make it visible
        print(json.dumps({"systemMessage": f"pretrain gate hook error (allowed): {e!r}"}))


if __name__ == "__main__":
    main()
