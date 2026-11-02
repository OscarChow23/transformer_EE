#!/usr/bin/env python
"""TransformerEE experiment loop: trial specs -> run folders -> GO -> submission (issue #2).

    loop.py resolve   TRIAL         show each run's effective config vs the baseline (writes nothing)
    loop.py prepare   TRIAL         write RUNS_ROOT/TRIAL/: trial.yaml, <horn>/effective_config.json,
                                    <horn>/train.sbatch, resolved.json
    loop.py go-header TRIAL         print the hash lines a GO record must contain (skill pastes them)
    loop.py check-go  TRIAL         verify RUNS_ROOT/TRIAL/go-record.md against files and checkout
    loop.py submit    TRIAL         Oscar only: refuse unless signed, clean and at the signed
                                    commit, then sbatch every horn
    loop.py job-start TRIAL HORN    first step inside each job: record provenance.json, abort
                                    unless clean, at the signed commit and nothing changed

A trial spec is pcnumu/trials/TRIAL/trial.yaml. A derived trial names a baseline trial and lists
only `overrides` (dotted keys that must already exist in the baseline config), optional
`horn_overrides: {fhc: {...}}`, `add` (intentionally new keys) and `resources`. A root trial
(`imported`) pins finished runs' saved input.json files by sha256. The loop sets each run's
`save_path` (RUNS_ROOT/TRIAL/HORN) and `model_phys_name`; specs may not.
"""
import argparse
import copy
import datetime
import glob
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile

import yaml

def _canonical(path):
    """Prefer the /data/nova/... spelling used in every config; /data links into /.data."""
    if path.startswith("/.data/"):
        alt = "/data/" + path[len("/.data/"):]
        if os.path.exists(alt) and os.path.samefile(alt, path):
            return alt
    return path


REPO = _canonical(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRIALS_DIR = os.path.join(REPO, "pcnumu", "trials")
TEMPLATE = os.path.join(REPO, "pcnumu", "loop", "train.sbatch.template")
RUNS_ROOT = os.environ.get("PCNUMU_RUNS_ROOT", "/data/nova/chow/pcnumu-runs")
GIT = "/data/chow/tools/git-env/bin/git"
sys.path.insert(0, os.path.join(REPO, "pcnumu", "pretrain-checklist", "scripts"))

TRIAL_RE = re.compile(r"^r\d{2}[a-z]?(_[A-Za-z0-9._-]+)?$")
IDENTITY_KEYS = ("save_path", "model_phys_name")
TEST_SET_KEYS = ("data_path", "split_column", "target")  # changing these changes what is measured
SPEC_KEYS = {"trial", "status", "baseline", "hypothesis", "horns", "imported", "overrides",
             "horn_overrides", "add", "resources"}
DEFAULT_RESOURCES = {"mem": "80G", "time": "08:00:00", "cpus": 12, "partition": "normalrun"}
WANDB_ARGV = ["--wandb-project", "pcnumu", "--wandb-id", "<id>", "--wandb-dir", "<dir>"]


class LoopError(Exception):
    pass


# ---------- small helpers ----------
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def dhas(d, dotted):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True


def dset(d, dotted, value):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def git(code_dir, *args):
    exe = GIT if os.path.exists(GIT) else "git"
    return subprocess.run([exe, "-C", code_dir, *args], capture_output=True, text=True, check=True).stdout.rstrip()


def git_state(code_dir):
    """commit, branch and dirty flag (tracked files only; untracked files are ignored)."""
    status = git(code_dir, "status", "--porcelain", "--untracked-files=no")
    return dict(commit=git(code_dir, "rev-parse", "HEAD"), branch=git(code_dir, "rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(status), dirty_files=status.splitlines())


# ---------- specs and resolution ----------
def spec_path(trial):
    return os.path.join(TRIALS_DIR, trial, "trial.yaml")


def load_spec(trial):
    path = spec_path(trial)
    if not os.path.exists(path):
        raise LoopError(f"no trial spec {path}")
    spec = yaml.safe_load(open(path))
    if not isinstance(spec, dict):
        raise LoopError(f"{path}: not a mapping")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise LoopError(f"{path}: unknown keys {sorted(unknown)}")
    if spec.get("trial") != trial or not TRIAL_RE.match(trial):
        raise LoopError(f"{path}: `trial` must equal the folder name and look like rNN_<description>")
    if not spec.get("hypothesis") or not spec.get("horns"):
        raise LoopError(f"{path}: needs a hypothesis and horns")
    bad_res = set(spec.get("resources") or {}) - set(DEFAULT_RESOURCES)
    if bad_res:
        raise LoopError(f"{path}: unknown resources {sorted(bad_res)}; allowed {sorted(DEFAULT_RESOURCES)}")
    if bool(spec.get("imported")) == bool(spec.get("baseline")):
        raise LoopError(f"{path}: give exactly one of `baseline` (derived trial) or `imported` (root trial)")
    return spec


def run_identity(trial, horn):
    return {"save_path": os.path.join(RUNS_ROOT, trial, horn),
            "model_phys_name": f"NOvA_PC_Numu_mprod6_1_fd_{horn}_{trial}"}


def resolve(trial, _seen=()):
    """{horn: effective config} for a trial, following baselines down to a root trial."""
    if trial in _seen:
        raise LoopError(f"baseline cycle: {' -> '.join(_seen + (trial,))}")
    spec = load_spec(trial)
    if spec.get("imported"):
        out = {}
        for horn in spec["horns"]:
            imp = spec["imported"].get(horn) or {}
            if sha256_file(imp["path"]) != imp["sha256"]:
                raise LoopError(f"{trial}/{horn}: {imp['path']} no longer matches the pinned sha256")
            out[horn] = json.load(open(imp["path"]))
        return out
    base = resolve(spec["baseline"], _seen + (trial,))
    out = {}
    for horn in spec["horns"]:
        if horn not in base:
            raise LoopError(f"{trial}: baseline {spec['baseline']} has no {horn} run")
        cfg = copy.deepcopy(base[horn])
        changes = dict(spec.get("overrides") or {})
        changes.update((spec.get("horn_overrides") or {}).get(horn) or {})
        for key, value in changes.items():
            if key in IDENTITY_KEYS:
                raise LoopError(f"{trial}: `{key}` is set by the loop, not by a spec")
            if not dhas(cfg, key):
                raise LoopError(f"{trial}: override `{key}` is not in the baseline config "
                                "(typo? an intentionally new key goes under `add`)")
            dset(cfg, key, value)
        for key, value in (spec.get("add") or {}).items():
            if dhas(cfg, key):
                raise LoopError(f"{trial}: `add` key `{key}` already exists in the baseline; use `overrides`")
            dset(cfg, key, value)
        cfg.update(run_identity(trial, horn))
        out[horn] = cfg
    return out


def compare(base_cfg, cfg):
    """(settings changes, identity changes) as {key: [baseline, trial]}."""
    a, b = flatten(base_cfg), flatten(cfg)
    diff = {k: [a.get(k, "<absent>"), b.get(k, "<absent>")] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    ident = {k: v for k, v in diff.items() if k in IDENTITY_KEYS}
    return {k: v for k, v in diff.items() if k not in IDENTITY_KEYS}, ident


def launcher_check(cfg):
    """Run the config through precheck.simulate() (a mirror of train_wide.main()) with the
    template's command line. Returns the predicted model dir; raises if the launcher would
    fill or change any key, i.e. if the config is not already effective."""
    import contextlib  # pylint: disable=C0415
    import io  # pylint: disable=C0415
    import precheck  # pylint: disable=C0415

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "effective_config.json")
        json.dump(cfg, open(path, "w"))
        rep = precheck.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, sim, _, mhash = precheck.simulate(REPO, ["--base-config", path] + WANDB_ARGV, rep)
    if sim != cfg:
        a, b = flatten(cfg), flatten(sim)
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise LoopError(f"train_wide.py would fill or change {changed}; put them in the config explicitly")
    problems = [i["msg"] for i in rep.items if i["level"] in ("FAIL", "WARN")]
    if problems:
        raise LoopError("launcher check: " + "; ".join(problems))
    return os.path.join(cfg["save_path"], "model_" + mhash)


# ---------- run folders ----------
def render_sbatch(trial, horn, run_dir, resources):
    text = open(TEMPLATE).read()
    values = dict(DEFAULT_RESOURCES, **(resources or {}))
    values.update(trial=trial, horn=horn, run_dir=run_dir, code_dir=REPO, job_name=f"{trial[:3]}_{horn}")
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    left = re.findall(r"(?<!\$)\{[a-z_]+\}", text)
    if left:
        raise LoopError(f"template placeholders not filled: {left}")
    return text


def plan(trial):
    """Everything prepare would write, as {relative path: bytes}, plus the summary."""
    spec = load_spec(trial)
    if spec.get("imported"):
        raise LoopError(f"{trial} is a root (imported) trial; there is nothing to prepare")
    base = resolve(spec["baseline"])
    cfgs = resolve(trial)
    files = {"trial.yaml": open(spec_path(trial), "rb").read()}
    summary = {"trial": trial, "baseline": spec["baseline"], "spec_sha256": sha256_file(spec_path(trial)),
               "hypothesis": spec["hypothesis"], "runs": {}}
    for horn, cfg in cfgs.items():
        model_dir = launcher_check(cfg)
        settings, ident = compare(base[horn], cfg)
        cfg_bytes = (json.dumps(cfg, indent=4) + "\n").encode()
        sb_bytes = render_sbatch(trial, horn, cfg["save_path"], spec.get("resources")).encode()
        files[f"{horn}/effective_config.json"] = cfg_bytes
        files[f"{horn}/train.sbatch"] = sb_bytes
        summary["runs"][horn] = {
            "run": f"{trial}-{horn}", "run_dir": cfg["save_path"], "predicted_model_dir": model_dir,
            "settings_vs_baseline": settings, "identity_vs_baseline": ident,
            "test_set_keys_changed": sorted(k for k in settings if k.split(".")[0] in TEST_SET_KEYS),
            "effective_config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
            "sbatch_sha256": hashlib.sha256(sb_bytes).hexdigest()}
    files["resolved.json"] = (json.dumps(summary, indent=2) + "\n").encode()
    return files, summary


def prepare(trial):
    files, summary = plan(trial)
    tdir = os.path.join(RUNS_ROOT, trial)
    existing = {rel: open(os.path.join(tdir, rel), "rb").read() for rel in files if os.path.exists(os.path.join(tdir, rel))}
    if existing and existing == files and len(existing) == len(files):
        return summary, "unchanged"
    if os.path.exists(os.path.join(tdir, "go-record.md")):
        raise LoopError(f"{tdir} has a GO record; signed run folders are not rewritten. "
                        "Make a new trial, or remove the GO record deliberately and re-sign.")
    for horn in summary["runs"]:
        if glob.glob(os.path.join(tdir, horn, "model_*")) or os.path.exists(os.path.join(tdir, horn, "provenance.json")):
            raise LoopError(f"{tdir}/{horn} has already started training; not rewriting it")
    for rel, data in files.items():
        os.makedirs(os.path.dirname(os.path.join(tdir, rel)), exist_ok=True)
        with open(os.path.join(tdir, rel), "wb") as f:
            f.write(data)
    return summary, "written"


# ---------- GO record ----------
GO_LINE = re.compile(r"^GO given by Oscar\b", re.M)


def go_header(trial, code_dir=REPO):
    """The exact lines a GO record must start with, from the prepared folder and HEAD."""
    tdir = os.path.join(RUNS_ROOT, trial)
    spec = load_spec(trial)
    lines = [f"# GO record: {trial}", "",
             f"trial_spec: {os.path.join(tdir, 'trial.yaml')}  sha256 {sha256_file(os.path.join(tdir, 'trial.yaml'))}",
             f"commit: {git_state(code_dir)['commit']}"]
    for horn in spec["horns"]:
        for kind, name in (("config", "effective_config.json"), ("sbatch", "train.sbatch")):
            p = os.path.join(tdir, horn, name)
            lines.append(f"{kind}[{horn}]: {p}  sha256 {sha256_file(p)}")
    return "\n".join(lines) + "\n"


def parse_go(trial):
    path = os.path.join(RUNS_ROOT, trial, "go-record.md")
    if not os.path.exists(path):
        raise LoopError(f"no GO record at {path} (run the pretrain-checklist skill)")
    text = open(path).read()
    rec = {"path": path, "text": text, "files": {}, "go": bool(GO_LINE.search(text))}
    m = re.search(r"^commit:\s*([0-9a-f]{40})\s*$", text, re.M)
    rec["commit"] = m.group(1) if m else None
    for m in re.finditer(r"^(trial_spec|config\[\w+\]|sbatch\[\w+\]):\s*(\S+)\s+sha256\s+([0-9a-f]{64})\s*$", text, re.M):
        rec["files"][m.group(1)] = (m.group(2), m.group(3))
    return rec


def check_go(trial, code_dir=REPO):
    """Problems that block submission; empty list means the GO record holds."""
    problems = []
    try:
        rec = parse_go(trial)
        spec = load_spec(trial)
    except LoopError as e:
        return [str(e)]
    if not rec["go"]:
        problems.append("GO record has no `GO given by Oscar` line")
    if not rec["commit"]:
        problems.append("GO record has no `commit: <40-hex>` line")
    need = ["trial_spec"] + [f"{k}[{h}]" for h in spec["horns"] for k in ("config", "sbatch")]
    for key in need:
        if key not in rec["files"]:
            problems.append(f"GO record lacks a `{key}: <path>  sha256 <hex>` line")
            continue
        path, digest = rec["files"][key]
        if not os.path.exists(path) or sha256_file(path) != digest:
            problems.append(f"{path} changed (or is missing) since GO")
    if "trial_spec" in rec["files"] and sha256_file(spec_path(trial)) != rec["files"]["trial_spec"][1]:
        problems.append(f"the spec in Git ({spec_path(trial)}) differs from the signed one")
    if not problems:  # the signed folder must still be what the spec resolves to
        try:
            files, _ = plan(trial)
        except LoopError as e:
            problems.append(f"spec no longer resolves: {e}")
        else:
            for rel, data in files.items():
                p = os.path.join(RUNS_ROOT, trial, rel)
                if rel != "resolved.json" and (not os.path.exists(p) or open(p, "rb").read() != data):
                    problems.append(f"{p} is not what the spec resolves to now")
    try:
        st = git_state(code_dir)
    except (OSError, subprocess.CalledProcessError) as e:
        return problems + [f"cannot read Git state: {e}"]
    if st["dirty"]:
        problems.append("checkout has uncommitted changes to tracked files: " + "; ".join(st["dirty_files"]))
    if rec["commit"] and st["commit"] != rec["commit"]:
        problems.append(f"checkout is at {st['commit'][:12]}, GO was signed at {rec['commit'][:12]}")
    try:
        git(code_dir, "ls-files", "--error-unmatch", os.path.relpath(spec_path(trial), code_dir))
    except subprocess.CalledProcessError:
        problems.append(f"{spec_path(trial)} is not committed")
    return problems


# ---------- submission and job start ----------
def submit(trial):
    problems = check_go(trial)
    spec = load_spec(trial)
    for horn in spec["horns"]:
        rd = os.path.join(RUNS_ROOT, trial, horn)
        if glob.glob(os.path.join(rd, "slurm-*.out")) or os.path.exists(os.path.join(rd, "provenance.json")):
            problems.append(f"{rd} was already submitted")
    if problems:
        raise LoopError("submission refused:\n  - " + "\n  - ".join(problems))
    log = os.path.join(RUNS_ROOT, trial, "submissions.log")
    for horn in spec["horns"]:
        sb = os.path.join(RUNS_ROOT, trial, horn, "train.sbatch")
        out = subprocess.run(["sbatch", sb], capture_output=True, text=True, check=True).stdout.strip()
        with open(log, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {trial}-{horn} {out} ({sb})\n")
        print(f"{trial}-{horn}: {out}")


def job_start(trial, horn, code_dir=REPO):
    """Inside the job, before training: write provenance.json; raise if anything is off."""
    rd = os.path.join(RUNS_ROOT, trial, horn)
    prov_path = os.path.join(rd, "provenance.json")
    if os.path.exists(prov_path) or glob.glob(os.path.join(rd, "model_*")):
        raise LoopError(f"{rd} already holds a started run; refusing to train into it again")
    try:
        st = git_state(code_dir)
    except (OSError, subprocess.CalledProcessError) as e:
        st = dict(commit=None, branch=None, dirty=None, dirty_files=[f"git failed: {e}"])
    problems = check_go(trial, code_dir)
    cfg = os.path.join(rd, "effective_config.json")
    prov = {"trial": trial, "horn": horn, "run": f"{trial}-{horn}", "code_dir": code_dir,
            "commit": st["commit"], "branch": st["branch"], "dirty": st["dirty"], "dirty_files": st["dirty_files"],
            "effective_config_sha256": sha256_file(cfg) if os.path.exists(cfg) else None,
            "go_record_sha256": sha256_file(os.path.join(RUNS_ROOT, trial, "go-record.md"))
            if os.path.exists(os.path.join(RUNS_ROOT, trial, "go-record.md")) else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "refused": problems or None}
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    if problems:
        raise LoopError("job start refused (see provenance.json):\n  - " + "\n  - ".join(problems))
    return prov


# ---------- CLI ----------
def print_resolve(trial):
    spec = load_spec(trial)
    cfgs = resolve(trial)
    if spec.get("imported"):
        for horn, cfg in cfgs.items():
            print(f"{trial}-{horn}: root trial, effective config imported from {spec['imported'][horn]['path']}")
        return
    base = resolve(spec["baseline"])
    for horn, cfg in cfgs.items():
        settings, ident = compare(base[horn], cfg)
        print(f"{trial}-{horn} vs {spec['baseline']}-{horn}")
        for k, (a, b) in settings.items():
            print(f"  setting   {k}: {a!r} -> {b!r}" + ("   [changes the test set/targets]" if k.split(".")[0] in TEST_SET_KEYS else ""))
        for k, (a, b) in ident.items():
            print(f"  identity  {k}: {b}")
        print(f"  predicted model dir: {launcher_check(cfg)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["resolve", "prepare", "go-header", "check-go", "submit", "job-start"])
    p.add_argument("trial")
    p.add_argument("horn", nargs="?")
    a = p.parse_args()
    try:
        if a.command == "resolve":
            print_resolve(a.trial)
        elif a.command == "prepare":
            summary, state = prepare(a.trial)
            print(f"{os.path.join(RUNS_ROOT, a.trial)}: {state}")
            for horn, r in summary["runs"].items():
                print(f"  {r['run']}: {len(r['settings_vs_baseline'])} setting(s) changed "
                      f"{sorted(r['settings_vs_baseline'])}; model dir {r['predicted_model_dir']}")
        elif a.command == "go-header":
            print(go_header(a.trial), end="")
        elif a.command == "check-go":
            problems = check_go(a.trial)
            print("GO record holds: ready to submit" if not problems else "NOT ready:\n  - " + "\n  - ".join(problems))
            sys.exit(1 if problems else 0)
        elif a.command == "submit":
            submit(a.trial)
        elif a.command == "job-start":
            if not a.horn:
                p.error("job-start needs TRIAL HORN")
            prov = job_start(a.trial, a.horn)
            print(f"provenance: commit {prov['commit']} dirty={prov['dirty']}")
    except LoopError as e:
        sys.exit(f"loop: {e}")


if __name__ == "__main__":
    main()
