"""Acceptance tests for the experiment loop (issue #2).

    python pcnumu/loop/test_loop.py     (transformer_ee env; or pytest)

Uses the real r05 and r06 specs and r05's saved input.json files (read-only). Everything that
writes goes to temporary directories: a throwaway Git repo stands in for the checkout, and
RUNS_ROOT points at a temp folder.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop  # noqa: E402

R05, R06 = "r05_v3data-allsamples_v3features", "r06_flat-alpha001"
REAL_TRIALS = loop.TRIALS_DIR
HOOK = os.path.join(loop.REPO, "pcnumu", "pretrain-checklist", "scripts", "gate_hook.py")


def _git(repo, *args):
    exe = loop.GIT if os.path.exists(loop.GIT) else "git"
    subprocess.run([exe, "-C", repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
                   check=True, capture_output=True)


class Sandbox:
    """Temp Git repo holding copies of the trial specs, plus a temp RUNS_ROOT."""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        trials = os.path.join(self.repo, "pcnumu", "trials")
        for t in (R05, R06):
            shutil.copytree(os.path.join(REAL_TRIALS, t), os.path.join(trials, t))
        with open(os.path.join(self.repo, "code.py"), "w") as f:
            f.write("x = 1\n")
        _git(self.repo, "init", "-q")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-q", "-m", "init")
        self.saved = (loop.TRIALS_DIR, loop.RUNS_ROOT)
        loop.TRIALS_DIR, loop.RUNS_ROOT = trials, os.path.join(self.tmp, "runs")
        return self

    def __exit__(self, *exc):
        loop.TRIALS_DIR, loop.RUNS_ROOT = self.saved
        shutil.rmtree(self.tmp)

    def sign(self, trial, go=True):
        text = loop.go_header(trial, code_dir=self.repo) + ("\nGO given by Oscar at test time\n" if go else "")
        with open(os.path.join(loop.RUNS_ROOT, trial, "go-record.md"), "w") as f:
            f.write(text)


def raises(fn, *args, contains=""):
    try:
        fn(*args)
    except loop.LoopError as e:
        assert contains in str(e), f"expected {contains!r} in {e}"
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} did not raise")


# ---- criterion 1: r05 with no overrides resolves to r05's saved effective config ----
def test_r05_resolves_to_saved_input_json():
    spec = loop.load_spec(R05)
    cfgs = loop.resolve(R05)
    for horn in ("fhc", "rhc"):
        assert cfgs[horn] == json.load(open(spec["imported"][horn]["path"])), horn


# ---- criterion 2: draft r06 differs from r05 only in alpha ----
def test_r06_differs_from_r05_only_in_alpha():
    base, r06 = loop.resolve(R05), loop.resolve(R06)
    for horn in ("fhc", "rhc"):
        settings, ident = loop.compare(base[horn], r06[horn])
        assert settings == {"weight.kwargs.alpha": [0.05, 0.01]}, settings
        assert set(ident) == {"save_path", "model_phys_name"}
        assert r06[horn]["save_path"] == os.path.join(loop.RUNS_ROOT, R06, horn)
        model_dir = loop.launcher_check(r06[horn])  # train_wide.py would leave it unchanged
        assert os.path.dirname(model_dir) == r06[horn]["save_path"]


def test_spec_mistakes_are_refused():
    with Sandbox():
        path = loop.spec_path(R06)
        good = open(path).read()
        for bad, msg in ((good.replace("weight.kwargs.alpha", "weight.kwarg.alpha"), "not in the baseline"),
                         (good.replace("  weight.kwargs.alpha: 0.01", "  weight.kwargs.alpha: 0.01\n  save_path: /tmp/x"),
                          "set by the loop"),
                         (good + "  gpus: 2\n", "unknown resources"),
                         (good.replace("baseline: r05", "imported: {fhc: {path: x, sha256: y}}\nbaseline: r05"), None)):
            open(path, "w").write(bad)
            if msg == "unknown resources":
                raises(loop.load_spec, R06, contains=msg)
            elif msg:
                raises(loop.resolve, R06, contains=msg)
            else:
                raises(loop.load_spec, R06)
        open(path, "w").write(good)


# ---- criteria 3-5: one GO per trial, provenance, dirty/commit refusal ----
def test_go_record_and_job_start():
    with Sandbox() as sb:
        summary, state = loop.prepare(R06)
        assert state == "written" and set(summary["runs"]) == {"fhc", "rhc"}
        assert loop.prepare(R06)[1] == "unchanged"
        sbatch = open(os.path.join(loop.RUNS_ROOT, R06, "fhc", "train.sbatch")).read()
        assert "loop.py job-start" in sbatch and "effective_config.json" in sbatch and "{" + "mem}" not in sbatch

        assert any("no GO record" in p for p in loop.check_go(R06, sb.repo))
        sb.sign(R06, go=False)
        assert any("GO given by Oscar" in p for p in loop.check_go(R06, sb.repo))
        sb.sign(R06)  # one record covers both horns
        assert loop.check_go(R06, sb.repo) == [], loop.check_go(R06, sb.repo)

        cfg = os.path.join(loop.RUNS_ROOT, R06, "rhc", "effective_config.json")
        original = open(cfg).read()
        open(cfg, "w").write(original.replace("0.01", "0.02"))
        assert any("changed" in p for p in loop.check_go(R06, sb.repo))
        open(cfg, "w").write(original)

        code = os.path.join(sb.repo, "code.py")
        open(code, "w").write("x = 2\n")  # dirty tracked file
        assert any("uncommitted" in p for p in loop.check_go(R06, sb.repo))
        raises(loop.job_start, R06, "fhc", sb.repo, contains="uncommitted")
        prov_path = os.path.join(loop.RUNS_ROOT, R06, "fhc", "provenance.json")
        prov = json.load(open(prov_path))
        assert prov["dirty"] is True and prov["refused"]
        os.remove(prov_path)

        _git(sb.repo, "commit", "-q", "-am", "edit after GO")  # clean but not the signed commit
        assert any("GO was signed at" in p for p in loop.check_go(R06, sb.repo))
        _git(sb.repo, "reset", "-q", "--hard", "HEAD~1")

        prov = loop.job_start(R06, "fhc", sb.repo)  # clean, at the signed commit
        assert prov["dirty"] is False and prov["refused"] is None and len(prov["commit"]) == 40
        raises(loop.job_start, R06, "fhc", sb.repo, contains="already holds a started run")

        spec = loop.spec_path(R06)
        open(spec, "a").write("# edited after GO\n")
        raises(loop.prepare, R06, contains="has a GO record")


# ---- criterion 6: nothing submits on Oscar's behalf ----
def _hook(command, cwd="/"):
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"tool_input": {"command": command}, "cwd": cwd}),
                         capture_output=True, text=True).stdout
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if "permissionDecision" in out else "allow"


def test_hook_denies_claude_submissions():
    with tempfile.TemporaryDirectory() as tmp:
        train, prep = os.path.join(tmp, "train.sbatch"), os.path.join(tmp, "prep.sbatch")
        open(train, "w").write("#!/bin/bash\npython train_wide.py \\\n  --base-config x.json\n")
        open(prep, "w").write("#!/bin/bash\npython -I make_split_csv.py\n")
        assert _hook(f"python pcnumu/loop/loop.py submit {R06}") == "deny"
        assert _hook(f"python -I pcnumu/loop/loop.py job-start {R06} fhc") == "deny"
        assert _hook(f"sbatch --mem 80G {train}") == "deny"
        assert _hook("cd /x && python train_wide.py --base-config a.json") == "deny"
        assert _hook(f"sbatch {prep}") == "allow"
        assert _hook(f"python pcnumu/loop/loop.py prepare {R06}") == "allow"
        assert _hook("ls -la") == "allow"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
