"""End-to-end CLI flow on synthetic data: every documented command in execution order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(args, home, check=True):
    env = dict(os.environ, GOTRAIN_HOME=str(home), PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, "-m", "gotrain", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"command {args} failed ({proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def last_json(stdout: str):
    lines = stdout.strip().splitlines()
    for i in range(len(lines)):
        if lines[i][:1] in "{[":
            try:
                return json.loads("\n".join(lines[i:]))
            except json.JSONDecodeError:
                continue
    raise AssertionError(f"no JSON in output:\n{stdout}")


def test_cli_help_for_every_command():
    proc = run(["--help"], Path("."))
    for cmd in ("env-check", "test", "download", "inspect", "run", "smoke", "resume", "list", "show", "baseline", "agent", "sessions", "store-model", "list-versions", "retrieve-model", "verify-model", "promote-model", "register-dataset", "retrieve-dataset", "freeze", "final-eval", "report", "s3-smoke"):
        assert cmd in proc.stdout, cmd
        assert run([cmd, "--help"], Path(".")).returncode == 0


@pytest.mark.skipif(os.environ.get("GOTRAIN_SKIP_SLOW") == "1", reason="slow")
def test_full_synthetic_flow(tmp_path):
    home = tmp_path / "home"
    env_info = last_json(run(["env-check"], home).stdout)
    assert "torch" in env_info
    out = last_json(run(["run", "--task", "synthetic", "--model", "cnn_small", "--epochs", "2", "--batch-size", "16", "--max-minutes", "2", "--num-workers", "0", "--device", "cpu", "--tag", "baseline"], home).stdout)
    assert out["status"] == "completed"
    exp_id = out["experiment_id"]
    dup = last_json(run(["run", "--task", "synthetic", "--model", "cnn_small", "--epochs", "2", "--batch-size", "16", "--max-minutes", "2", "--num-workers", "0", "--device", "cpu", "--tag", "baseline"], home).stdout)
    assert dup["duplicate_of"] == exp_id
    listing = run(["list"], home).stdout
    assert exp_id in listing
    shown = last_json(run(["show", exp_id], home).stdout)
    assert shown["checkpoint"]["best_sha256"]
    assert run(["resume", exp_id, "--device", "cpu"], home).returncode == 0

    agent = last_json(run(["agent", "--task", "synthetic", "--scripted", "--session-minutes", "10", "--max-experiments", "3", "--run-max-minutes", "2", "--max-rounds", "2", "--device", "cpu"], home).stdout)
    assert agent["status"] == "completed" and "synthetic" in agent["finished_tasks"]
    sessions = last_json(run(["sessions"], home).stdout)
    assert sessions[0]["session_id"] == agent["session_id"]

    selected = agent["finished_tasks"]["synthetic"]["selected"]
    stored = last_json(run(["store-model", selected, "--note", "cli test"], home).stdout)
    vid = stored["version_id"]
    versions = last_json(run(["list-versions"], home).stdout)
    assert versions["versions"][0]["version_id"] == vid
    retrieved = last_json(run(["retrieve-model", vid, "--dest", str(tmp_path / "ret")], home).stdout)
    assert retrieved["sha256_verified"] and (tmp_path / "ret" / "checkpoint.pt").exists()
    verified = last_json(run(["verify-model", "all"], home).stdout)
    assert all(v["verified"] for v in verified)
    promoted = last_json(run(["promote-model", vid, "--rationale", "best validation score in cli test"], home).stdout)
    assert promoted["version_id"] == vid

    frozen = last_json(run(["freeze", "--task", "synthetic"], home).stdout)
    assert frozen["tasks"]["synthetic"]["experiment_id"] == selected
    refused = run(["final-eval"], home, check=False)
    assert refused.returncode != 0 and "refusing" in (refused.stderr + refused.stdout)
    final = last_json(run(["final-eval", "--i-understand-this-uses-the-test-split", "--device", "cpu", "--num-workers", "0"], home).stdout)
    assert final["synthetic"]["status"] == "completed"
    again = last_json(run(["final-eval", "--i-understand-this-uses-the-test-split", "--device", "cpu", "--num-workers", "0"], home).stdout)
    assert again["synthetic"]["evaluated_at"] == final["synthetic"]["evaluated_at"]
    report = run(["report"], home).stdout.strip()
    assert Path(report).exists() and "Test results" in Path(report).read_text()
