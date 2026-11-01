import json
import os
import subprocess
import sys
from pathlib import Path

from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.artifacts.verify import verify_version
from gotrain.config import RunConfig
from gotrain.experiments.runner import run_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.submission_check import format_report, run_check
from gotrain.util.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _ready_home(tmp_path) -> tuple[Home, dict]:
    home = Home(tmp_path / "home").ensure()
    rec = run_experiment(home, RunConfig(task="synthetic", model="cnn_small", epochs=1, batch_size=16, max_minutes=2, num_workers=0), source="synthetic", device="cpu", log=lambda s: None)
    ExperimentStore(home).set_best("synthetic", rec["experiment_id"], by="agent", rationale="only run")
    store = FilesystemArtifactStore(home)
    v = store.store_model(rec["experiment_id"])
    verify_version(store, v["version_id"])
    return home, rec


def _statuses(result, prefix):
    return {i["check"]: i["status"] for i in result["items"] if i["check"].startswith(prefix)}


def test_ready_to_begin_final_eval_but_not_to_submit(tmp_path):
    home, rec = _ready_home(tmp_path)
    result = run_check(home, tasks=["synthetic"])
    st = _statuses(result, "synthetic:")
    assert st["synthetic: selected checkpoint"] == "ok"
    assert st["synthetic: normalization"] == "ok"
    assert st["synthetic: model version"] == "ok"
    assert st["synthetic: final test score"] == "pending"
    assert result["exit_code"] == 0
    assert "ready to begin final evaluation" in result["verdict"] and "NOT ready to submit" in result["verdict"]
    text = format_report(result)
    assert "VERDICT" in text and "ready to submit" not in text.replace("NOT ready to submit", "")


def test_mismatched_checkpoint_hash_is_blocking(tmp_path):
    home, rec = _ready_home(tmp_path)
    ck = home.root / rec["checkpoint"]["best_path"]
    ck.write_bytes(ck.read_bytes()[:-2] + b"xx")
    result = run_check(home, tasks=["synthetic"])
    assert _statuses(result, "synthetic:")["synthetic: selected checkpoint"] == "fail"
    assert result["exit_code"] == 1 and result["verdict"].startswith("NOT READY")


def test_missing_pointer_and_missing_home(tmp_path):
    home = Home(tmp_path / "empty").ensure()
    result = run_check(home, tasks=["m-eurosat"])
    assert _statuses(result, "m-eurosat:")["m-eurosat: selected checkpoint"] == "fail"
    assert result["exit_code"] == 1
    import pytest

    with pytest.raises(FileNotFoundError):
        run_check(Home(tmp_path / "does-not-exist"), tasks=["m-eurosat"])


def test_partial_final_evaluation_record_is_flagged(tmp_path):
    home, rec = _ready_home(tmp_path)
    write_json(home.final_eval / "results.json", {"manifest_hash": "abc", "tasks": {"synthetic": {"status": "failed", "error": "simulated crash", "experiment_id": rec["experiment_id"]}}})
    from gotrain.util.io import append_jsonl

    append_jsonl(home.final_eval / "exposure.jsonl", {"kind": "test_split_opened", "task": "synthetic"})
    result = run_check(home, tasks=["synthetic"])
    st = _statuses(result, "synthetic:")
    assert st["synthetic: final test score"] == "fail" and st["synthetic: test exposure"] == "fail"
    assert result["exit_code"] == 1 and result["final"]["exposure_without_result"] == ["synthetic"]


def test_completed_final_scores_change_verdict(tmp_path):
    home, rec = _ready_home(tmp_path)
    write_json(home.final_eval / "results.json", {"manifest_hash": "abc", "tasks": {"synthetic": {"status": "completed", "test_metric": 0.5, "metric": {"name": "accuracy"}, "n_test_samples": 64, "evaluated_at": "2026-11-01T10:00:00+00:00"}}})
    result = run_check(home, tasks=["synthetic"])
    assert result["final"]["final_complete"] and result["exit_code"] == 0
    assert "complete" in result["verdict"]


def test_cli_exit_codes_and_json(tmp_path):
    home, rec = _ready_home(tmp_path)
    env = dict(os.environ, GOTRAIN_HOME=str(home.root), PYTHONPATH=str(ROOT))
    out = tmp_path / "check.json"
    proc = subprocess.run([sys.executable, "-m", "gotrain", "submission-check", "--task", "synthetic", "--json", str(out)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "VERDICT" in proc.stdout and read_json(out)["exit_code"] == 0
    proc = subprocess.run([sys.executable, "-m", "gotrain", "--home", str(tmp_path / "nowhere"), "submission-check"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 2
