import os

import pytest

from gotrain.config import RunConfig
from gotrain.experiments.runner import config_hash, resume_experiment, run_experiment
from gotrain.experiments.store import ExperimentStore, GpuLock
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl, write_json


def _home(tmp_path) -> Home:
    return Home(tmp_path / "home").ensure()


def _run(home, model="cnn_small", **kw):
    cfg = dict(task="synthetic", model=model, epochs=2, batch_size=16, max_minutes=2, num_workers=0, tag="test")
    cfg.update(kw)
    return run_experiment(home, RunConfig(**cfg), source="synthetic", device="cpu", log=lambda s: None)


def test_record_complete_and_events_appended(tmp_path):
    home = _home(tmp_path)
    rec = _run(home)
    assert rec["status"] == "completed"
    for key in ("experiment_id", "run_config", "config_hash", "seed", "normalization", "model", "environment", "metric", "result", "compute", "checkpoint", "val_examples"):
        assert key in rec, key
    assert rec["metric"]["name"] == "accuracy" and rec["metric"]["direction"] == "max"
    assert rec["checkpoint"]["best_sha256"] and (home.root / rec["checkpoint"]["best_path"]).exists()
    assert rec["result"]["val_metrics_at_best"]["accuracy"] == rec["result"]["best_val_metric"]
    events = list(iter_jsonl(home.experiments / rec["experiment_id"] / "events.jsonl"))
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "created" and "status" in kinds
    assert (home.experiments / rec["experiment_id"] / "log.txt").exists()


def test_duplicate_config_is_not_retrained(tmp_path):
    home = _home(tmp_path)
    first = _run(home)
    second = _run(home)
    assert second["duplicate_of"] == first["experiment_id"]
    assert len(ExperimentStore(home).all_records()) == 1
    third = run_experiment(home, RunConfig(task="synthetic", model="cnn_small", epochs=2, batch_size=16, max_minutes=2, num_workers=0), source="synthetic", device="cpu", log=lambda s: None, allow_duplicate=True)
    assert "duplicate_of" not in third and third["experiment_id"] != first["experiment_id"]


def test_config_hash_ignores_narrative_fields():
    a = RunConfig(task="m-eurosat", model="cnn_small", hypothesis="x", rationale="y", tag="t1")
    b = RunConfig(task="m-eurosat", model="cnn_small", hypothesis="z", rationale="w", tag="t2")
    c = RunConfig(task="m-eurosat", model="cnn_small", lr=2e-3)
    assert config_hash(a, "geobench") == config_hash(b, "geobench")
    assert config_hash(a, "geobench") != config_hash(c, "geobench")
    assert config_hash(a, "geobench") != config_hash(a, "synthetic")


def test_screening_runs_are_excluded_from_ranking_and_selection(tmp_path):
    home = _home(tmp_path)
    full = _run(home, lr=1e-3)
    screen = _run(home, lr=2e-3, subset_fraction=0.5, screening=True)
    store = ExperimentStore(home)
    ranking = store.ranking("synthetic", source="synthetic")
    assert [r["experiment_id"] for r in ranking] == [full["experiment_id"]]
    with pytest.raises(ValueError, match="screening"):
        store.set_best("synthetic", screen["experiment_id"], by="test")
    pointer = store.set_best("synthetic", full["experiment_id"], by="test", rationale="only full run")
    assert store.best("synthetic")["experiment_id"] == pointer["experiment_id"]


def test_resume_via_runner_after_interruption(tmp_path):
    home = _home(tmp_path)
    store = ExperimentStore(home)
    calls = {"n": 0}

    def stop_after_first_epoch():
        calls["n"] += 1
        return calls["n"] > 1

    rec = run_experiment(home, RunConfig(task="synthetic", model="cnn_small", epochs=3, batch_size=16, max_minutes=2, num_workers=0), source="synthetic", device="cpu", log=lambda s: None, stop_requested=stop_after_first_epoch)
    assert rec["status"] == "interrupted"
    assert rec["result"]["epochs_completed"] == 1
    done = resume_experiment(home, rec["experiment_id"], device="cpu", log=lambda s: None)
    assert done["status"] == "completed"
    assert done["result"]["epochs_completed"] == 3
    assert done["compute"]["resume_count"] == 1
    again = resume_experiment(home, rec["experiment_id"], device="cpu", log=lambda s: None)
    assert again["result"]["epochs_completed"] == 3
    assert len(store.all_records()) == 1


def test_gpu_lock_blocks_second_job_and_ignores_stale(tmp_path):
    home = _home(tmp_path)
    write_json(home.locks / "training.lock", {"pid": os.getppid(), "experiment_id": "exp-a"})
    other = GpuLock(home)
    with pytest.raises(RuntimeError, match="another training job"):
        other.acquire("exp-b")
    write_json(home.locks / "training.lock", {"pid": 999999999, "experiment_id": "exp-dead"})
    other.acquire("exp-c")
    other.release()
    assert not (home.locks / "training.lock").exists()


def test_invalid_experiment_id_rejected(tmp_path):
    store = ExperimentStore(_home(tmp_path))
    with pytest.raises(ValueError):
        store.read("../escape")
    with pytest.raises(FileNotFoundError):
        store.read("exp-does-not-exist")


def test_auto_best_never_overrides_agent_or_manual_selection(tmp_path):
    home = _home(tmp_path)
    first = _run(home, lr=1e-3)
    store = ExperimentStore(home)
    store.set_best("synthetic", first["experiment_id"], by="agent", rationale="chosen")
    _run(home, lr=2e-3)
    _run(home, lr=5e-4)
    assert store.best("synthetic")["experiment_id"] == first["experiment_id"]
    assert store.best("synthetic")["selected_by"] == "agent"


def test_partial_validation_runs_cannot_be_selected(tmp_path):
    home = _home(tmp_path)
    partial = run_experiment(home, RunConfig(task="synthetic", model="cnn_small", epochs=1, batch_size=16, max_minutes=2, num_workers=0, val_limit=8), source="synthetic", device="cpu", log=lambda s: None)
    assert partial["partial_validation"] is True
    with pytest.raises(ValueError, match="partial-validation"):
        ExperimentStore(home).set_best("synthetic", partial["experiment_id"], by="test")
    assert ExperimentStore(home).best("synthetic") is None
