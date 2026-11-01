import pytest

from gotrain.config import RunConfig
from gotrain.data.base import FinalEvalToken, TestSplitAccessError
from gotrain.data.synthetic import SyntheticClassification
from gotrain.experiments.runner import run_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.final_eval.evaluate_test import final_evaluate, results_path
from gotrain.final_eval.freeze import freeze_selection, load_manifest, manifest_path
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl, read_json, write_json


def _home(tmp_path) -> Home:
    return Home(tmp_path / "home").ensure()


def _train(home, model="cnn_small", lr=1e-3, screening=False):
    return run_experiment(home, RunConfig(task="synthetic", model=model, lr=lr, epochs=2, batch_size=16, max_minutes=2, num_workers=0, screening=screening, subset_fraction=0.5 if screening else 1.0), source="synthetic", device="cpu", log=lambda s: None)


def test_freeze_requires_best_pointer_and_refuses_screening(tmp_path):
    home = _home(tmp_path)
    with pytest.raises(ValueError, match="no best pointer"):
        freeze_selection(home, ["synthetic"], log=lambda s: None)
    screen = _train(home, screening=True)
    with pytest.raises(ValueError, match="not a completed full run"):
        freeze_selection(home, ["synthetic"], overrides={"synthetic": screen["experiment_id"]}, log=lambda s: None)


def test_freeze_then_final_eval_once(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    ExperimentStore(home).set_best("synthetic", rec["experiment_id"], by="test", rationale="only run")
    manifest = freeze_selection(home, ["synthetic"], log=lambda s: None)
    entry = manifest["tasks"]["synthetic"]
    assert entry["experiment_id"] == rec["experiment_id"] and entry["checkpoint_sha256"] == rec["checkpoint"]["best_sha256"]
    assert entry["model_version"] and entry["model_version"].startswith("mv-")
    with pytest.raises(FileExistsError):
        freeze_selection(home, ["synthetic"], log=lambda s: None)
    with pytest.raises(PermissionError):
        final_evaluate(home, device="cpu", acknowledge=False, log=lambda s: None)
    res = final_evaluate(home, device="cpu", acknowledge=True, num_workers=0, log=lambda s: None)
    t = res["tasks"]["synthetic"]
    assert t["status"] == "completed" and 0.0 <= t["test_metric"] <= 1.0 and t["n_test_samples"] == 64
    assert t["metric"]["name"] == "accuracy" and t["test_metrics"]["n_samples"] == 64
    exposures = list(iter_jsonl(home.final_eval / "exposure.jsonl"))
    assert len(exposures) == 1 and exposures[0]["task"] == "synthetic"
    res2 = final_evaluate(home, device="cpu", acknowledge=True, num_workers=0, log=lambda s: None)
    assert res2["tasks"]["synthetic"]["evaluated_at"] == t["evaluated_at"]
    assert len(list(iter_jsonl(home.final_eval / "exposure.jsonl"))) == 1
    with pytest.raises(PermissionError, match="no post-test tuning"):
        freeze_selection(home, ["synthetic"], replace=True, log=lambda s: None)


def test_edited_manifest_is_refused(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    ExperimentStore(home).set_best("synthetic", rec["experiment_id"], by="test")
    freeze_selection(home, ["synthetic"], log=lambda s: None)
    m = read_json(manifest_path(home))
    m["tasks"]["synthetic"]["best_val_metric"] = 0.999
    write_json(manifest_path(home), m)
    with pytest.raises(ValueError, match="edited after freezing"):
        load_manifest(manifest_path(home))
    with pytest.raises(ValueError):
        final_evaluate(home, device="cpu", acknowledge=True, num_workers=0, log=lambda s: None)
    assert not results_path(home).exists()


def test_results_from_other_manifest_block_reevaluation(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    ExperimentStore(home).set_best("synthetic", rec["experiment_id"], by="test")
    freeze_selection(home, ["synthetic"], log=lambda s: None)
    write_json(results_path(home), {"manifest_hash": "other", "tasks": {}})
    with pytest.raises(PermissionError, match="different frozen manifest"):
        final_evaluate(home, device="cpu", acknowledge=True, num_workers=0, log=lambda s: None)


def test_only_token_opens_test_split():
    with pytest.raises(TestSplitAccessError):
        SyntheticClassification("test", token="not a token")
    assert SyntheticClassification("test", token=FinalEvalToken("unit")).split == "test"
