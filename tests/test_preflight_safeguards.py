import pytest
from gotrain.paths import Home
from gotrain.final_eval.evaluate_test import _reject_prior_exposure
from gotrain.final_eval.freeze import freeze_selection
from gotrain.util.io import append_jsonl


def test_no_exposure_allows_preflight(tmp_path):
    home = Home(tmp_path / "home").ensure()
    _reject_prior_exposure(home, "m-chesapeake")


def test_prior_exposure_blocks_same_task_only(tmp_path):
    home = Home(tmp_path / "home").ensure()
    append_jsonl(home.final_eval / "exposure.jsonl", {
        "kind": "test_split_opened", "task": "m-chesapeake"
    })
    with pytest.raises(PermissionError, match="prior test exposure"):
        _reject_prior_exposure(home, "m-chesapeake")
    _reject_prior_exposure(home, "m-eurosat")


def test_exposure_prevents_new_selection(tmp_path):
    home = Home(tmp_path / "home").ensure()
    append_jsonl(home.final_eval / "exposure.jsonl", {
        "kind": "test_split_opened", "task": "m-chesapeake"
    })
    with pytest.raises(PermissionError, match="Test exposure"):
        freeze_selection(home, [], store_versions=False)
