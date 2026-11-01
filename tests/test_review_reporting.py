from types import SimpleNamespace

import pytest

from gotrain.agent.runner import reconcile_finished_selection
from gotrain.paths import Home
from gotrain.review.page import review_presentation
from gotrain.submission_check import check_review


def test_round_reconciles_stale_finished_selection_without_changing_pointer():
    events = []
    state = {"finished_tasks": {"m-nz-cattle": {"selected": "old", "best_val_metric": 0.8, "summary": "original statement"}}}
    session = SimpleNamespace(state=state, record_decision=lambda *a, **kw: events.append((a, kw)), save=lambda: None)
    pointer = {"experiment_id": "new", "best_val_metric": 0.84}
    store = SimpleNamespace(best=lambda task: pointer.copy())
    reconcile_finished_selection(session, store, "m-nz-cattle")
    assert state["finished_tasks"]["m-nz-cattle"]["selected"] == "new"
    assert state["finished_tasks"]["m-nz-cattle"]["best_val_metric"] == 0.84
    assert state["finished_tasks"]["m-nz-cattle"]["summary"] == "original statement"
    assert pointer == {"experiment_id": "new", "best_val_metric": 0.84}
    assert len(events) == 1
    reconcile_finished_selection(session, store, "m-nz-cattle")
    assert len(events) == 1


def test_reconciliation_does_not_finish_an_unfinished_task():
    session = SimpleNamespace(state={"finished_tasks": {}})
    reconcile_finished_selection(session, SimpleNamespace(), "m-nz-cattle")
    assert session.state["finished_tasks"] == {}


def test_checklist_command_uses_actual_home_and_task(tmp_path):
    home = Home(tmp_path / "cattle_home").ensure()
    result = check_review(home, ["m-nz-cattle"])
    assert result[0]["status"] == "pending"
    assert "--task m-nz-cattle" in result[0]["detail"]
    assert str(home.root) in result[0]["detail"]
    assert "m-pv4ger-seg" not in result[0]["detail"]
    other = check_review(home, ["m-eurosat"])
    assert other[0]["status"] == "manual"
    assert "review-build --task m-pv4ger-seg" not in other[0]["detail"]


def test_binary_page_labels_follow_task():
    def prov(task, n=2):
        return {"model": {"task": task, "architecture": {"n_classes": n}}}
    # The real-data guard must not break the existing three-class demo fixture.
    fixture = review_presentation(prov("synthetic", 3))
    assert fixture["title"] == "Review solar-panel predictions"
    cattle = review_presentation(prov("m-nz-cattle"))
    assert cattle["title"] == "Review cattle predictions"
    assert cattle["foreground"] == "cattle"
    assert "solar" not in str(cattle)
    solar = review_presentation(prov("m-pv4ger-seg"))
    assert solar["title"] == "Review solar-panel predictions"
    with pytest.raises(ValueError, match="binary segmentation"):
        review_presentation(prov("m-chesapeake", 7))
