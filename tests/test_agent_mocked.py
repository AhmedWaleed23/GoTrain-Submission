import json

import pytest

pytest.importorskip("strands")

from gotrain.agent.fake_model import BadArgsPolicy, OverBudgetPolicy, ReactivePolicy, ScriptedModel
from gotrain.agent.runner import run_agent
from gotrain.agent.session import AgentSession, session_dir
from gotrain.agent.tools import ToolRuntime, build_tools, tool_names
from gotrain.config import Budget, Limits
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl


def _home(tmp_path) -> Home:
    return Home(tmp_path / "home").ensure()


def _budget(**kw) -> Budget:
    base = dict(agent_session_minutes=10, agent_max_experiments=3, agent_run_max_minutes=2)
    base.update(kw)
    return Budget(**base)


def test_reactive_session_reaches_selection_and_records_everything(tmp_path):
    home = _home(tmp_path)
    state = run_agent(home, ["synthetic"], ScriptedModel(ReactivePolicy("synthetic", "cnn_small")), _budget(), Limits.load(), device="cpu", source="synthetic", max_rounds=3, log=lambda s: None)
    assert state["status"] == "completed"
    assert state["experiments_launched"] == 2
    assert "synthetic" in state["finished_tasks"]
    store = ExperimentStore(home)
    records = store.all_records()
    assert len(records) == 2 and all(r["tag"] == "agent" for r in records)
    assert all(r["hypothesis"] and r["rationale"] and r["expected_tradeoff"] for r in records)
    best = store.best("synthetic")
    assert best["selected_by"] == "agent"
    assert best["best_val_metric"] == max(r["result"]["best_val_metric"] for r in records)
    kinds = [d["kind"] for d in iter_jsonl(session_dir(home, state["session_id"]) / "decisions.jsonl")]
    for k in ("session_start", "propose_experiment", "experiment_result", "select_best", "finish_task", "session_end"):
        assert k in kinds
    transcript = list(iter_jsonl(session_dir(home, state["session_id"]) / "transcript.jsonl"))
    assert any(t.get("kind") == "round" and t.get("messages") for t in transcript)


def test_guard_blocks_runs_beyond_experiment_limit(tmp_path):
    home = _home(tmp_path)
    state = run_agent(home, ["synthetic"], ScriptedModel(OverBudgetPolicy("synthetic", "cnn_small")), _budget(agent_max_experiments=2), Limits.load(), device="cpu", source="synthetic", max_rounds=2, log=lambda s: None)
    assert state["experiments_launched"] == 2
    assert len(ExperimentStore(home).all_records()) == 2
    decisions = list(iter_jsonl(session_dir(home, state["session_id"]) / "decisions.jsonl"))
    blocks = [d for d in decisions if d["kind"] == "guard_block"]
    assert blocks and "experiment limit reached" in blocks[0]["problems"][0]


def test_bad_arguments_are_rejected_without_training(tmp_path):
    home = _home(tmp_path)
    state = run_agent(home, ["synthetic"], ScriptedModel(BadArgsPolicy("synthetic")), _budget(), Limits.load(), device="cpu", source="synthetic", max_rounds=1, log=lambda s: None)
    assert state["experiments_launched"] == 0
    assert ExperimentStore(home).all_records() == []
    decisions = list(iter_jsonl(session_dir(home, state["session_id"]) / "decisions.jsonl"))
    block = next(d for d in decisions if d["kind"] == "guard_block")
    joined = " ".join(block["problems"])
    assert "lr" in joined and "hypothesis" in joined


def test_tools_have_no_test_split_access_and_validate_directly(tmp_path):
    home = _home(tmp_path)
    limits = Limits(**{**Limits.load().__dict__, "allowed_tasks": ["synthetic"]})
    session = AgentSession(home, _budget(), limits, tasks=["synthetic"], source="synthetic")
    rt = ToolRuntime(home=home, session=session, limits=limits, budget=_budget(), device="cpu", source="synthetic", log=lambda s: None)
    tools = build_tools(rt)
    names = tool_names(tools)
    assert "run_experiment" in names and "select_best_model" in names
    assert not any("test" in n or "final" in n for n in names)
    by_name = {getattr(t, "tool_name", None): t for t in tools}
    rejected = json.loads(by_name["run_experiment"](task="synthetic", model="cnn_small", lr=1e-3, batch_size=16, epochs=1, max_minutes=50.0, hypothesis="A sentence long enough to pass.", rationale="Another sentence long enough.", expected_tradeoff="Costs nothing much really."))
    assert rejected["status"] == "rejected" and any("session budget" in p or "per-run cap" in p for p in rejected["problems"])
    assert session.state["experiments_launched"] == 0
    err = json.loads(by_name["select_best_model"](task="synthetic", experiment_id="exp-nope", rationale="long enough rationale here"))
    assert "error" in err
    unfinished = json.loads(by_name["finish_task"](task="synthetic", summary="nothing selected"))
    assert "error" in unfinished


def test_session_resume_keeps_counters(tmp_path):
    home = _home(tmp_path)
    limits = Limits.load()
    s1 = AgentSession(home, _budget(), limits, tasks=["m-eurosat"])
    s1.state["tool_calls"] = 5
    s1.state["experiments_launched"] = 1
    s1.save()
    s2 = AgentSession(home, _budget(), limits, session_id=s1.session_id)
    assert s2.state["tool_calls"] == 5 and s2.state["experiments_launched"] == 1
    assert s2.state["resume_count"] == 1
    assert s2.experiments_remaining() == 2


def test_session_resume_keeps_elapsed_time(tmp_path):
    home = _home(tmp_path)
    limits = Limits.load()
    s1 = AgentSession(home, _budget(agent_session_minutes=10), limits, tasks=["m-eurosat"])
    s1.state["elapsed_seconds"] = 9 * 60
    s1.state["elapsed_before_seconds"] = 9 * 60
    from gotrain.util.io import write_json

    write_json(s1.dir / "state.json", s1.state)
    s2 = AgentSession(home, _budget(agent_session_minutes=10), limits, session_id=s1.session_id)
    assert 0.9 <= s2.remaining_minutes() <= 1.0
    ok, reason = s2.can_launch(1.0)
    assert not ok and "remain" in reason
    s3 = AgentSession(home, _budget(agent_session_minutes=30), limits, session_id=s1.session_id)
    assert 20.5 <= s3.remaining_minutes() <= 21.0
