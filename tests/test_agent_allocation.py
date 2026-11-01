import pytest

pytest.importorskip("strands")

from gotrain.agent.fake_model import MultiTaskPolicy, ScriptedModel
from gotrain.agent.runner import run_agent
from gotrain.agent.session import AgentSession, session_dir, split_allowances
from gotrain.config import Budget, Limits
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl

THREE = ["synthetic", "synthetic-b", "synthetic-c"]


def _home(tmp_path) -> Home:
    return Home(tmp_path / "home").ensure()


def _budget(**kw) -> Budget:
    base = dict(agent_session_minutes=20, agent_max_experiments=6, agent_run_max_minutes=2)
    base.update(kw)
    return Budget(**base)


def _run(home, tasks, policy, budget, max_rounds=3):
    return run_agent(home, tasks, ScriptedModel(policy), budget, Limits.load(), device="cpu", source="synthetic", max_rounds=max_rounds, log=lambda s: None)


def test_split_allowances_even_with_remainders_and_small_caps():
    assert split_allowances(["a", "b", "c"], 6) == {"a": 2, "b": 2, "c": 2}
    assert split_allowances(["a", "b", "c"], 7) == {"a": 3, "b": 2, "c": 2}
    assert split_allowances(["a", "b", "c"], 2) == {"a": 1, "b": 1, "c": 0}
    assert split_allowances(["a"], 6) == {"a": 6}
    assert sum(split_allowances(["a", "b", "c", "d"], 10).values()) == 10


def test_greedy_first_task_cannot_consume_the_shared_budget(tmp_path):
    home = _home(tmp_path)
    state = _run(home, THREE, MultiTaskPolicy(mode="greedy"), _budget(agent_max_experiments=6))
    by_task = state["launched_by_task"]
    assert by_task["synthetic"] == 2 and by_task["synthetic-b"] == 2 and by_task["synthetic-c"] == 2
    assert state["experiments_launched"] == 6
    records = ExperimentStore(home).all_records()
    assert len(records) == 6 and {r["task"] for r in records} == set(THREE)
    assert state["status"] == "completed" and state["unfinished_tasks"] == []
    decisions = list(iter_jsonl(session_dir(home, state["session_id"]) / "decisions.jsonl"))
    blocks = [d for d in decisions if d["kind"] == "guard_block"]
    assert blocks and any("allowance for synthetic used up" in p for p in blocks[0]["problems"])


def test_single_task_keeps_full_budget(tmp_path):
    home = _home(tmp_path)
    state = _run(home, ["synthetic"], MultiTaskPolicy(mode="greedy"), _budget(agent_max_experiments=3))
    assert state["allowances"] == {"synthetic": 3}
    assert state["experiments_launched"] == 3 and state["launched_by_task"]["synthetic"] == 3
    assert state["status"] == "completed"


def test_insufficient_budget_reports_partial_coverage(tmp_path):
    home = _home(tmp_path)
    state = _run(home, THREE, MultiTaskPolicy(mode="one"), _budget(agent_max_experiments=2))
    assert state["allowances"] == {"synthetic": 1, "synthetic-b": 1, "synthetic-c": 0}
    assert state["experiments_launched"] == 2
    assert state["status"] == "partial"
    assert state["unfinished_tasks"] == ["synthetic-c"]
    assert set(state["finished_tasks"]) == {"synthetic", "synthetic-b"}
    assert state["coverage"]["requested"] == THREE and state["coverage"]["unfinished"] == ["synthetic-c"]


def test_unfinished_task_is_not_called_completed(tmp_path):
    home = _home(tmp_path)
    state = _run(home, ["synthetic", "synthetic-b"], MultiTaskPolicy(mode="launch_only"), _budget(agent_max_experiments=4), max_rounds=2)
    assert state["status"] == "partial"
    assert set(state["unfinished_tasks"]) == {"synthetic", "synthetic-b"}
    assert ExperimentStore(home).best("synthetic") is None or state["finished_tasks"] == {}


def test_resume_keeps_allowances_and_counts(tmp_path):
    home = _home(tmp_path)
    first = _run(home, THREE, MultiTaskPolicy(mode="one"), _budget(agent_max_experiments=2))
    assert first["status"] == "partial"
    resumed = run_agent(home, THREE, ScriptedModel(MultiTaskPolicy(mode="one")), _budget(agent_max_experiments=2), Limits.load(), device="cpu", source="synthetic", session_id=first["session_id"], max_rounds=2, log=lambda s: None)
    assert resumed["allowances"] == first["allowances"]
    assert resumed["launched_by_task"] == first["launched_by_task"]
    assert resumed["experiments_launched"] == 2
    assert resumed["status"] == "partial" and resumed["unfinished_tasks"] == ["synthetic-c"]
    assert resumed["resume_count"] == 1


def test_session_api_allowance_accounting(tmp_path):
    home = _home(tmp_path)
    s = AgentSession(home, _budget(agent_max_experiments=5), Limits.load(), tasks=["a", "b"])
    assert s.task_allowance("a") == 3 and s.task_allowance("b") == 2
    s.count_launch("a", 1)
    s.count_launch("a", 1)
    s.count_launch("a", 1)
    assert s.task_remaining("a") == 0 and s.task_remaining("b") == 2
    ok, reason = s.can_launch(1.0, task="a")
    assert not ok and "allowance for a used up" in reason
    s.count_launch("a", -1)
    assert s.task_remaining("a") == 1 and s.experiments_remaining() == 3
    s.mark_finished("a", {"selected": "x"})
    assert s.state["unfinished_tasks"] == ["b"]
    assert s.finalize()["status"] == "partial"
