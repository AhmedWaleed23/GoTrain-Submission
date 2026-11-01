"""Persisted agent session state: budgets, counters, decisions and transcript."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from gotrain.config import Budget, Limits
from gotrain.paths import Home
from gotrain.util.ids import new_id, validate_id
from gotrain.util.io import append_jsonl, iter_jsonl, read_json, write_json
from gotrain.util.timing import utc_now_iso

LAUNCH_MARGIN_MINUTES = 2.0


def split_allowances(tasks: list[str], total: int) -> dict[str, int]:
    """Even split of the global experiment cap; integer remainders go to earlier tasks.
    Never exceeds the cap and never inflates a cap smaller than the task count."""
    tasks = list(tasks)
    if not tasks:
        return {}
    total = max(0, int(total))
    base, rem = divmod(total, len(tasks))
    return {t: base + (1 if i < rem else 0) for i, t in enumerate(tasks)}


class AgentSession:
    def __init__(self, home: Home, budget: Budget, limits: Limits, tasks: list[str] | None = None, session_id: str | None = None, source: str = "geobench", model_id: str = "") -> None:
        self.home = home.ensure()
        self.budget = budget
        self.limits = limits
        if session_id:
            validate_id(session_id, "session id")
            self.dir = home.agent / "sessions" / session_id
            if not (self.dir / "state.json").exists():
                raise FileNotFoundError(f"no agent session {session_id}")
            self.state = read_json(self.dir / "state.json")
            self.state["elapsed_before_seconds"] = float(self.state.get("elapsed_seconds", self.state.get("elapsed_before_seconds", 0.0)))
            self.state["resume_count"] = self.state.get("resume_count", 0) + 1
            self.state["status"] = "running"
            self.state["budget"]["session_minutes"] = min(budget.agent_session_minutes, limits.max_session_minutes)
            self.state.setdefault("allowances", split_allowances(self.state.get("tasks", []), int(self.state["budget"]["max_experiments"])))
            self.state.setdefault("launched_by_task", {t: 0 for t in self.state.get("tasks", [])})
            self.state.setdefault("unfinished_tasks", [t for t in self.state.get("tasks", []) if t not in self.state.get("finished_tasks", {})])
            self.record_decision("session_resumed", resume_count=self.state["resume_count"], elapsed_minutes=round(self.state["elapsed_before_seconds"] / 60.0, 1), session_minutes=self.state["budget"]["session_minutes"])
        else:
            sid = new_id("agent")
            self.dir = home.agent / "sessions" / sid
            self.dir.mkdir(parents=True, exist_ok=False)
            self.state = {
                "session_id": sid,
                "tasks": list(tasks or []),
                "source": source,
                "model_id": model_id,
                "status": "running",
                "started_at": utc_now_iso(),
                "tool_calls": 0,
                "experiments_launched": 0,
                "rounds": 0,
                "elapsed_before_seconds": 0.0,
                "finished_tasks": {},
                "budget": {"session_minutes": budget.agent_session_minutes, "max_experiments": min(budget.agent_max_experiments, limits.max_experiments_per_session), "run_max_minutes": budget.agent_run_max_minutes, "max_tool_calls": limits.max_tool_calls_per_session},
                "allowances": split_allowances(list(tasks or []), min(budget.agent_max_experiments, limits.max_experiments_per_session)),
                "launched_by_task": {t: 0 for t in (tasks or [])},
                "unfinished_tasks": list(tasks or []),
                "errors": [],
                "resume_count": 0,
            }
        self._start = time.time()
        self.save()

    @property
    def session_id(self) -> str:
        return self.state["session_id"]

    def save(self) -> None:
        self.state["elapsed_seconds"] = self.elapsed_seconds()
        self.state["updated_at"] = utc_now_iso()
        write_json(self.dir / "state.json", self.state)

    def elapsed_seconds(self) -> float:
        return float(self.state.get("elapsed_before_seconds", 0.0)) + (time.time() - self._start)

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0

    def remaining_minutes(self) -> float:
        cap = min(self.budget.agent_session_minutes, self.limits.max_session_minutes)
        return cap - self.elapsed_minutes()

    def experiments_remaining(self) -> int:
        return int(self.state["budget"]["max_experiments"]) - int(self.state["experiments_launched"])

    def tool_calls_remaining(self) -> int:
        return int(self.state["budget"]["max_tool_calls"]) - int(self.state["tool_calls"])

    def task_allowance(self, task: str) -> int:
        return int(self.state.get("allowances", {}).get(task, self.state["budget"]["max_experiments"]))

    def task_launched(self, task: str) -> int:
        return int(self.state.get("launched_by_task", {}).get(task, 0))

    def task_remaining(self, task: str) -> int:
        return max(0, min(self.task_allowance(task) - self.task_launched(task), self.experiments_remaining()))

    def count_launch(self, task: str, delta: int = 1) -> None:
        self.state["experiments_launched"] = int(self.state["experiments_launched"]) + delta
        by_task = self.state.setdefault("launched_by_task", {})
        by_task[task] = int(by_task.get(task, 0)) + delta

    def mark_finished(self, task: str, info: dict) -> None:
        self.state["finished_tasks"][task] = info
        self.state["unfinished_tasks"] = [t for t in self.state["tasks"] if t not in self.state["finished_tasks"]]

    def finalize(self, guard_blocks: int = 0) -> dict:
        """Set a truthful end status: completed only when every requested task finished."""
        unfinished = [t for t in self.state["tasks"] if t not in self.state["finished_tasks"]]
        self.state["unfinished_tasks"] = unfinished
        self.state["status"] = "completed" if not unfinished else "partial"
        self.state["coverage"] = {"requested": list(self.state["tasks"]), "finished": list(self.state["finished_tasks"]), "unfinished": unfinished, "launched_by_task": dict(self.state.get("launched_by_task", {})), "allowances": dict(self.state.get("allowances", {}))}
        self.record_decision("session_end", status=self.state["status"], finished_tasks=list(self.state["finished_tasks"]), unfinished_tasks=unfinished, experiments_launched=self.state["experiments_launched"], launched_by_task=self.state.get("launched_by_task", {}), tool_calls=self.state["tool_calls"], elapsed_minutes=round(self.elapsed_minutes(), 1), guard_blocks=guard_blocks)
        self.save()
        return self.state

    def can_launch(self, max_minutes: float, task: str | None = None) -> tuple[bool, str]:
        if self.experiments_remaining() <= 0:
            return False, f"experiment limit reached ({self.state['budget']['max_experiments']})"
        if task is not None and task in self.state.get("allowances", {}) and self.task_remaining(task) <= 0:
            return False, f"experiment allowance for {task} used up ({self.task_launched(task)} of {self.task_allowance(task)}); use list_experiments, select_best_model and finish_task for this task"
        remaining = self.remaining_minutes()
        if max_minutes + LAUNCH_MARGIN_MINUTES > remaining:
            return False, f"requested {max_minutes:.1f} min plus a {LAUNCH_MARGIN_MINUTES:.0f} min overhead margin but only {remaining:.1f} min of session budget remain"
        if max_minutes > self.budget.agent_run_max_minutes:
            return False, f"max_minutes {max_minutes} exceeds the per-run cap {self.budget.agent_run_max_minutes}"
        return True, "ok"

    def record_decision(self, kind: str, **data: Any) -> None:
        append_jsonl(self.dir / "decisions.jsonl", {"at": utc_now_iso(), "session_id": self.session_id, "kind": kind, **data})

    def record_transcript(self, entry: dict[str, Any]) -> None:
        append_jsonl(self.dir / "transcript.jsonl", {"at": utc_now_iso(), **entry})

    def decisions(self) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.dir / "decisions.jsonl"))

    def log(self, msg: str) -> None:
        with open(self.dir / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(f"[{utc_now_iso()}] {msg}\n")


def list_sessions(home: Home) -> list[dict[str, Any]]:
    root = home.agent / "sessions"
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if (d / "state.json").exists():
                try:
                    out.append(read_json(d / "state.json"))
                except ValueError:
                    continue
    return out


def session_dir(home: Home, session_id: str) -> Path:
    validate_id(session_id, "session id")
    return home.agent / "sessions" / session_id
