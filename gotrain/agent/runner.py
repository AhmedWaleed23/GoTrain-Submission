"""The autonomous loop: short rounds, compact state summaries, persisted decisions.

Each round builds a fresh Strands Agent with the same tools and guard, hands it a
compact JSON summary of everything on disk (dataset facts, budget, ranked
experiments, failures, the agent's own earlier conclusions) and lets it act until it
stops. The loop ends when the agent calls finish_task, when the budget is gone, or
when two rounds pass without a tool call.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from gotrain.agent.baseline import baseline_record
from gotrain.agent.guard import BudgetGuard
from gotrain.agent.prompts import ROUND_PROMPT, SYSTEM_PROMPT
from gotrain.agent.session import AgentSession
from gotrain.agent.tools import ToolRuntime, build_tools, tool_names
from gotrain.config import Budget, Limits, load_task_config
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import dumps_json


def build_task_summary(rt: ToolRuntime, task: str, store: ExperimentStore, last_conclusions: list[str]) -> dict[str, Any]:
    session = rt.session
    ranked = [store.compact_summary(r) for r in store.ranking(task, source=rt.source)]
    all_rows = [store.compact_summary(r) for r in store.list(task=task, source=rt.source)]
    ranked_ids = {r["experiment_id"] for r in ranked}
    screening = sorted([r for r in all_rows if r["screening"] and r["status"] == "completed"], key=lambda r: -(r["best_val_metric"] or 0))
    failed = [r for r in all_rows if r["status"] in ("failed", "interrupted")]
    base = baseline_record(store, task)
    best = store.best(task)
    small = lambda r: {k: r[k] for k in ("experiment_id", "model", "lr", "batch_size", "epochs_completed", "subset_fraction", "input_size", "best_val_metric", "stop_reason", "wall_minutes", "adjustments", "error", "hypothesis") if k in r}
    return {
        "task": task,
        "selection_metric": ranked[0]["metric"] if ranked else (base["metric"]["name"] if base else None),
        "budget": {
            "remaining_minutes": round(session.remaining_minutes(), 1),
            "experiments_remaining": session.experiments_remaining(),
            "this_task_experiment_allowance": session.task_allowance(task),
            "this_task_experiments_launched": session.task_launched(task),
            "this_task_experiments_remaining": session.task_remaining(task),
            "launches_allowed_for_this_task": session.task_remaining(task) > 0,
            "tool_calls_remaining": session.tool_calls_remaining(),
            "per_run_max_minutes": rt.budget.agent_run_max_minutes,
            "finalist_budget": {"epochs": rt.budget.finalist_epochs, "max_minutes": rt.budget.finalist_max_minutes},
        },
        "baseline": small(store.compact_summary(base)) if base else None,
        "ranked_full_runs": [small(r) for r in ranked[:8]],
        "screening_runs": [small(r) for r in screening[:6]],
        "failed_or_interrupted": [small(r) for r in failed[:4]],
        "best_pointer": {"experiment_id": best["experiment_id"], "best_val_metric": best["best_val_metric"], "selected_by": best["selected_by"]} if best else None,
        "your_earlier_conclusions": last_conclusions[-3:],
        "finished": task in session.state["finished_tasks"],
    }


def _result_text(result: Any) -> str:
    try:
        parts = []
        for block in result.message.get("content", []):
            if "text" in block:
                parts.append(block["text"])
        return "\n".join(parts).strip()
    except Exception:
        return str(result)[:2000]


def _supports_limits() -> bool:
    try:
        from strands import Agent

        return "limits" in inspect.signature(Agent.__call__).parameters
    except Exception:
        return False


def reconcile_finished_selection(session, store, task: str) -> None:
    """Reconcile after all round tool calls settle; never change a model pointer."""
    finished = session.state.get("finished_tasks", {}).get(task)
    if finished is None:
        return
    selected = store.best(task)
    if selected is None:
        return
    if (finished.get("selected"), finished.get("best_val_metric")) == (selected["experiment_id"], selected["best_val_metric"]):
        return
    previous = finished.get("selected")
    finished.update(selected=selected["experiment_id"], best_val_metric=selected["best_val_metric"])
    session.record_decision("finish_selection_reconciled", task=task, previously_reported=previous, selected=selected["experiment_id"], best_val_metric=selected["best_val_metric"])
    session.save()


def run_agent(
    home: Home,
    tasks: list[str],
    model: Any,
    budget: Budget,
    limits: Limits,
    device: str | None = None,
    geobench_root: Path | None = None,
    source: str = "geobench",
    session_id: str | None = None,
    max_rounds: int = 8,
    turns_per_round: int = 16,
    log: Callable[[str], None] = print,
    registry: Any = None,
    model_id: str = "",
) -> dict[str, Any]:
    from strands import Agent
    from strands.agent.conversation_manager import SlidingWindowConversationManager

    if source == "synthetic":
        limits = Limits(**{**limits.__dict__, "allowed_tasks": list(tasks)})
    session = AgentSession(home, budget, limits, tasks=tasks, session_id=session_id, source=source, model_id=model_id)
    store = ExperimentStore(home)
    rt = ToolRuntime(home=home, session=session, limits=limits, budget=budget, device=device, geobench_root=geobench_root, source=source, log=log, registry=registry)
    tools = build_tools(rt)
    guard = BudgetGuard(session, limits, tool_names(tools), task_cfg_loader=None if source == "synthetic" else load_task_config)
    supports_limits = _supports_limits()
    session.record_decision("session_start", tasks=tasks, budget=session.state["budget"], allowances=session.state.get("allowances"), source=source, model_id=model_id)
    log(f"[agent] session {session.session_id}: tasks {tasks}, {session.remaining_minutes():.0f} min, {session.experiments_remaining()} experiments, allowances {session.state.get('allowances')}")

    for task in session.state["tasks"]:
        if task in session.state["finished_tasks"]:
            log(f"[agent] {task} already finished in this session; skipping")
            continue
        conclusions: list[str] = []
        idle_rounds = 0
        errors = 0
        for round_index in range(1, max_rounds + 1):
            if session.remaining_minutes() <= 0:
                session.record_decision("budget_exhausted", task=task, round=round_index)
                log(f"[agent] {task}: session budget exhausted before round {round_index}")
                break
            if session.tool_calls_remaining() <= 0:
                session.record_decision("tool_call_limit", task=task, round=round_index)
                break
            summary = build_task_summary(rt, task, store, conclusions)
            prompt = ROUND_PROMPT.format(task=task, round_index=round_index, max_rounds=max_rounds, summary=dumps_json(summary))
            if session.task_remaining(task) <= 0:
                prompt += "\nNo more experiments may be launched for this task (its allowance is used). Review the results, call select_best_model if a completed full run exists, then finish_task.\n"
                if round_index > 2 and task not in session.state["finished_tasks"]:
                    session.record_decision("allowance_exhausted_stop", task=task, round=round_index)
                    log(f"[agent] {task}: allowance used and no finish after {round_index - 1} wrap-up rounds; moving on")
                    break
            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                hooks=[guard],
                callback_handler=None,
                conversation_manager=SlidingWindowConversationManager(window_size=24),
            )
            calls_before = session.state["tool_calls"]
            session.state["rounds"] += 1
            session.record_decision("round_start", task=task, round=round_index, remaining_minutes=round(session.remaining_minutes(), 1))
            try:
                result = agent(prompt, limits={"turns": turns_per_round}) if supports_limits else agent(prompt)
            except Exception as exc:
                errors += 1
                session.state["errors"].append(f"round {round_index} on {task}: {type(exc).__name__}: {exc}"[:500])
                session.record_decision("round_error", task=task, round=round_index, error=f"{type(exc).__name__}: {exc}"[:500])
                session.save()
                log(f"[agent] {task} round {round_index} failed: {exc}")
                if errors >= 2:
                    break
                continue
            text = _result_text(result)
            stop_reason = getattr(result, "stop_reason", None)
            usage = {}
            try:
                usage = dict(result.metrics.accumulated_usage)
            except Exception:
                usage = {}
            session.record_transcript({"kind": "round", "task": task, "round": round_index, "stop_reason": stop_reason, "usage": usage, "text": text[:4000], "messages": _compact_messages(agent.messages)})
            session.record_decision("round_end", task=task, round=round_index, stop_reason=stop_reason, tool_calls=session.state["tool_calls"] - calls_before, text=text[:2000])
            if text:
                conclusions.append(text[:600])
            session.save()
            if task in session.state["finished_tasks"]:
                reconcile_finished_selection(session, store, task)
                log(f"[agent] {task} finished: {session.state['finished_tasks'][task]['selected']}")
                break
            if session.state["tool_calls"] == calls_before:
                idle_rounds += 1
                if idle_rounds >= 2:
                    session.record_decision("idle_stop", task=task, round=round_index)
                    log(f"[agent] {task}: two rounds without tool calls; stopping")
                    break
            else:
                idle_rounds = 0
        if task not in session.state["finished_tasks"]:
            best = store.best(task)
            session.record_decision("task_unfinished", task=task, best_pointer=best["experiment_id"] if best else None, remaining_minutes=round(session.remaining_minutes(), 1))
            log(f"[agent] {task}: left unfinished by the agent; best pointer = {best['experiment_id'] if best else None}")
    state = session.finalize(guard_blocks=guard.blocked)
    if state["status"] != "completed":
        log(f"[agent] session {session.session_id} ended PARTIAL: unfinished tasks {state['unfinished_tasks']}")
    return state


def _compact_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.get("role")}
        parts = []
        for block in m.get("content", []):
            if "text" in block:
                parts.append({"text": block["text"][:1500]})
            elif "toolUse" in block:
                parts.append({"toolUse": {"name": block["toolUse"]["name"], "input": block["toolUse"].get("input")}})
            elif "toolResult" in block:
                txt = "".join(c.get("text", "") for c in block["toolResult"].get("content", []))
                parts.append({"toolResult": {"status": block["toolResult"].get("status"), "text": txt[:1500]}})
        entry["content"] = parts
        out.append(entry)
    return out
