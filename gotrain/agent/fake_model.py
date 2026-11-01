"""A scripted Strands model for offline orchestration tests.

It implements the Model interface and turns a policy's decisions into the same
stream events a Bedrock model would emit. The reactive policy below reads the tool
results it receives and chooses the next tool call from them, which exercises the
tools, the guard and the session loop without any LLM. It is a test double: it
proves the plumbing, not the quality of Sonnet's decisions.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Callable

from strands.models.model import Model


def _tool_results(messages: list[dict]) -> list[tuple[str, Any]]:
    """(tool_name, parsed_result) pairs in order, pairing toolUse ids with toolResult blocks."""
    uses: dict[str, str] = {}
    out: list[tuple[str, Any]] = []
    for m in messages:
        for block in m.get("content", []):
            if "toolUse" in block:
                uses[block["toolUse"]["toolUseId"]] = block["toolUse"]["name"]
            if "toolResult" in block:
                tr = block["toolResult"]
                name = uses.get(tr["toolUseId"], "?")
                text = ""
                for c in tr.get("content", []):
                    if "text" in c:
                        text += c["text"]
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = text
                out.append((name, parsed))
    return out


class ScriptedModel(Model):
    def __init__(self, policy: Callable[[list[dict], list], tuple[str, Any]]) -> None:
        self.policy = policy
        self.calls = 0
        self.config: dict[str, Any] = {"model_id": "scripted-fake"}

    def get_config(self) -> dict[str, Any]:
        return self.config

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError("ScriptedModel has no structured output")
        yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncGenerator[dict, None]:
        self.calls += 1
        kind, payload = self.policy(messages, tool_specs or [])
        yield {"messageStart": {"role": "assistant"}}
        if kind == "tool":
            name, args = payload
            tool_id = f"tooluse_{uuid.uuid4().hex[:12]}"
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_id, "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": str(payload)}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 1}}}


class ReactivePolicy:
    """Explores one task: inspect, budget, baseline-like run, one follow-up chosen from the
    first result, selection of the better of the two, finish. Each step is derived from the
    tool results visible in the conversation."""

    def __init__(self, task: str, model: str, epochs: int = 2, max_minutes: float = 2.0, batch_size: int = 16) -> None:
        self.task = task
        self.model = model
        self.epochs = epochs
        self.max_minutes = max_minutes
        self.batch_size = batch_size
        self.history: list[tuple[str, Any]] = []

    def __call__(self, messages: list[dict], tool_specs: list) -> tuple[str, Any]:
        results = _tool_results(messages)
        self.history = results
        seen = [name for name, _ in results]
        if results and isinstance(results[-1][1], str) and ("BLOCKED" in results[-1][1] or "Error" in results[-1][1]):
            return "text", f"Stopping after a blocked or failed call: {results[-1][1][:200]}"
        runs = [(n, r) for n, r in results if n == "run_experiment" and isinstance(r, dict)]
        if "inspect_dataset" not in seen:
            return "tool", ("inspect_dataset", {"task": self.task})
        if "get_budget_status" not in seen:
            return "tool", ("get_budget_status", {})
        if len(runs) == 0:
            return "tool", ("run_experiment", self._run(1e-3, "Reference run at lr 1e-3 should converge within the epoch budget.", "No experiments observed yet for this task; dataset inspection shows balanced classes.", "Costs one short run; may underfit."))
        first = runs[0][1]
        if len(runs) == 1:
            if first.get("status") == "rejected":
                return "text", "The run was rejected; stopping."
            score = first.get("best_val_metric")
            return "tool", ("run_experiment", self._run(3e-3, "A 3x higher learning rate should reach a better score in the same epochs.", f"The first run reached {score} with stop reason {first.get('stop_reason')}; the loss was still decreasing so faster learning may help.", "Risk of instability; same compute cost."))
        second = runs[1][1]
        if "select_best_model" not in seen:
            cands = [r for r in (first, second) if r.get("status") == "completed" and not r.get("screening")]
            best = max(cands, key=lambda r: r.get("best_val_metric") or -1)
            other = [r for r in cands if r is not best]
            rationale = f"{best['experiment_id']} scored {best['best_val_metric']}" + (f" versus {other[0]['best_val_metric']} for {other[0]['experiment_id']}" if other else "")
            return "tool", ("select_best_model", {"task": self.task, "experiment_id": best["experiment_id"], "rationale": rationale})
        if "finish_task" not in seen:
            return "tool", ("finish_task", {"task": self.task, "summary": "Two learning rates were compared at equal budget and the better validation score was selected. No failures occurred. Budget remained for more runs but the scripted policy stops here."})
        return "text", "Done."

    def _run(self, lr: float, hypothesis: str, rationale: str, tradeoff: str) -> dict:
        return {"task": self.task, "model": self.model, "lr": lr, "batch_size": self.batch_size, "epochs": self.epochs, "max_minutes": self.max_minutes, "hypothesis": hypothesis, "rationale": rationale, "expected_tradeoff": tradeoff}


class OverBudgetPolicy:
    """Keeps asking for runs until the guard blocks it, then finishes."""

    def __init__(self, task: str, model: str) -> None:
        self.task = task
        self.model = model

    def __call__(self, messages: list[dict], tool_specs: list) -> tuple[str, Any]:
        results = _tool_results(messages)
        blocked = [r for n, r in results if n == "run_experiment" and isinstance(r, str) and "BLOCKED" in r]
        if blocked:
            return "text", "Blocked by the guard; stopping."
        return "tool", ("run_experiment", {"task": self.task, "model": self.model, "lr": 1e-3, "batch_size": 16, "epochs": 1, "max_minutes": 1.0, "seed": len(results), "hypothesis": "Another run should still fit in budget.", "rationale": "Previous runs completed; probing the budget guard.", "expected_tradeoff": "Costs one short run."})


class BadArgsPolicy:
    def __init__(self, task: str) -> None:
        self.task = task

    def __call__(self, messages: list[dict], tool_specs: list) -> tuple[str, Any]:
        results = _tool_results(messages)
        if results:
            return "text", "Rejected as expected."
        return "tool", ("run_experiment", {"task": self.task, "model": "cnn_small", "lr": 50.0, "batch_size": 16, "epochs": 1, "max_minutes": 1.0, "hypothesis": "x", "rationale": "y", "expected_tradeoff": "z"})


def _current_task(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") != "user":
            continue
        for block in m.get("content", []):
            text = block.get("text", "")
            if text.startswith("Current task: "):
                return text.split("\n", 1)[0].replace("Current task: ", "").strip()
    return None


class MultiTaskPolicy:
    """Per-task scripted behaviour for multi-task session tests.

    mode="greedy": keeps launching new configs for the current task until the guard
    blocks it, then selects and finishes. mode="one": launches once per task, then
    selects and finishes. mode="launch_only": launches once and never finishes.
    """

    def __init__(self, model: str = "cnn_small", mode: str = "greedy", epochs: int = 1, max_minutes: float = 1.0) -> None:
        self.model = model
        self.mode = mode
        self.epochs = epochs
        self.max_minutes = max_minutes

    def __call__(self, messages: list[dict], tool_specs: list) -> tuple[str, Any]:
        task = _current_task(messages) or "synthetic"
        results = _tool_results(messages)
        seen = [n for n, _ in results]
        runs = [r for n, r in results if n == "run_experiment"]
        blocked = any(isinstance(r, str) and "BLOCKED" in r for r in runs)
        completed = [r for r in runs if isinstance(r, dict) and r.get("status") == "completed"]
        if self.mode == "launch_only":
            if runs:
                return "text", "Launched once; not finishing on purpose."
            return "tool", ("run_experiment", self._run(task, seed=0))
        want_more = (self.mode == "greedy" and not blocked) or (self.mode == "one" and not runs)
        if want_more:
            return "tool", ("run_experiment", self._run(task, seed=len(runs)))
        if "select_best_model" not in seen:
            if not completed:
                return "text", "Nothing completed for this task."
            best = max(completed, key=lambda r: r.get("best_val_metric") or -1)
            return "tool", ("select_best_model", {"task": task, "experiment_id": best["experiment_id"], "rationale": f"best of {len(completed)} completed runs with score {best.get('best_val_metric')}"})
        if "finish_task" not in seen:
            return "tool", ("finish_task", {"task": task, "summary": f"Scripted multi-task policy finished {task} after {len(completed)} completed runs; the guard blocked further launches: {blocked}."})
        return "text", "Done with this task."

    def _run(self, task: str, seed: int) -> dict:
        return {"task": task, "model": self.model, "lr": 1e-3, "batch_size": 16, "epochs": self.epochs, "max_minutes": self.max_minutes, "seed": seed, "hypothesis": "Another short run should still be within this task's allowance.", "rationale": "Scripted policy probing the per-task allowance in a multi-task session.", "expected_tradeoff": "Costs one short run of budget."}
