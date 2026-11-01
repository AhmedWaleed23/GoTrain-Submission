"""Budget guard: a Strands hook that enforces limits outside the prompt.

Every tool call passes through ``before`` where the name, counters, elapsed time
and run_experiment arguments are checked against configs/limits.json and the
session budget. A blocked call is cancelled with an explanatory message that the
model sees, and the block is recorded in decisions.jsonl.
"""

from __future__ import annotations

from typing import Any

from gotrain.agent.session import AgentSession
from gotrain.config import Limits, RunConfig

try:
    from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry
except ImportError:  # strands missing: guard logic stays testable through check()
    HookProvider = object  # type: ignore[assignment,misc]
    BeforeToolCallEvent = AfterToolCallEvent = HookRegistry = None  # type: ignore[assignment,misc]


def validate_run_args(args: dict[str, Any], limits: Limits, session: AgentSession, task_cfg=None) -> list[str]:
    problems: list[str] = []
    required = {"task", "model", "lr", "batch_size", "epochs", "max_minutes", "hypothesis", "rationale", "expected_tradeoff"}
    missing = [k for k in required if k not in args or args[k] in (None, "")]
    if missing:
        problems.append(f"missing required arguments: {sorted(missing)}")
        return problems
    for key in ("hypothesis", "rationale", "expected_tradeoff"):
        if len(str(args[key]).strip()) < 12:
            problems.append(f"{key} must be a real sentence (got {str(args[key])[:30]!r})")
    if args["task"] not in session.state["tasks"]:
        problems.append(f"task {args['task']!r} is not part of this session {session.state['tasks']}")
    try:
        run = RunConfig(
            task=str(args["task"]),
            model=str(args["model"]),
            lr=float(args["lr"]),
            batch_size=int(args["batch_size"]),
            epochs=int(args["epochs"]),
            max_minutes=float(args["max_minutes"]),
            weight_decay=float(args.get("weight_decay", 1e-4)),
            seed=int(args.get("seed", 0)),
            augment=bool(args.get("augment", True)),
            subset_fraction=float(args.get("subset_fraction", 1.0)),
            screening=bool(args.get("screening", False)),
            input_size=int(args["input_size"]) if args.get("input_size") else None,
            val_limit=int(args["val_limit"]) if args.get("val_limit") else None,
        )
    except (TypeError, ValueError) as exc:
        problems.append(f"bad argument types: {exc}")
        return problems
    problems.extend(run.validate(limits, task_cfg))
    ok, reason = session.can_launch(run.max_minutes, task=run.task)
    if not ok:
        problems.append(reason)
    return problems


class BudgetGuard(HookProvider):
    def __init__(self, session: AgentSession, limits: Limits, allowed_tools: set[str], task_cfg_loader=None) -> None:
        self.session = session
        self.limits = limits
        self.allowed_tools = set(allowed_tools)
        self.task_cfg_loader = task_cfg_loader
        self.blocked = 0

    def register_hooks(self, registry: "HookRegistry", **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before)
        registry.add_callback(AfterToolCallEvent, self.after)

    def check(self, name: str, args: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        if name not in self.allowed_tools:
            return [f"tool {name!r} is not allowed in this session"]
        if self.session.tool_calls_remaining() <= 0:
            problems.append(f"tool call limit reached ({self.session.state['budget']['max_tool_calls']})")
        if self.session.remaining_minutes() <= 0 and name in ("run_experiment",):
            problems.append("session time budget exhausted")
        if name == "run_experiment":
            task_cfg = None
            if self.task_cfg_loader is not None and args.get("task"):
                try:
                    task_cfg = self.task_cfg_loader(args["task"])
                except Exception:
                    task_cfg = None
            problems.extend(validate_run_args(args, self.limits, self.session, task_cfg))
        return problems

    def before(self, event: "BeforeToolCallEvent") -> None:
        name = event.tool_use.get("name", "")
        args = dict(event.tool_use.get("input") or {})
        problems = self.check(name, args)
        if problems:
            self.blocked += 1
            message = "BLOCKED by budget guard: " + "; ".join(problems)
            self.session.record_decision("guard_block", tool=name, args=_compact(args), problems=problems)
            self.session.log(f"guard blocked {name}: {problems}")
            event.cancel_tool = message

    def after(self, event: "AfterToolCallEvent") -> None:
        name = event.tool_use.get("name", "")
        self.session.state["tool_calls"] += 1
        status = None
        try:
            status = event.result.get("status") if isinstance(event.result, dict) else None
        except Exception:
            status = None
        self.session.record_transcript({"kind": "tool_call", "tool": name, "args": _compact(dict(event.tool_use.get("input") or {})), "status": status, "duration_s": event.duration, "cancelled": bool(event.cancel_message)})
        self.session.save()


def _compact(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 300:
            v = v[:300] + "..."
        out[k] = v
    return out
