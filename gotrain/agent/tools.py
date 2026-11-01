"""Typed, bounded Strands tools. Each tool validates its own arguments in Python,
returns compact JSON text, and records what it did in the session. None of them can
reach the test split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from gotrain.agent.baseline import baseline_record
from gotrain.agent.guard import validate_run_args
from gotrain.agent.session import AgentSession
from gotrain.config import Budget, Limits, RunConfig, load_task_config
from gotrain.experiments.runner import run_experiment as launch_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.models.factory import MODEL_SPECS, models_for
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl, read_json, write_json

try:
    from strands import tool
except ImportError:  # pragma: no cover - strands absent locally
    def tool(fn=None, **_kw):  # type: ignore[misc]
        return fn if fn is not None else (lambda f: f)


@dataclass
class ToolRuntime:
    home: Home
    session: AgentSession
    limits: Limits
    budget: Budget
    device: str | None = None
    geobench_root: Path | None = None
    source: str = "geobench"
    log: Callable[[str], None] = print
    registry: Any = None


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _task_cfg(rt: ToolRuntime, task: str):
    return None if rt.source == "synthetic" else load_task_config(task)


def _speed_estimates(store: ExperimentStore, task: str) -> dict[str, Any]:
    """Seconds per full-data epoch per model, from completed runs' curves."""
    out: dict[str, Any] = {}
    for rec in store.list(task=task, statuses=("completed",)):
        model = rec["run_config"]["model"]
        frac = float(rec["run_config"].get("subset_fraction", 1.0) or 1.0)
        rows = list(iter_jsonl(store.exp_dir(rec["experiment_id"]) / "checkpoints" / "curves.jsonl"))
        if not rows:
            continue
        secs = [float(r.get("train_seconds", 0)) + float(r.get("val_seconds", 0)) for r in rows]
        per_epoch = sum(secs) / len(secs)
        if rec["run_config"].get("val_limit"):
            per_epoch_full = per_epoch / max(frac, 1e-6)
        else:
            per_epoch_full = (sum(float(r.get("train_seconds", 0)) for r in rows) / len(rows)) / max(frac, 1e-6) + sum(float(r.get("val_seconds", 0)) for r in rows) / len(rows)
        key = f"{model}@{rec['run_config'].get('input_size') or 'native'}"
        cur = out.get(key)
        if cur is None or rec["experiment_id"] > cur["from_experiment"]:
            out[key] = {"seconds_per_full_epoch_estimate": round(per_epoch_full, 1), "measured_on_subset_fraction": frac, "from_experiment": rec["experiment_id"], "device": rec.get("device")}
    return out


def build_tools(rt: ToolRuntime) -> list:
    store = ExperimentStore(rt.home)
    session = rt.session

    @tool
    def inspect_dataset(task: str) -> str:
        """Describe a task's dataset: sizes of train/valid splits, classes and their frequencies, channels, image size, value ranges. Uses train and validation data only.

        Args:
            task: task name, e.g. m-eurosat
        """
        if task not in session.state["tasks"]:
            return _json({"error": f"task {task!r} is not in this session"})
        if rt.source == "synthetic":
            from gotrain.data.synthetic import synthetic_splits

            tr, va = synthetic_splits("classification")
            return _json({"task": task, "source": "synthetic", "meta": tr.meta.to_dict(), "note": "synthetic fixture data"})
        cache = rt.home.root / "inspect" / f"{task}.json"
        if cache.exists():
            rep = read_json(cache)
        else:
            from gotrain.data.geobench_adapter import inspect_task

            rep = inspect_task(load_task_config(task), rt.geobench_root or _default_root(), max_label_samples=150)
            write_json(cache, rep)
        if not rep.get("present"):
            return _json({"error": f"dataset {task} missing", "hint": rep.get("hint")})
        meta = rep["meta"]
        compact = {
            "task": task,
            "task_type": meta["task_type"],
            "n_classes": meta["n_classes"],
            "class_names": meta["class_names"],
            "in_channels": meta["in_channels"],
            "image_size": [meta["height"], meta["width"]],
            "split_sizes": {k: v for k, v in rep["split_sizes"].items() if k != "test"},
            "train_label_fractions": {k: round(v, 3) for k, v in rep["splits"]["train"]["label_fractions"].items()},
            "valid_label_fractions": {k: round(v, 3) for k, v in rep["splits"]["valid"]["label_fractions"].items()},
            "invalid_labels": {s: rep["splits"][s]["invalid_labels"] for s in ("train", "valid")},
            "value_range": [round(min(rep["train_value_ranges"]["min_per_band"]), 1), round(max(rep["train_value_ranges"]["max_per_band"]), 1)],
            "warnings": rep.get("warnings", []),
            "band_order_note": meta.get("extra", {}).get("band_order_note", ""),
            "selection_metric": load_task_config(task).selection_metric,
        }
        session.record_transcript({"kind": "inspect", "task": task})
        return _json(compact)

    @tool
    def list_experiments(task: str, include_screening: bool = True) -> str:
        """List experiments for a task: ranked completed full runs, screening runs, failures, the baseline and the current best pointer.

        Args:
            task: task name
            include_screening: include partial-data screening runs in a separate list
        """
        if task not in session.state["tasks"]:
            return _json({"error": f"task {task!r} is not in this session"})
        ranked = [store.compact_summary(r) for r in store.ranking(task, source=rt.source)]
        others = [store.compact_summary(r) for r in store.list(task=task, source=rt.source) if r["experiment_id"] not in {x["experiment_id"] for x in ranked}]
        screening = [o for o in others if o["screening"] and o["status"] == "completed"]
        failed = [o for o in others if o["status"] in ("failed", "interrupted")]
        base = baseline_record(store, task)
        best = store.best(task)
        return _json(
            {
                "task": task,
                "metric": ranked[0]["metric"] if ranked else None,
                "ranked_full_runs": ranked[:10],
                "screening_runs": (sorted(screening, key=lambda s: -(s["best_val_metric"] or 0))[:10] if include_screening else "hidden"),
                "failed_or_interrupted": failed[:5],
                "baseline": store.compact_summary(base) if base else None,
                "best_pointer": {"experiment_id": best["experiment_id"], "best_val_metric": best["best_val_metric"], "selected_by": best["selected_by"]} if best else None,
            }
        )

    @tool
    def get_budget_status() -> str:
        """Remaining session time, experiments and tool calls, the hard limits, and measured training speed per model so runs can be sized to the budget."""
        speeds = {t: _speed_estimates(store, t) for t in session.state["tasks"]}
        return _json(
            {
                "elapsed_minutes": round(session.elapsed_minutes(), 1),
                "remaining_minutes": round(session.remaining_minutes(), 1),
                "experiments_remaining": session.experiments_remaining(),
                "per_task": {t: {"allowance": session.task_allowance(t), "launched": session.task_launched(t), "remaining": session.task_remaining(t), "finished": t in session.state["finished_tasks"]} for t in session.state["tasks"]},
                "tool_calls_remaining": session.tool_calls_remaining(),
                "per_run_max_minutes": rt.budget.agent_run_max_minutes,
                "finalist_budget": {"epochs": rt.budget.finalist_epochs, "max_minutes": rt.budget.finalist_max_minutes},
                "screening_defaults": {"subset_fraction": rt.budget.screening_subset_fraction, "epochs": rt.budget.screening_epochs},
                "limits": {"lr": [rt.limits.lr_min, rt.limits.lr_max], "batch_size": [rt.limits.batch_size_min, rt.limits.batch_size_max], "epochs": [rt.limits.epochs_min, rt.limits.epochs_max], "input_sizes": rt.limits.allowed_input_sizes},
                "models": {m: MODEL_SPECS[m]["description"] for m in rt.limits.allowed_models},
                "measured_speed": speeds,
                "device": rt.device or "auto",
            }
        )

    @tool
    def run_experiment(
        task: str,
        model: str,
        lr: float,
        batch_size: int,
        epochs: int,
        max_minutes: float,
        hypothesis: str,
        rationale: str,
        expected_tradeoff: str,
        screening: bool = False,
        subset_fraction: float = 1.0,
        weight_decay: float = 0.0001,
        augment: bool = True,
        input_size: Optional[int] = None,
        seed: int = 0,
        parent_experiment: Optional[str] = None,
    ) -> str:
        """Train and validate one configuration (blocking; returns when training ends). Records hypothesis, rationale and expected tradeoff before the run and the measured result after it.

        Args:
            task: task name
            model: one of the allowed models for the task type
            lr: learning rate (AdamW, cosine schedule)
            batch_size: training batch size
            epochs: maximum epochs (early stopping may end sooner)
            max_minutes: wall-clock cap for this run
            hypothesis: what you expect and why (one or two sentences)
            rationale: which observed results motivate this run
            expected_tradeoff: what this costs or risks (time, memory, overfitting)
            screening: true for a cheap partial-data screening run (never ranked with full runs)
            subset_fraction: fraction of the training split to use (< 1 requires screening=true)
            weight_decay: AdamW weight decay
            augment: flips and 90-degree rotations on training images (and masks)
            input_size: optional resize of images (and masks, nearest) to this square size
            seed: random seed
            parent_experiment: experiment id this run builds on, if any
        """
        args = {k: v for k, v in locals().items() if k not in ("args",)}
        problems = validate_run_args(args, rt.limits, session, _task_cfg(rt, task))
        if problems:
            session.record_decision("rejected_experiment", task=task, problems=problems, args={k: args[k] for k in ("model", "lr", "batch_size", "epochs", "max_minutes")})
            return _json({"status": "rejected", "problems": problems})
        run = RunConfig(
            task=task, model=model, lr=float(lr), batch_size=int(batch_size), epochs=int(epochs), max_minutes=float(max_minutes),
            weight_decay=float(weight_decay), seed=int(seed), augment=bool(augment), subset_fraction=float(subset_fraction), screening=bool(screening),
            input_size=int(input_size) if input_size else None, num_workers=2 if rt.source != "synthetic" else 0, tag="agent",
            hypothesis=hypothesis.strip(), rationale=rationale.strip(), expected_tradeoff=expected_tradeoff.strip(), parent_experiment=parent_experiment,
            early_stopping_patience=max(2, min(5, int(epochs) // 2)) if int(epochs) > 2 else 2,
        )
        session.count_launch(task, 1)
        session.record_decision("propose_experiment", task=task, config=run.to_dict())
        session.save()
        rt.log(f"[agent] launching {model} lr={lr} bs={batch_size} epochs={epochs} max_minutes={max_minutes} screening={screening}: {hypothesis[:120]}")
        try:
            rec = launch_experiment(rt.home, run, source=rt.source, device=rt.device, log=rt.log, limits=rt.limits, geobench_root=rt.geobench_root)
        except Exception as exc:
            session.count_launch(task, -1)
            session.record_decision("experiment_result", task=task, experiment_id=None, status="not_started", error=f"{type(exc).__name__}: {exc}"[:500])
            session.save()
            return _json({"status": "not_started", "error": f"{type(exc).__name__}: {exc}"[:500], "hint": "fix the cause (another job running, dataset missing or mismatched) before retrying"})
        summary = store.compact_summary(rec)
        if rec.get("duplicate_of"):
            summary["note"] = f"identical config already completed as {rec['duplicate_of']}; not retrained"
            session.count_launch(task, -1)
        val = (rec.get("result") or {}).get("val_metrics_at_best") or {}
        summary["val_diagnostics"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items() if k in ("accuracy", "macro_f1", "miou", "pixel_accuracy", "foreground_iou", "n_classes_present", "loss")}
        rows = list(iter_jsonl(store.exp_dir(rec["experiment_id"]) / "checkpoints" / "curves.jsonl"))
        summary["curve"] = [{"epoch": r["epoch"], "train_loss": round(r["train_loss"], 4), "val": round(r.get(f"val_{summary['metric']}", 0.0), 4), "sec": round(r.get("train_seconds", 0) + r.get("val_seconds", 0))} for r in rows][-12:]
        summary["remaining_minutes"] = round(session.remaining_minutes(), 1)
        summary["experiments_remaining"] = session.experiments_remaining()
        summary["task_experiments_remaining"] = session.task_remaining(task)
        session.record_decision("experiment_result", task=task, experiment_id=rec["experiment_id"], status=rec.get("status"), best_val_metric=summary["best_val_metric"], stop_reason=summary["stop_reason"], adjustments=summary["adjustments"], error=summary["error"])
        session.save()
        return _json(summary)

    @tool
    def get_experiment(experiment_id: str) -> str:
        """Full details of one experiment: config, per-epoch curve, validation diagnostics, adjustments and errors.

        Args:
            experiment_id: experiment id from list_experiments
        """
        try:
            rec = store.read(experiment_id)
        except (FileNotFoundError, ValueError) as exc:
            return _json({"error": str(exc)})
        rows = list(iter_jsonl(store.exp_dir(experiment_id) / "checkpoints" / "curves.jsonl"))
        out = store.compact_summary(rec)
        out["curve"] = [{"epoch": r["epoch"], "train_loss": round(r["train_loss"], 4), "val": r.get(f"val_{out['metric']}"), "lr": r.get("lr"), "sec": round(r.get("train_seconds", 0) + r.get("val_seconds", 0))} for r in rows]
        val = (rec.get("result") or {}).get("val_metrics_at_best") or {}
        out["val_diagnostics"] = {k: v for k, v in val.items() if k not in ("confusion_matrix", "conventions", "per_class")}
        out["per_class"] = val.get("per_class")
        out["rationale"] = rec.get("rationale")
        out["expected_tradeoff"] = rec.get("expected_tradeoff")
        return _json(out)

    @tool
    def select_best_model(task: str, experiment_id: str, rationale: str) -> str:
        """Set the selected model for a task. Only completed, full-data (non-screening) experiments can be selected.

        Args:
            task: task name
            experiment_id: the experiment to select
            rationale: why this one (cite validation scores of the alternatives)
        """
        if task not in session.state["tasks"]:
            return _json({"error": f"task {task!r} is not in this session"})
        if len(rationale.strip()) < 12:
            return _json({"error": "rationale must be a real sentence citing scores"})
        try:
            pointer = store.set_best(task, experiment_id, by="agent", rationale=rationale.strip())
        except (ValueError, FileNotFoundError) as exc:
            return _json({"error": str(exc)})
        session.record_decision("select_best", task=task, experiment_id=experiment_id, rationale=rationale.strip(), best_val_metric=pointer["best_val_metric"])
        session.save()
        return _json({"selected": pointer["experiment_id"], "best_val_metric": pointer["best_val_metric"], "metric": pointer["metric"]["name"]})

    @tool
    def record_note(text: str) -> str:
        """Record an observation or plan in the session log (no side effects).

        Args:
            text: the note
        """
        session.record_decision("note", text=text.strip()[:2000])
        return _json({"recorded": True})

    @tool
    def finish_task(task: str, summary: str) -> str:
        """Declare the task finished. Call after select_best_model. The summary is recorded as the agent's final statement for the task.

        Args:
            task: task name
            summary: 3-5 sentences: what was tried, what worked, what was selected and why
        """
        if task not in session.state["tasks"]:
            return _json({"error": f"task {task!r} is not in this session"})
        best = store.best(task)
        if best is None:
            return _json({"error": "no best model selected yet; call select_best_model first"})
        session.mark_finished(task, {"summary": summary.strip()[:3000], "selected": best["experiment_id"], "best_val_metric": best["best_val_metric"]})
        session.record_decision("finish_task", task=task, summary=summary.strip()[:3000], selected=best["experiment_id"])
        session.save()
        return _json({"finished": task, "selected": best["experiment_id"]})

    tools = [inspect_dataset, list_experiments, get_budget_status, run_experiment, get_experiment, select_best_model, record_note, finish_task]
    if rt.registry is not None:
        from gotrain.artifacts.tools import build_registry_tools

        tools.extend(build_registry_tools(rt))
    return tools


def _default_root() -> Path:
    from gotrain.paths import geobench_dir

    return geobench_dir()


def tool_names(tools: list) -> set[str]:
    names = set()
    for t in tools:
        names.add(getattr(t, "tool_name", getattr(t, "__name__", str(t))))
    return names
