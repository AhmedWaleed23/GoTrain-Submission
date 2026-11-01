"""Fixed-config baselines: one run per task with no LLM involved. The agent is judged
against these at a comparable budget."""

from __future__ import annotations

from pathlib import Path

from gotrain.config import Budget, RunConfig, load_task_config
from gotrain.experiments.runner import run_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home

BASELINE_BY_TYPE = {
    "classification": {"model": "cnn_small", "lr": 1e-3, "batch_size": 32},
    "segmentation": {"model": "unet_small", "lr": 1e-3, "batch_size": 16},
}


def baseline_config(task: str, budget: Budget, epochs: int | None = None, max_minutes: float | None = None, num_workers: int = 2, seed: int = 0) -> RunConfig:
    cfg = load_task_config(task)
    base = BASELINE_BY_TYPE[cfg.task_type]
    return RunConfig(
        task=task,
        model=base["model"],
        lr=base["lr"],
        batch_size=base["batch_size"],
        epochs=epochs or budget.baseline_epochs,
        max_minutes=max_minutes or budget.baseline_max_minutes,
        seed=seed,
        num_workers=num_workers,
        tag="baseline",
        hypothesis="fixed reference configuration; no tuning",
        rationale="establishes a comparison point for the agent at the same budget",
        expected_tradeoff="none; this is the control",
    )


def run_baselines(home: Home, tasks: list[str], budget: Budget, device: str | None = None, geobench_root: Path | None = None, epochs: int | None = None, max_minutes: float | None = None, num_workers: int = 2, seed: int = 0, log=print) -> dict:
    store = ExperimentStore(home)
    out = {}
    for task in tasks:
        run = baseline_config(task, budget, epochs=epochs, max_minutes=max_minutes, num_workers=num_workers, seed=seed)
        rec = run_experiment(home, run, source="geobench", device=device, geobench_root=geobench_root, log=log)
        out[task] = store.compact_summary(rec)
        if rec.get("duplicate_of"):
            out[task]["duplicate_of"] = rec["duplicate_of"]
    return out


def baseline_record(store: ExperimentStore, task: str) -> dict | None:
    rows = [r for r in store.list(task=task, statuses=("completed",)) if r.get("tag") == "baseline" and not r.get("screening")]
    rows.sort(key=lambda r: r["experiment_id"])
    return rows[-1] if rows else None
