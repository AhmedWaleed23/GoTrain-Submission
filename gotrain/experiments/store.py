"""Experiment records: one directory per experiment, atomic record.json, append-only
events, and an explicit best-model pointer per task."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gotrain.paths import Home
from gotrain.train.metrics import SELECTION_METRIC
from gotrain.util.ids import new_id, validate_id
from gotrain.util.io import append_jsonl, dumps_json, iter_jsonl, read_json, write_json
from gotrain.util.timing import utc_now_iso

STATUSES = ("created", "running", "completed", "failed", "interrupted")


class ExperimentStore:
    def __init__(self, home: Home) -> None:
        self.home = home.ensure()
        self.root = home.experiments
        self.events_path = self.root / "events.jsonl"
        self.best_dir = self.root / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)

    def exp_dir(self, exp_id: str) -> Path:
        validate_id(exp_id, "experiment id")
        return self.root / exp_id

    def create(self, record: dict[str, Any]) -> str:
        exp_id = new_id("exp")
        while (self.root / exp_id).exists():
            exp_id = new_id("exp")
        record = dict(record)
        record["experiment_id"] = exp_id
        record["status"] = "created"
        record["created_at"] = utc_now_iso()
        d = self.exp_dir(exp_id)
        d.mkdir(parents=True, exist_ok=False)
        write_json(d / "config.json", record["run_config"])
        write_json(d / "record.json", record)
        self.log_event(exp_id, "created", task=record.get("task"), tag=record.get("tag"))
        return exp_id

    def read(self, exp_id: str) -> dict[str, Any]:
        path = self.exp_dir(exp_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"no experiment {exp_id}")
        return read_json(path)

    def exists(self, exp_id: str) -> bool:
        try:
            return (self.exp_dir(exp_id) / "record.json").exists()
        except ValueError:
            return False

    def update(self, exp_id: str, **fields: Any) -> dict[str, Any]:
        record = self.read(exp_id)
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"bad status {fields['status']}")
        record.update(fields)
        record["updated_at"] = utc_now_iso()
        write_json(self.exp_dir(exp_id) / "record.json", record)
        if "status" in fields:
            self.log_event(exp_id, "status", status=fields["status"], error=fields.get("error"))
        return record

    def log_event(self, exp_id: str, kind: str, **data: Any) -> None:
        event = {"at": utc_now_iso(), "experiment_id": exp_id, "kind": kind, **data}
        append_jsonl(self.exp_dir(exp_id) / "events.jsonl", event)
        append_jsonl(self.events_path, event)

    def all_records(self) -> list[dict[str, Any]]:
        records = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("exp-") and (d / "record.json").exists():
                try:
                    records.append(read_json(d / "record.json"))
                except ValueError:
                    continue
        return records

    def list(self, task: str | None = None, include_screening: bool = True, statuses=None, source: str | None = None) -> list[dict[str, Any]]:
        out = []
        for r in self.all_records():
            if task and r.get("task") != task:
                continue
            if not include_screening and r.get("screening"):
                continue
            if statuses and r.get("status") not in statuses:
                continue
            if source and r.get("source") != source:
                continue
            out.append(r)
        return out

    def ranking(self, task: str, source: str = "geobench") -> list[dict[str, Any]]:
        """Completed, full-validation, non-screening runs ordered by the frozen metric."""
        rows = [r for r in self.list(task, include_screening=False, statuses=("completed",), source=source) if r.get("result", {}).get("best_val_metric") is not None and not r.get("partial_validation")]
        rows.sort(key=lambda r: (-float(r["result"]["best_val_metric"]), r["experiment_id"]))
        return rows

    def find_completed_duplicate(self, config_hash: str) -> dict[str, Any] | None:
        for r in self.all_records():
            if r.get("config_hash") == config_hash and r.get("status") == "completed":
                return r
        return None

    def set_best(self, task: str, exp_id: str, by: str, rationale: str = "") -> dict[str, Any]:
        record = self.read(exp_id)
        if record.get("status") != "completed":
            raise ValueError(f"{exp_id} is {record.get('status')}, only completed experiments can be selected")
        if record.get("screening") or record.get("partial_validation"):
            raise ValueError(f"{exp_id} is a screening or partial-validation run and cannot be selected")
        if record.get("task") != task:
            raise ValueError(f"{exp_id} belongs to task {record.get('task')}, not {task}")
        pointer = {
            "task": task,
            "experiment_id": exp_id,
            "metric": record["metric"],
            "best_val_metric": record["result"]["best_val_metric"],
            "checkpoint": record["checkpoint"],
            "selected_by": by,
            "rationale": rationale,
            "selected_at": utc_now_iso(),
        }
        write_json(self.best_dir / f"{task}.json", pointer)
        self.log_event(exp_id, "selected_best", task=task, by=by, rationale=rationale)
        return pointer

    def best(self, task: str) -> dict[str, Any] | None:
        path = self.best_dir / f"{task}.json"
        return read_json(path) if path.exists() else None

    def compact_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        """Small dict for agent prompts and tables."""
        rc = record.get("run_config", {})
        res = record.get("result", {}) or {}
        comp = record.get("compute", {}) or {}
        return {
            "experiment_id": record["experiment_id"],
            "task": record.get("task"),
            "status": record.get("status"),
            "tag": record.get("tag"),
            "screening": bool(record.get("screening")),
            "model": rc.get("model"),
            "lr": rc.get("lr"),
            "batch_size": rc.get("batch_size"),
            "epochs_requested": rc.get("epochs"),
            "epochs_completed": res.get("epochs_completed"),
            "subset_fraction": rc.get("subset_fraction"),
            "input_size": rc.get("input_size"),
            "augment": rc.get("augment"),
            "weight_decay": rc.get("weight_decay"),
            "metric": record.get("metric", {}).get("name"),
            "best_val_metric": res.get("best_val_metric"),
            "best_epoch": res.get("best_epoch"),
            "stop_reason": res.get("stop_reason"),
            "wall_minutes": round(float(comp.get("wall_time_seconds") or 0.0) / 60.0, 2),
            "adjustments": comp.get("adjustments", []),
            "error": record.get("error"),
            "hypothesis": record.get("hypothesis", ""),
        }


class GpuLock:
    """One training job at a time. The lock file records pid and experiment id; a lock
    held by a dead process is treated as stale."""

    def __init__(self, home: Home, name: str = "training") -> None:
        self.path = home.locks / f"{name}.lock"
        self.held = False

    def _alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def holder(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            info = read_json(self.path)
        except ValueError:
            return None
        if not self._alive(int(info.get("pid", -1))):
            return None
        return info

    def acquire(self, exp_id: str) -> None:
        payload = dumps_json({"pid": os.getpid(), "experiment_id": exp_id, "at": utc_now_iso()})
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = self.holder()
                if holder is None:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if holder.get("pid") == os.getpid():
                    write_json(self.path, {"pid": os.getpid(), "experiment_id": exp_id, "at": utc_now_iso()})
                    self.held = True
                    return
                raise RuntimeError(f"another training job is running: {holder} (lock {self.path}); wait for it or remove a stale lock")
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            self.held = True
            return
        raise RuntimeError(f"could not acquire the training lock at {self.path}")

    def release(self) -> None:
        if self.held and self.path.exists():
            try:
                info = read_json(self.path)
                if info.get("pid") == os.getpid():
                    self.path.unlink()
            except (ValueError, FileNotFoundError):
                pass
        self.held = False


def metric_definition(task_type: str) -> dict[str, str]:
    name = SELECTION_METRIC[task_type]
    if name == "accuracy":
        definition = "overall accuracy on the full validation split (torchmetrics multiclass Accuracy convention, as in geo-bench-experiments)"
    else:
        definition = "macro mean IoU over classes present in target or prediction, confusion matrix accumulated over the full validation split (torchmetrics multiclass JaccardIndex convention, as in geo-bench-experiments)"
    return {"name": name, "direction": "max", "definition": definition, "split": "valid"}


def list_events(home: Home, exp_id: str | None = None) -> list[dict[str, Any]]:
    path = home.experiments / exp_id / "events.jsonl" if exp_id else home.experiments / "events.jsonl"
    return list(iter_jsonl(path))
