"""Training loop with bounded time, early stopping, AMP, OOM/nonfinite recovery and resume.

The trainer knows nothing about experiment IDs or the agent. It receives already
built datasets, a model, a run config and an output directory, and writes:
  curves.jsonl      one line per epoch (train loss, val metrics, timing)
  best.pt / last.pt checkpoints (atomic)
  trainer_state.json  progress for resume, adjustments made (batch size, lr)
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from gotrain.config import RunConfig
from gotrain.data.transforms import TransformedDataset
from gotrain.train.checkpoint import load_checkpoint, save_checkpoint
from gotrain.train.evaluate import evaluate
from gotrain.train.metrics import SELECTION_METRIC
from gotrain.util.io import append_jsonl, atomic_write_text, dumps_json, iter_jsonl, read_json, sha256_file, write_json
from gotrain.util.seed import seed_everything
from gotrain.util.timing import utc_now_iso


class TrainingAborted(RuntimeError):
    pass


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cublas_status_alloc_failed" in msg


def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, seed: int, device: str) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=gen if shuffle else None,
        pin_memory=device == "cuda",
        drop_last=False,
        persistent_workers=False,
    )


class Trainer:
    def __init__(
        self,
        run: RunConfig,
        model: nn.Module,
        train_ds: TransformedDataset,
        valid_ds: TransformedDataset,
        task_type: str,
        n_classes: int,
        class_names: list[str],
        out_dir: Path,
        device: str,
        ignore_index: int | None = None,
        exclude_from_diagnostic_miou=(),
        log: Callable[[str], None] = print,
        max_retries: int = 2,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.run = run
        self.model = model.to(device)
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.task_type = task_type
        self.n_classes = n_classes
        self.class_names = class_names
        self.out_dir = Path(out_dir)
        self.device = device
        self.ignore_index = ignore_index
        self.exclude = exclude_from_diagnostic_miou
        self.log = log
        self.max_retries = max_retries
        self.stop_requested = stop_requested or (lambda: False)
        self.metric_name = SELECTION_METRIC[task_type]
        self.use_amp = self._decide_amp()
        self.state_path = self.out_dir / "trainer_state.json"
        self.curves_path = self.out_dir / "curves.jsonl"
        self.best_path = self.out_dir / "best.pt"
        self.last_path = self.out_dir / "last.pt"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100,
            label_smoothing=run.label_smoothing,
        )

    def _decide_amp(self) -> bool:
        if self.run.precision == "fp32":
            return False
        if self.run.precision == "fp16":
            return self.device == "cuda"
        return self.device == "cuda"

    def _build_optimizer(self, lr: float, completed_epochs: int = 0):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.run.weight_decay)
        if self.run.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.run.epochs), eta_min=lr * 0.05)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(completed_epochs):
                    sched.step()
        else:
            sched = None
        return opt, sched

    def _truncate_curves(self, epoch_completed: int) -> None:
        """Drop curve rows beyond the last committed epoch (an interrupted write can leave one)."""
        if not self.curves_path.exists():
            return
        rows = [r for r in iter_jsonl(self.curves_path) if int(r.get("epoch", 0)) <= epoch_completed]
        seen = set()
        kept = []
        for r in rows:
            if r["epoch"] in seen:
                continue
            seen.add(r["epoch"])
            kept.append(r)
        atomic_write_text(self.curves_path, "".join(dumps_json(r, indent=None) + "\n" for r in kept))

    def _reconcile_best(self, state: dict) -> None:
        """If best.pt on disk is not the checkpoint the state knows about (interrupted epoch),
        the best tracking is reset and re-established by the next evaluated epoch."""
        if state.get("best_sha256") and self.best_path.exists() and sha256_file(self.best_path) != state["best_sha256"]:
            state["adjustments"].append({"epoch": state["epoch_completed"], "kind": "best_checkpoint_reset", "reason": "best.pt did not match the committed state after an interruption", "at": utc_now_iso()})
            state["best_metric"] = None
            state["best_epoch"] = None
            state["best_sha256"] = None
            state["epochs_without_improvement"] = 0
            self.log("best checkpoint from an interrupted epoch discarded; the next evaluated epoch re-establishes it")

    def _initial_state(self) -> dict:
        return {
            "epoch_completed": 0,
            "best_metric": None,
            "best_epoch": None,
            "epochs_without_improvement": 0,
            "effective_batch_size": self.run.batch_size,
            "effective_lr": self.run.lr,
            "adjustments": [],
            "retries": 0,
            "nonfinite_steps": 0,
            "wall_time_seconds": 0.0,
            "resume_count": 0,
            "status": "running",
            "stop_reason": None,
            "use_amp": self.use_amp,
            "device": self.device,
            "started_at": utc_now_iso(),
        }

    def _save_state(self, state: dict) -> None:
        write_json(self.state_path, state)

    def fit(self, resume: bool = False) -> dict:
        seed_everything(self.run.seed)
        state = self._initial_state()
        opt, sched = self._build_optimizer(self.run.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if resume and self.last_path.exists() and self.state_path.exists():
            state = read_json(self.state_path)
            if state.get("status") == "completed":
                self.log("run already completed; nothing to resume")
                return state
            ck = load_checkpoint(self.last_path, map_location=self.device, expected_sha256=state.get("last_sha256"))
            self.model.load_state_dict(ck["model_state"])
            opt, sched = self._build_optimizer(state["effective_lr"])
            opt.load_state_dict(ck["optimizer_state"])
            if sched is not None and ck.get("scheduler_state"):
                sched.load_state_dict(ck["scheduler_state"])
            if ck.get("scaler_state") and self.use_amp:
                scaler.load_state_dict(ck["scaler_state"])
            torch.set_rng_state(ck["rng_cpu"].cpu().to(torch.uint8))
            self._truncate_curves(int(state["epoch_completed"]))
            self._reconcile_best(state)
            state["resume_count"] = state.get("resume_count", 0) + 1
            state["status"] = "running"
            state["stop_reason"] = None
            self.log(f"resuming after epoch {state['epoch_completed']} (resume #{state['resume_count']})")
        elif resume:
            self.log("no checkpoint to resume from; starting fresh")

        valid_loader = _make_loader(self.valid_ds, max(1, state["effective_batch_size"]), False, self.run.num_workers, self.run.seed, self.device)
        budget_seconds = self.run.max_minutes * 60.0
        session_start = time.monotonic()
        prior_wall = float(state.get("wall_time_seconds", 0.0))

        def elapsed_total() -> float:
            return prior_wall + (time.monotonic() - session_start)

        epoch = state["epoch_completed"]
        while epoch < self.run.epochs:
            if elapsed_total() >= budget_seconds:
                state["status"] = "completed"
                state["stop_reason"] = "time_budget"
                break
            if self.stop_requested():
                state["status"] = "interrupted"
                state["stop_reason"] = "stop_requested"
                break
            self.train_ds.set_epoch(epoch)
            train_loader = _make_loader(self.train_ds, state["effective_batch_size"], True, self.run.num_workers, self.run.seed + epoch, self.device)
            t0 = time.monotonic()
            try:
                train_loss, nonfinite, truncated = self._train_one_epoch(train_loader, opt, scaler, state, budget_seconds, elapsed_total)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                if state["retries"] >= self.max_retries or state["effective_batch_size"] <= 2:
                    state["status"] = "failed"
                    state["stop_reason"] = "oom_unrecoverable"
                    state["error"] = str(exc)[:500]
                    self._save_state(state)
                    raise TrainingAborted(f"CUDA OOM after {state['retries']} retries") from exc
                new_bs = max(2, state["effective_batch_size"] // 2)
                state["adjustments"].append(
                    {"epoch": epoch, "kind": "batch_size", "from": state["effective_batch_size"], "to": new_bs, "reason": "cuda_oom", "at": utc_now_iso()}
                )
                state["effective_batch_size"] = new_bs
                state["retries"] += 1
                self._save_state(state)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                self.log(f"CUDA OOM: retrying epoch {epoch + 1} with batch size {new_bs}")
                valid_loader = _make_loader(self.valid_ds, new_bs, False, self.run.num_workers, self.run.seed, self.device)
                continue

            if train_loss is None:
                if state["retries"] >= self.max_retries or state["effective_lr"] <= 1e-6:
                    state["status"] = "failed"
                    state["stop_reason"] = "nonfinite_loss"
                    self._save_state(state)
                    raise TrainingAborted("training loss was non-finite repeatedly")
                new_lr = state["effective_lr"] / 10.0
                state["adjustments"].append(
                    {"epoch": epoch, "kind": "lr", "from": state["effective_lr"], "to": new_lr, "reason": "nonfinite_loss", "at": utc_now_iso()}
                )
                state["effective_lr"] = new_lr
                state["retries"] += 1
                opt, sched = self._build_optimizer(new_lr, completed_epochs=epoch)
                self._save_state(state)
                self.log(f"non-finite loss: retrying epoch {epoch + 1} with lr {new_lr:g}")
                continue

            if sched is not None:
                sched.step()
            train_seconds = time.monotonic() - t0
            t1 = time.monotonic()
            val = evaluate(
                self.model, valid_loader, self.task_type, self.n_classes, self.class_names, self.device,
                ignore_index=self.ignore_index, exclude_from_diagnostic_miou=self.exclude, use_amp=self.use_amp, loss_fn=self.loss_fn,
            )
            val_seconds = time.monotonic() - t1
            metric = float(val[self.metric_name])
            epoch += 1
            improved = state["best_metric"] is None or metric > state["best_metric"]
            if improved:
                state["best_metric"] = metric
                state["best_epoch"] = epoch
                state["epochs_without_improvement"] = 0
                state["best_sha256"] = save_checkpoint(self.best_path, self._payload(opt, sched, scaler, epoch, val, is_best=True))
            else:
                state["epochs_without_improvement"] += 1
            state["epoch_completed"] = epoch
            state["nonfinite_steps"] += nonfinite
            state["wall_time_seconds"] = elapsed_total()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val.get("loss"),
                f"val_{self.metric_name}": metric,
                "val_metrics_summary": _summary(val),
                "lr": opt.param_groups[0]["lr"],
                "batch_size": state["effective_batch_size"],
                "train_seconds": round(train_seconds, 2),
                "val_seconds": round(val_seconds, 2),
                "elapsed_seconds": round(state["wall_time_seconds"], 2),
                "nonfinite_steps": nonfinite,
                "truncated_by_budget": truncated,
                "improved": improved,
                "at": utc_now_iso(),
            }
            state["last_sha256"] = save_checkpoint(self.last_path, self._payload(opt, sched, scaler, epoch, val, is_best=False))
            self._save_state(state)
            append_jsonl(self.curves_path, row)
            self.log(
                f"epoch {epoch}/{self.run.epochs} train_loss={train_loss:.4f} val_{self.metric_name}={metric:.4f} "
                f"best={state['best_metric']:.4f}@{state['best_epoch']} ({train_seconds:.0f}s+{val_seconds:.0f}s)"
            )
            if state["epochs_without_improvement"] >= self.run.early_stopping_patience:
                state["status"] = "completed"
                state["stop_reason"] = "early_stopping"
                break
        else:
            state["status"] = "completed"
            state["stop_reason"] = "max_epochs"

        if state["status"] == "running":
            state["status"] = "completed"
        if state["status"] == "completed" and state["best_metric"] is None:
            state["status"] = "failed"
            state["stop_reason"] = state["stop_reason"] or "no_epoch_completed"
        state["wall_time_seconds"] = elapsed_total()
        state["finished_at"] = utc_now_iso()
        self._save_state(state)
        return state

    def _train_one_epoch(self, loader, opt, scaler, state, budget_seconds, elapsed_total):
        self.model.train()
        total = 0.0
        n = 0
        nonfinite = 0
        truncated = False
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                nonfinite += 1
                if nonfinite > 20:
                    return None, nonfinite, truncated
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            n += 1
            if elapsed_total() >= budget_seconds and n > 0:
                truncated = True
                break
        if n == 0:
            return None, nonfinite, truncated
        return total / n, nonfinite, truncated

    def _payload(self, opt, sched, scaler, epoch, val, is_best: bool) -> dict:
        return {
            "format": "gotrain-checkpoint-v1",
            "epoch": epoch,
            "is_best": is_best,
            "model_state": self.model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict() if sched is not None else None,
            "scaler_state": scaler.state_dict() if self.use_amp else None,
            "rng_cpu": torch.get_rng_state(),
            "val_metric_name": self.metric_name,
            "val_metric": float(val[self.metric_name]),
            "val_metrics": val,
            "run_config": self.run.to_dict(),
            "saved_at": utc_now_iso(),
        }


def _summary(val: dict) -> dict:
    keys = ("accuracy", "macro_f1", "miou", "pixel_accuracy", "foreground_iou", "n_classes_present")
    out = {k: val[k] for k in keys if k in val and val[k] is not None}
    if "miou_excluding" in val:
        out["miou_excluding"] = val["miou_excluding"]["value"]
    return out


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
