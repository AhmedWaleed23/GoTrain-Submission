"""Run or resume one experiment end to end and persist a complete record."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gotrain.config import Limits, RunConfig, TaskConfig, load_task_config
from gotrain.data.geobench_adapter import open_split
from gotrain.data.normalization import Normalizer, display_rgb, fit_band_stats, load_stats, save_stats
from gotrain.data.synthetic import synthetic_splits
from gotrain.data.transforms import TransformedDataset, logits_to_mask_resolution
from gotrain.experiments.store import ExperimentStore, GpuLock, metric_definition
from gotrain.models.factory import build_model, count_parameters
from gotrain.paths import Home, geobench_dir
from gotrain.train.checkpoint import load_checkpoint
from gotrain.train.trainer import Trainer, TrainingAborted
from gotrain.util.env import capture_environment, default_device
from gotrain.util.io import read_json, sha256_json, write_json
from gotrain.util.timing import utc_now_iso

N_VAL_EXAMPLES = 8


class DataBundle:
    def __init__(self, train, valid, meta, norm_stats: dict, norm_path: Path | None, train_indices, valid_limit) -> None:
        self.train = train
        self.valid = valid
        self.meta = meta
        self.norm_stats = norm_stats
        self.norm_path = norm_path
        self.train_indices = train_indices
        self.valid_limit = valid_limit


def normalization_for_task(home: Home, task_cfg: TaskConfig, train_ds, log: Callable[[str], None]) -> tuple[dict, Path]:
    path = home.normalization / f"{task_cfg.name}.json"
    if path.exists():
        stats = load_stats(path)
        if stats.get("n_samples_in_split") == len(train_ds) and stats.get("band_names") == list(train_ds.meta.band_names):
            return stats, path
        log(f"normalization cache for {task_cfg.name} does not match the current train split; refitting")
    log(f"fitting per-band normalization on the train split of {task_cfg.name} ({len(train_ds)} samples)")
    stats = fit_band_stats(train_ds, log=log)
    save_stats(stats, path)
    return load_stats(path), path


def prepare_data(home: Home, run: RunConfig, source: str, log: Callable[[str], None], geobench_root: Path | None = None, task_cfg: TaskConfig | None = None) -> DataBundle:
    if source == "synthetic":
        task_type = "classification" if run.model in ("cnn_small", "resnet_small") else "segmentation"
        train, valid = synthetic_splits(task_type, seed=run.seed)
        stats = fit_band_stats(train)
        norm_path = None
    else:
        task_cfg = task_cfg or load_task_config(run.task)
        root = geobench_root or geobench_dir()
        train = open_split(task_cfg, "train", root)
        valid = open_split(task_cfg, "valid", root)
        stats, norm_path = normalization_for_task(home, task_cfg, train, log)
    meta = train.meta
    train_indices = None
    if run.subset_fraction < 1.0:
        n = max(1, int(round(len(train) * run.subset_fraction)))
        gen = torch.Generator().manual_seed(run.seed)
        train_indices = sorted(torch.randperm(len(train), generator=gen)[:n].tolist())
    valid_indices = list(range(min(len(valid), run.val_limit))) if run.val_limit else None
    norm = Normalizer.from_stats(stats)
    train_t = TransformedDataset(train, norm, meta.task_type, augment=run.augment, input_size=run.input_size, seed=run.seed, indices=train_indices)
    valid_t = TransformedDataset(valid, norm, meta.task_type, augment=False, input_size=run.input_size, indices=valid_indices)
    return DataBundle(train_t, valid_t, meta, stats, norm_path, train_indices, run.val_limit)


def config_hash(run: RunConfig, source: str) -> str:
    payload = run.to_dict()
    for k in ("hypothesis", "rationale", "expected_tradeoff", "tag", "parent_experiment", "num_workers"):
        payload.pop(k, None)
    payload["source"] = source
    return sha256_json(payload)


def _capture_val_examples(model, bundle: DataBundle, device: str, out_path: Path, use_amp: bool) -> dict:
    model.eval()
    n = min(N_VAL_EXAMPLES, len(bundle.valid))
    rgbs, labels, preds = [], [], []
    with torch.no_grad():
        for i in range(n):
            x, y = bundle.valid[i]
            raw_x, _ = bundle.valid.base[bundle.valid.indices[i] if bundle.valid.indices is not None else i]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device == "cuda"):
                logits = model(x.unsqueeze(0).to(device))
            pred = logits_to_mask_resolution(logits.float(), y.unsqueeze(0).to(logits.device)).argmax(dim=1)[0].cpu()
            rgb = display_rgb(raw_x, bundle.meta.rgb_indices)
            if rgb.shape[0] > 160:
                step = int(np.ceil(rgb.shape[0] / 160))
                rgb = rgb[::step, ::step]
                if bundle.meta.task_type == "segmentation":
                    y = y[::step, ::step]
                    pred = pred[::step, ::step]
            rgbs.append(rgb)
            labels.append(y.numpy().astype(np.int16))
            preds.append(pred.numpy().astype(np.int16))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, rgb=np.stack(rgbs), label=np.stack(labels), pred=np.stack(preds), indices=np.arange(n))
    return {"path": str(out_path.relative_to(out_path.parents[1])), "n": n, "split": "valid", "indices": list(range(n))}


def run_experiment(
    home: Home,
    run: RunConfig,
    source: str = "geobench",
    device: str | None = None,
    log: Callable[[str], None] = print,
    limits: Limits | None = None,
    geobench_root: Path | None = None,
    allow_duplicate: bool = False,
    resume_id: str | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict:
    limits = limits or Limits.load()
    store = ExperimentStore(home)
    device = device or default_device()
    task_cfg = None if source == "synthetic" else load_task_config(run.task)
    if source != "synthetic":
        problems = run.validate(limits, task_cfg)
        if problems:
            raise ValueError("invalid run config: " + "; ".join(problems))
    chash = config_hash(run, source)

    if resume_id is None:
        dup = None if allow_duplicate else store.find_completed_duplicate(chash)
        if dup is not None:
            log(f"identical completed experiment exists: {dup['experiment_id']} (use allow_duplicate to retrain)")
            dup = dict(dup)
            dup["duplicate_of"] = dup["experiment_id"]
            return dup
        task_type = task_cfg.task_type if task_cfg else ("classification" if run.model in ("cnn_small", "resnet_small") else "segmentation")
        record = {
            "task": run.task,
            "task_type": task_type,
            "source": source,
            "tag": run.tag,
            "screening": bool(run.screening),
            "partial_validation": bool(run.val_limit),
            "run_config": run.to_dict(),
            "config_hash": chash,
            "seed": run.seed,
            "hypothesis": run.hypothesis,
            "rationale": run.rationale,
            "expected_tradeoff": run.expected_tradeoff,
            "parent_experiment": run.parent_experiment,
            "metric": metric_definition(task_type),
            "pretrained": None,
            "environment": capture_environment(),
            "device": device,
        }
        exp_id = store.create(record)
    else:
        exp_id = resume_id
        record = store.read(exp_id)
        if record["status"] == "completed":
            log(f"{exp_id} already completed; not re-running")
            return record
        run = RunConfig.from_dict(record["run_config"])
        source = record["source"]
        task_cfg = None if source == "synthetic" else load_task_config(run.task)
        store.log_event(exp_id, "resume_requested")

    exp_dir = store.exp_dir(exp_id)
    log_path = exp_dir / "log.txt"

    def tee(msg: str) -> None:
        line = f"[{utc_now_iso()}] {msg}"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        log(line)

    lock = GpuLock(home)
    try:
        lock.acquire(exp_id)
    except RuntimeError as exc:
        store.update(exp_id, status="failed", error=str(exc))
        raise

    try:
        store.update(exp_id, status="running", started_at=record.get("started_at") or utc_now_iso())
        bundle = prepare_data(home, run, source, tee, geobench_root=geobench_root, task_cfg=task_cfg)
        meta = bundle.meta
        model = build_model(run.model, meta.in_channels, meta.n_classes, meta.task_type)
        store.update(
            exp_id,
            dataset={
                "source": source,
                "dataset_dir": meta.extra.get("dataset_dir"),
                "split_sizes": meta.split_sizes,
                "train_samples_used": len(bundle.train),
                "valid_samples_used": len(bundle.valid),
                "subset_fraction": run.subset_fraction,
                "train_subset_indices_hash": sha256_json(bundle.train_indices) if bundle.train_indices is not None else None,
                "band_names": meta.band_names,
                "input_size": run.input_size or [meta.height, meta.width],
                "resized": bool(run.input_size and run.input_size != meta.height),
            },
            normalization={"path": str(bundle.norm_path) if bundle.norm_path else "in-memory (synthetic)", "stats_hash": bundle.norm_stats.get("stats_hash"), "n_samples_used": bundle.norm_stats.get("n_samples_used"), "split": "train"},
            model={"name": run.model, "in_channels": meta.in_channels, "n_classes": meta.n_classes, "class_names": meta.class_names, "parameters": count_parameters(model), "ignore_index": meta.ignore_index},
        )
        tee(f"{exp_id}: {run.model} on {run.task} ({source}), {len(bundle.train)} train / {len(bundle.valid)} valid samples, device {device}")
        exclude = task_cfg.diagnostic_exclude_classes if task_cfg else ()
        trainer = Trainer(
            run, model, bundle.train, bundle.valid, meta.task_type, meta.n_classes, meta.class_names, exp_dir / "checkpoints", device,
            ignore_index=meta.ignore_index, exclude_from_diagnostic_miou=exclude, log=tee, max_retries=limits.max_retries_per_run, stop_requested=stop_requested,
        )
        store.update(exp_id, precision={"use_amp": trainer.use_amp, "requested": run.precision})
        state = trainer.fit(resume=resume_id is not None)
        result = {
            "best_val_metric": state.get("best_metric"),
            "best_epoch": state.get("best_epoch"),
            "epochs_completed": state.get("epoch_completed"),
            "stop_reason": state.get("stop_reason"),
        }
        compute = {
            "wall_time_seconds": state.get("wall_time_seconds"),
            "resume_count": state.get("resume_count", 0),
            "adjustments": state.get("adjustments", []),
            "effective_batch_size": state.get("effective_batch_size"),
            "effective_lr": state.get("effective_lr"),
            "nonfinite_steps": state.get("nonfinite_steps", 0),
            "device": device,
        }
        checkpoint = {}
        val_examples = None
        if state.get("best_metric") is not None:
            best_path = exp_dir / "checkpoints" / "best.pt"
            ck = load_checkpoint(best_path, map_location=device, expected_sha256=state.get("best_sha256"))
            model.load_state_dict(ck["model_state"])
            result["val_metrics_at_best"] = ck.get("val_metrics")
            checkpoint = {
                "best_path": str(best_path.relative_to(home.root)),
                "best_sha256": state.get("best_sha256"),
                "last_path": str((exp_dir / "checkpoints" / "last.pt").relative_to(home.root)),
                "last_sha256": state.get("last_sha256"),
                "format": "gotrain-checkpoint-v1",
            }
            val_examples = _capture_val_examples(model, bundle, device, exp_dir / "predictions" / "val_examples.npz", trainer.use_amp)
        status = "completed" if state.get("status") == "completed" else state.get("status", "failed")
        rec = store.update(exp_id, status=status, result=result, compute=compute, checkpoint=checkpoint, val_examples=val_examples, finished_at=utc_now_iso(), error=state.get("error"))
        if status == "completed" and not run.screening and source != "synthetic" and not run.val_limit:
            current = store.best(run.task)
            if current is None or (current.get("selected_by") == "auto" and float(rec["result"]["best_val_metric"]) > float(current["best_val_metric"])):
                store.set_best(run.task, exp_id, by="auto", rationale="highest validation metric among completed full runs")
        tee(f"{exp_id}: {status} ({result['stop_reason']}), best {record['metric']['name']}={result['best_val_metric']} at epoch {result['best_epoch']}")
        return rec
    except TrainingAborted as exc:
        tee(f"{exp_id}: aborted: {exc}")
        return store.update(exp_id, status="failed", error=str(exc), finished_at=utc_now_iso(), compute=_compute_from_state(exp_dir))
    except KeyboardInterrupt:
        tee(f"{exp_id}: interrupted by user; resume with: python -m gotrain resume {exp_id}")
        store.update(exp_id, status="interrupted", finished_at=utc_now_iso(), compute=_compute_from_state(exp_dir))
        raise
    except Exception as exc:
        tee(f"{exp_id}: failed: {exc}\n{traceback.format_exc()}")
        return store.update(exp_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000], finished_at=utc_now_iso())
    finally:
        lock.release()


def _compute_from_state(exp_dir: Path) -> dict:
    path = exp_dir / "checkpoints" / "trainer_state.json"
    if not path.exists():
        return {}
    st = read_json(path)
    return {"wall_time_seconds": st.get("wall_time_seconds"), "resume_count": st.get("resume_count", 0), "adjustments": st.get("adjustments", []), "effective_batch_size": st.get("effective_batch_size"), "effective_lr": st.get("effective_lr")}


def resume_experiment(home: Home, exp_id: str, device: str | None = None, log=print, geobench_root: Path | None = None, stop_requested=None) -> dict:
    store = ExperimentStore(home)
    record = store.read(exp_id)
    run = RunConfig.from_dict(record["run_config"])
    return run_experiment(home, run, source=record["source"], device=device, log=log, geobench_root=geobench_root, resume_id=exp_id, stop_requested=stop_requested)


def write_selection_note(home: Home, task: str, note: dict) -> None:
    write_json(home.selection / f"{task}_note.json", note)
