"""One-shot test-split evaluation from a frozen manifest.

Guarantees:
- the test split is opened only here, with a FinalEvalToken created only here;
- every opening is logged to final_eval/exposure.jsonl before any prediction;
- results are persisted per task as they complete, so a crash keeps finished tasks;
- a task with persisted results is never evaluated again by this command;
- a manifest that differs from the one already evaluated is refused.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gotrain.config import TaskConfig
from gotrain.data.base import FinalEvalToken
from gotrain.data.normalization import Normalizer, load_stats
from gotrain.data.transforms import TransformedDataset
from gotrain.final_eval.freeze import load_manifest
from gotrain.models.factory import build_model
from gotrain.paths import Home, geobench_dir
from gotrain.train.checkpoint import load_checkpoint
from gotrain.train.evaluate import evaluate
from gotrain.util.env import capture_environment, default_device
from gotrain.util.io import append_jsonl, ensure_within, iter_jsonl, read_json, sha256_file, write_json
from gotrain.util.timing import utc_now_iso


def results_path(home: Home) -> Path:
    return home.final_eval / "results.json"


def _load_results(home: Home, manifest_hash: str) -> dict:
    path = results_path(home)
    if path.exists():
        res = read_json(path)
        if res.get("manifest_hash") != manifest_hash:
            raise PermissionError(
                "final_eval/results.json was produced from a different frozen manifest; "
                "re-evaluating the test split after changing the selection is not allowed"
            )
        return res
    return {"manifest_hash": manifest_hash, "tasks": {}, "created_at": utc_now_iso()}


def _reject_prior_exposure(home: Home, task: str) -> None:
    # Even an interrupted run can have exposed test predictions.
    for event in iter_jsonl(home.final_eval / "exposure.jsonl"):
        if event.get("task") == task and event.get("kind") == "test_split_opened":
            raise PermissionError(
                f"{task}: prior test exposure recorded; automatic retry is blocked. "
                "Preserve logs and review the interrupted run before any further test access."
            )


def final_evaluate(home: Home, manifest_file: Path | None = None, tasks: list[str] | None = None, device: str | None = None, geobench_root: Path | None = None, acknowledge: bool = False, batch_size: int = 32, num_workers: int = 2, log=print) -> dict:
    if not acknowledge:
        raise PermissionError("refusing: pass acknowledge=True (CLI: --i-understand-this-uses-the-test-split)")
    manifest_file = Path(manifest_file) if manifest_file else home.selection / "manifest.json"
    manifest = load_manifest(manifest_file)
    device = device or default_device()
    results = _load_results(home, manifest["manifest_hash"])
    exposure_log = home.final_eval / "exposure.jsonl"
    wanted = tasks or list(manifest["tasks"])
    for task in wanted:
        if task not in manifest["tasks"]:
            log(f"{task}: not in the frozen manifest; skipped")
            continue
        if task in results["tasks"] and results["tasks"][task].get("status") == "completed":
            log(f"{task}: already evaluated on {results['tasks'][task]['evaluated_at']}; refusing to re-evaluate")
            continue
        _reject_prior_exposure(home, task)
        entry = manifest["tasks"][task]
        rec_out = {"experiment_id": entry["experiment_id"], "status": "running", "started_at": utc_now_iso(), "device": device}
        try:
            ck_path = ensure_within(home.root, home.root / entry["checkpoint_path"])
            sha = sha256_file(ck_path)
            if sha != entry["checkpoint_sha256"]:
                raise ValueError("checkpoint hash does not match the frozen manifest")
            arch = entry["model"]
            model = build_model(arch["name"], arch["in_channels"], arch["n_classes"], entry["task_type"])
            ck = load_checkpoint(ck_path, map_location=device, expected_sha256=entry["checkpoint_sha256"])
            model.load_state_dict(ck["model_state"])
            model.to(device)
            if entry["source"] == "synthetic":
                from gotrain.data.normalization import fit_band_stats
                from gotrain.data.synthetic import SyntheticClassification, SyntheticSegmentation, synthetic_splits

                train, _ = synthetic_splits(entry["task_type"], seed=entry["seed"])
                stats = fit_band_stats(train)
                token = FinalEvalToken(f"final evaluation of {task} from manifest {manifest['manifest_hash'][:12]}")
                cls = SyntheticClassification if entry["task_type"] == "classification" else SyntheticSegmentation
                test_ds = cls("test", seed=entry["seed"], token=token)
                task_type = entry["task_type"]
                ignore_index = None
                exclude = ()
                class_names = test_ds.meta.class_names
                n_classes = test_ds.meta.n_classes
            else:
                if not entry.get("task_config"):
                    raise PermissionError("Manifest lacks frozen task config; freeze again before test access")
                task_cfg = TaskConfig(**entry["task_config"])
                stats = load_stats(Path(entry["normalization_path"]))
                if stats["stats_hash"] != entry["normalization_hash"]:
                    raise ValueError("normalization statistics do not match the frozen manifest")
                from gotrain.data.geobench_adapter import open_split

                token = FinalEvalToken(f"final evaluation of {task} from manifest {manifest['manifest_hash'][:12]}")
                append_jsonl(exposure_log, {"at": utc_now_iso(), "kind": "test_split_opened", "task": task, "experiment_id": entry["experiment_id"], "manifest_hash": manifest["manifest_hash"], "purpose": token.purpose})
                test_ds = open_split(task_cfg, "test", geobench_root or geobench_dir(), token=token)
                task_type = task_cfg.task_type
                ignore_index = task_cfg.ignore_index
                exclude = task_cfg.diagnostic_exclude_classes
                class_names = test_ds.meta.class_names
                n_classes = test_ds.meta.n_classes
            if entry["source"] == "synthetic":
                append_jsonl(exposure_log, {"at": utc_now_iso(), "kind": "test_split_opened", "task": task, "source": "synthetic", "experiment_id": entry["experiment_id"], "manifest_hash": manifest["manifest_hash"]})
            run_cfg = entry["run_config"]
            ds = TransformedDataset(test_ds, Normalizer.from_stats(stats), task_type, augment=False, input_size=run_cfg.get("input_size"))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            log(f"{task}: evaluating the test split ONCE ({len(ds)} samples) with {entry['experiment_id']}")
            metrics = evaluate(model, loader, task_type, n_classes, class_names, device, ignore_index=ignore_index, exclude_from_diagnostic_miou=exclude, use_amp=False)
            metric_name = entry["metric"]["name"]
            rec_out.update(
                {
                    "status": "completed",
                    "evaluated_at": utc_now_iso(),
                    "n_test_samples": len(ds),
                    "metric": entry["metric"],
                    "test_metric": float(metrics[metric_name]),
                    "val_metric_at_selection": entry["best_val_metric"],
                    "test_metrics": {k: v for k, v in metrics.items() if k not in ("conventions",)},
                    "checkpoint_sha256": sha,
                    "model_version": entry.get("model_version"),
                    "environment": capture_environment(),
                }
            )
            log(f"{task}: test {metric_name} = {rec_out['test_metric']:.4f} (validation at selection: {entry['best_val_metric']:.4f})")
        except Exception as exc:
            rec_out.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:800], "finished_at": utc_now_iso()})
            log(f"{task}: FAILED: {rec_out['error']}")
        results["tasks"][task] = rec_out
        results["updated_at"] = utc_now_iso()
        write_json(results_path(home), results)
        append_jsonl(home.final_eval / "events.jsonl", {"at": utc_now_iso(), "task": task, "status": rec_out["status"], "test_metric": rec_out.get("test_metric"), "error": rec_out.get("error")})
    return results
