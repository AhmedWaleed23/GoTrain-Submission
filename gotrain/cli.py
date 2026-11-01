"""Command line interface: python -m gotrain <command> [options]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gotrain.config import Budget, Limits, RunConfig, list_task_configs, load_task_config
from gotrain.paths import Home, geobench_dir
from gotrain.util.io import dumps_json

REQUIRED_TASKS = ("m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake")


def _home(args) -> Home:
    return Home.from_env(getattr(args, "home", None)).ensure()


def _tasks(arg: str) -> list[str]:
    from gotrain.util.ids import validate_task_name

    if arg == "all":
        return list(REQUIRED_TASKS)
    return [t.strip() if t.strip() == "synthetic" else validate_task_name(t.strip()) for t in arg.split(",") if t.strip()]


def _print(obj) -> None:
    print(dumps_json(obj))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_env_check(args) -> int:
    from gotrain.util.env import capture_environment

    info = capture_environment()
    info["geobench_dir"] = str(geobench_dir(args.geobench_dir))
    info["gotrain_home"] = str(_home(args).root)
    try:
        import strands  # noqa: F401

        info["strands_import"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on target env
        info["strands_import"] = f"failed: {exc}"
    try:
        import geobench  # noqa: F401

        info["geobench_import"] = "ok"
    except Exception as exc:  # pragma: no cover
        info["geobench_import"] = f"failed: {exc}"
    from gotrain.data.geobench_adapter import dataset_present

    info["datasets_present"] = {t: dataset_present(load_task_config(t), geobench_dir(args.geobench_dir)) for t in list_task_configs()}
    _print(info)
    problems = []
    if not info["cuda_available"]:
        problems.append("CUDA not available: training will run on CPU (fine for tests, too slow for real runs)")
    if info["strands_import"] != "ok":
        problems.append("strands not importable: the agent command will not work")
    for p in problems:
        print("WARNING:", p, file=sys.stderr)
    return 0


def cmd_test(args) -> int:
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Install it with:\n  python -m pip install -c requirements-constraints.txt pytest", file=sys.stderr)
        return 2

    extra = ["-q"] + (["-k", args.k] if args.k else [])
    return int(pytest.main([str(Path(__file__).resolve().parents[1] / "tests")] + extra))


def cmd_download(args) -> int:
    from gotrain.data.download import download_task

    root = geobench_dir(args.geobench_dir)
    out = []
    for t in _tasks(args.task):
        out.append(download_task(load_task_config(t), root, delete_zip=args.delete_zip, verify_remote_hash=not args.no_hash_check))
    _print(out)
    return 0


def cmd_inspect(args) -> int:
    from gotrain.data.geobench_adapter import inspect_task

    root = geobench_dir(args.geobench_dir)
    reports = {}
    for t in _tasks(args.task):
        rep = inspect_task(load_task_config(t), root, max_label_samples=args.max_label_samples)
        if args.fit_normalization and rep.get("present"):
            from gotrain.data.geobench_adapter import open_split
            from gotrain.experiments.runner import normalization_for_task

            stats, npath = normalization_for_task(_home(args), load_task_config(t), open_split(load_task_config(t), "train", root), _log)
            rep["normalization"] = {"path": str(npath), "stats_hash": stats.get("stats_hash"), "n_samples_used": stats.get("n_samples_used")}
        reports[t] = rep
        if args.save:
            home = _home(args)
            from gotrain.util.io import write_json

            write_json(home.root / "inspect" / f"{t}.json", rep)
    _print(reports)
    return 0


def _run_config_from_args(args, task: str, tag: str) -> RunConfig:
    return RunConfig(
        task=task,
        model=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_minutes=args.max_minutes,
        weight_decay=args.weight_decay,
        seed=args.seed,
        augment=not args.no_augment,
        precision=args.precision,
        subset_fraction=args.subset_fraction,
        screening=args.screening or args.subset_fraction < 1.0 or bool(args.val_limit),
        input_size=args.input_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.patience,
        tag=tag,
        hypothesis=args.hypothesis or "",
        rationale=args.rationale or "",
        val_limit=args.val_limit,
    )


def _add_run_args(p: argparse.ArgumentParser, defaults: dict | None = None) -> None:
    d = {"model": None, "lr": 1e-3, "batch_size": 32, "epochs": 10, "max_minutes": 20.0, "weight_decay": 1e-4, "seed": 0,
         "precision": "auto", "subset_fraction": 1.0, "input_size": None, "num_workers": 2, "patience": 5, "val_limit": None}
    d.update(defaults or {})
    p.add_argument("--model", default=d["model"], help="model name from configs/limits.json allowed_models (default: baseline model for the task type)")
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--max-minutes", type=float, default=d["max_minutes"], help="wall-clock budget for this run")
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--precision", default=d["precision"], choices=["auto", "fp32", "fp16"])
    p.add_argument("--subset-fraction", type=float, default=d["subset_fraction"], help="<1 marks the run as screening (partial data)")
    p.add_argument("--screening", action="store_true", help="label as a screening run (never ranked with full runs)")
    p.add_argument("--input-size", type=int, default=d["input_size"], help="resize images (and masks, nearest) to this size")
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    p.add_argument("--patience", type=int, default=d["patience"], help="early stopping patience in epochs")
    p.add_argument("--val-limit", type=int, default=d["val_limit"], help="evaluate only the first N validation samples (screening only)")
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--rationale", default=None)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--allow-duplicate", action="store_true", help="retrain even if an identical config already completed")


def _default_model(task: str) -> str:
    return "cnn_small" if load_task_config(task).task_type == "classification" else "unet_small"


def cmd_run(args) -> int:
    from gotrain.experiments.runner import run_experiment

    home = _home(args)
    source = "synthetic" if args.task == "synthetic" else "geobench"
    if source == "synthetic":
        model = args.model or "cnn_small"
    else:
        model = args.model or _default_model(args.task)
    args.model = model
    run = _run_config_from_args(args, args.task, args.tag)
    rec = run_experiment(home, run, source=source, device=args.device, geobench_root=geobench_dir(args.geobench_dir), allow_duplicate=args.allow_duplicate, log=_log)
    _print(_summary(rec))
    return 0 if rec.get("status") == "completed" else 1


def cmd_smoke(args) -> int:
    """Short real-data run per task to measure speed and prove the pipeline. Train/valid only."""
    from gotrain.experiments.runner import run_experiment

    home = _home(args)
    results = {}
    for t in _tasks(args.task):
        model = args.model or _default_model(t)
        run = RunConfig(
            task=t, model=model, lr=1e-3, batch_size=args.batch_size, epochs=args.epochs, max_minutes=args.max_minutes,
            subset_fraction=args.subset_fraction, screening=True, val_limit=args.val_limit, num_workers=args.num_workers,
            tag="smoke", hypothesis="pipeline smoke test; timing measurement only",
        )
        rec = run_experiment(home, run, source="geobench", device=args.device, geobench_root=geobench_dir(args.geobench_dir), allow_duplicate=True, log=_log)
        results[t] = _summary(rec)
    _print(results)
    return 0 if all(r["status"] == "completed" for r in results.values()) else 1


def cmd_resume(args) -> int:
    from gotrain.experiments.runner import resume_experiment

    rec = resume_experiment(_home(args), args.experiment_id, device=args.device, geobench_root=geobench_dir(args.geobench_dir), log=_log)
    _print(_summary(rec))
    return 0 if rec.get("status") == "completed" else 1


def _summary(rec: dict) -> dict:
    from gotrain.experiments.store import ExperimentStore

    s = ExperimentStore.compact_summary(None, rec)
    if rec.get("duplicate_of"):
        s["duplicate_of"] = rec["duplicate_of"]
    return s


def cmd_list(args) -> int:
    from gotrain.experiments.store import ExperimentStore

    store = ExperimentStore(_home(args))
    task = None if args.task in (None, "all") else args.task
    rows = [store.compact_summary(r) for r in store.list(task=task, include_screening=not args.no_screening)]
    if args.json:
        _print(rows)
        return 0
    cols = ["experiment_id", "task", "status", "tag", "screening", "model", "lr", "batch_size", "epochs_completed", "best_val_metric", "stop_reason", "wall_minutes"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r.get(c), float) and c == "best_val_metric" else str(r.get(c))) for c in cols))
    for t in sorted({r["task"] for r in rows}):
        best = store.best(t)
        if best:
            print(f"best[{t}] = {best['experiment_id']} ({best['metric']['name']}={best['best_val_metric']:.4f}, by {best['selected_by']})")
    return 0


def cmd_show(args) -> int:
    from gotrain.experiments.store import ExperimentStore, list_events

    store = ExperimentStore(_home(args))
    rec = store.read(args.experiment_id)
    if args.events:
        _print(list_events(_home(args), args.experiment_id))
    else:
        _print(rec)
    return 0


def cmd_baseline(args) -> int:
    from gotrain.agent.baseline import run_baselines

    home = _home(args)
    budget = Budget.load()
    out = run_baselines(home, _tasks(args.task), budget, device=args.device, geobench_root=geobench_dir(args.geobench_dir), epochs=args.epochs, max_minutes=args.max_minutes, num_workers=args.num_workers, seed=args.seed, log=_log)
    _print(out)
    return 0 if all(r.get("status") == "completed" for r in out.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gotrain", description="GoTrain: autonomous GEO-Bench training agent")
    p.add_argument("--home", default=None, help="storage root (default: $GOTRAIN_HOME or ./gotrain_runs)")
    p.add_argument("--geobench-dir", default=None, help="GEO-Bench data root (default: $GEO_BENCH_DIR or ~/dataset/geobench)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env-check", help="print versions, GPU, data presence").set_defaults(fn=cmd_env_check)
    s = sub.add_parser("test", help="run the CPU test suite")
    s.add_argument("-k", default=None)
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("download", help="selectively download and verify one or more tasks")
    s.add_argument("--task", required=True, help="task name, comma list, or 'all' (the four required tasks)")
    s.add_argument("--delete-zip", action="store_true")
    s.add_argument("--no-hash-check", action="store_true")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("inspect", help="metadata, label validation and value ranges (train/valid only)")
    s.add_argument("--task", required=True)
    s.add_argument("--max-label-samples", type=int, default=1000, help="samples per split used for label statistics")
    s.add_argument("--fit-normalization", action="store_true", help="also fit and cache train-split normalization so budgeted runs do not pay for it")
    s.add_argument("--save", action="store_true", help="also write <home>/inspect/<task>.json")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("run", help="train one experiment with an explicit config")
    s.add_argument("--task", required=True, help="task name or 'synthetic'")
    s.add_argument("--tag", default="manual")
    _add_run_args(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("smoke", help="short real-data screening run per task to measure speed")
    s.add_argument("--task", default="all")
    s.add_argument("--model", default=None)
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-minutes", type=float, default=5.0)
    s.add_argument("--subset-fraction", type=float, default=0.1)
    s.add_argument("--val-limit", type=int, default=100)
    s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("resume", help="resume an interrupted experiment")
    s.add_argument("experiment_id")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("list", help="list experiments and best pointers")
    s.add_argument("--task", default=None)
    s.add_argument("--no-screening", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("show", help="print one experiment record")
    s.add_argument("experiment_id")
    s.add_argument("--events", action="store_true")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("baseline", help="fixed-config baseline for each task (no LLM)")
    s.add_argument("--task", default="all")
    s.add_argument("--epochs", type=int, default=None, help="override configs/budget.json baseline_epochs")
    s.add_argument("--max-minutes", type=float, default=None, help="override configs/budget.json baseline_max_minutes")
    s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_baseline)

    from gotrain.cli_check import register_check
    from gotrain.cli_extra import register_extra

    register_extra(sub)
    register_check(sub)
    from gotrain.completion import register_completion
    register_completion(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)
