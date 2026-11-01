"""Subcommands for the agent, registry, freeze, final evaluation and report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gotrain.config import Budget, Limits
from gotrain.paths import Home, geobench_dir
from gotrain.util.io import dumps_json

REQUIRED_TASKS = ("m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake")


def _home(args) -> Home:
    return Home.from_env(getattr(args, "home", None)).ensure()


def _tasks(arg: str) -> list[str]:
    from gotrain.util.ids import validate_task_name

    return list(REQUIRED_TASKS) if arg == "all" else [t.strip() if t.strip() == "synthetic" else validate_task_name(t.strip()) for t in arg.split(",") if t.strip()]


def _print(obj) -> None:
    print(dumps_json(obj))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_agent(args) -> int:
    from gotrain.agent.runner import run_agent
    from gotrain.artifacts.store import FilesystemArtifactStore

    home = _home(args)
    budget = Budget.load()
    if args.session_minutes is not None:
        budget.agent_session_minutes = args.session_minutes
    if args.max_experiments is not None:
        budget.agent_max_experiments = args.max_experiments
    if args.run_max_minutes is not None:
        budget.agent_run_max_minutes = args.run_max_minutes
    source = "synthetic" if args.task == "synthetic" else "geobench"
    tasks = ["synthetic"] if source == "synthetic" else _tasks(args.task)
    if args.scripted:
        from gotrain.agent.fake_model import ReactivePolicy, ScriptedModel

        model = ScriptedModel(ReactivePolicy(tasks[0], "cnn_small" if source == "synthetic" else ("cnn_small" if tasks[0] in ("m-eurosat", "m-pv4ger") else "unet_small"), epochs=args.scripted_epochs, max_minutes=budget.agent_run_max_minutes))
        model_id = "scripted-fake"
    else:
        from gotrain.agent.bedrock import make_bedrock_model

        model = make_bedrock_model(model_id=args.model_id, region=args.region)
        model_id = args.model_id
    state = run_agent(
        home, tasks, model, budget, Limits.load(), device=args.device, geobench_root=geobench_dir(args.geobench_dir), source=source,
        session_id=args.resume_session, max_rounds=args.max_rounds, turns_per_round=args.turns_per_round, registry=FilesystemArtifactStore(home, geobench_root=geobench_dir(args.geobench_dir)), model_id=model_id, log=_log,
    )
    _print({k: state[k] for k in ("session_id", "status", "tool_calls", "experiments_launched", "rounds", "finished_tasks", "errors")})
    print(f"session files: {home.agent / 'sessions' / state['session_id']}", file=sys.stderr)
    return 0


def cmd_sessions(args) -> int:
    from gotrain.agent.session import list_sessions

    _print([{k: s.get(k) for k in ("session_id", "status", "tasks", "tool_calls", "experiments_launched", "rounds", "finished_tasks", "started_at", "elapsed_seconds")} for s in list_sessions(_home(args))])
    return 0


def cmd_store_model(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore

    m = FilesystemArtifactStore(_home(args), geobench_root=geobench_dir(args.geobench_dir)).store_model(args.experiment_id, note=args.note or "", parent_version=args.parent)
    _print({k: m[k] for k in ("version_id", "task", "experiment_id", "checkpoint_sha256", "metric", "storage_uri")})
    return 0


def cmd_list_versions(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore

    store = FilesystemArtifactStore(_home(args))
    _print({"versions": store.list_versions(task=args.task), "promoted": {t: store.promoted(t) for t in sorted({v["task"] for v in store.list_versions()})}})
    return 0


def cmd_retrieve_model(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore

    out = FilesystemArtifactStore(_home(args)).retrieve_model(args.version_id, dest=Path(args.dest) if args.dest else None)
    _print({k: out[k] for k in ("version_id", "checkpoint_path", "sha256", "sha256_verified", "copied")})
    return 0


def cmd_verify_model(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore
    from gotrain.artifacts.verify import verify_version

    store = FilesystemArtifactStore(_home(args), geobench_root=geobench_dir(args.geobench_dir))
    ids = [args.version_id] if args.version_id != "all" else [v["version_id"] for v in store.list_versions()]
    results = [verify_version(store, vid, device=args.device or "cpu", geobench_root=geobench_dir(args.geobench_dir)) for vid in ids]
    _print(results)
    return 0 if all(r.get("verified") for r in results) else 1


def cmd_promote_model(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore

    _print(FilesystemArtifactStore(_home(args)).promote_model(args.version_id, args.rationale, by="manual"))
    return 0


def cmd_register_dataset(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore
    from gotrain.config import load_task_config
    from gotrain.data.geobench_adapter import dataset_dir

    store = FilesystemArtifactStore(_home(args))
    out = {}
    for t in _tasks(args.task):
        out[t] = store.store_dataset(t, dataset_dir(load_task_config(t), geobench_dir(args.geobench_dir)), mode=args.mode)
    _print(out)
    return 0


def cmd_retrieve_dataset(args) -> int:
    from gotrain.artifacts.store import FilesystemArtifactStore

    _print(FilesystemArtifactStore(_home(args)).retrieve_dataset(args.task, dest=Path(args.dest) if args.dest else None))
    return 0


def cmd_freeze(args) -> int:
    from gotrain.final_eval.freeze import freeze_selection

    overrides = {}
    for item in args.experiment or []:
        task, _, exp = item.partition("=")
        if not exp:
            raise SystemExit(f"--experiment expects task=experiment_id, got {item!r}")
        overrides[task] = exp
    manifest = freeze_selection(_home(args), _tasks(args.task), overrides=overrides or None, by=args.by, replace=args.replace, log=_log, geobench_root=geobench_dir(args.geobench_dir))
    _print({"frozen_at": manifest["frozen_at"], "manifest_hash": manifest["manifest_hash"], "tasks": {t: {"experiment_id": e["experiment_id"], "best_val_metric": e["best_val_metric"], "model_version": e["model_version"]} for t, e in manifest["tasks"].items()}})
    return 0


def cmd_final_eval(args) -> int:
    from gotrain.final_eval.evaluate_test import final_evaluate

    res = final_evaluate(
        _home(args), manifest_file=Path(args.manifest) if args.manifest else None, tasks=_tasks(args.task) if args.task else None, device=args.device,
        geobench_root=geobench_dir(args.geobench_dir), acknowledge=args.i_understand_this_uses_the_test_split, batch_size=args.batch_size, num_workers=args.num_workers, log=_log,
    )
    _print({t: {k: r.get(k) for k in ("status", "test_metric", "val_metric_at_selection", "n_test_samples", "evaluated_at", "error")} for t, r in res["tasks"].items()})
    return 0 if all(r.get("status") == "completed" for r in res["tasks"].values()) else 1


def cmd_report(args) -> int:
    from gotrain.report.html import render_dashboard

    if args.fixture:
        from gotrain.report.fixture import build_fixture_home

        home = build_fixture_home(Path(args.fixture))
    else:
        home = _home(args)
    out = render_dashboard(home, Path(args.out) if args.out else None)
    print(out)
    return 0


def cmd_s3_smoke(args) -> int:
    from gotrain.artifacts.s3 import s3_smoke

    _print(s3_smoke(args.bucket, args.prefix))
    return 0


def cmd_review_build(args) -> int:
    from gotrain.review.build import build_review, default_out_dir, load_review, summarize
    from gotrain.review.model_source import from_bundle, from_home
    from gotrain.review.page import render_review_page

    home = _home(args)
    device = args.device or "cpu"
    if args.bundle:
        source = from_bundle(Path(args.bundle), device=device)
        if args.task and args.task != source.task:
            raise SystemExit(f"bundle is for {source.task}, not {args.task}")
    else:
        if not args.task:
            raise SystemExit("--task is required (or pass --bundle)")
        source = from_home(home, args.task, device=device, experiment_id=args.experiment)
    _log(f"model: {source.experiment_id} ({source.model_version or 'no registry version'}), input size {source.input_size}, device {device}")
    if source.reference_check:
        _log(f"reference check: max abs diff {source.reference_check['max_abs_diff']:.2e}, within tolerance {source.reference_check['within_tolerance']}")
    out_dir = Path(args.out) if args.out else default_out_dir(home, source, args.limit)
    if args.limit:
        _log(f"PARTIAL build: only the first {args.limit} validation images; do not read full-split conclusions from it")
    res = build_review(home, source, out_dir, device=device, batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit, geobench_root=geobench_dir(args.geobench_dir), thumb_size=args.thumb_size, log=_log, force=args.force)
    review = load_review(out_dir)
    page = render_review_page(review, max_embedded=args.max_embedded)
    _log(summarize(review))
    _print({**{k: res[k] for k in ("out_dir", "n_images", "n_images_in_split", "partial", "seconds", "split_miou_fp32", "metric_comparison")}, "page": str(page), "page_bytes": page.stat().st_size})
    return 0


def cmd_review_page(args) -> int:
    from gotrain.review.build import load_review, summarize
    from gotrain.review.page import render_review_page

    review = load_review(Path(args.review_dir))
    page = render_review_page(review, out_path=Path(args.out) if args.out else None, max_embedded=args.max_embedded)
    _log(summarize(review))
    print(page)
    return 0


def cmd_review_notes(args) -> int:
    """Validate an exported notes file against a review output and convert it to CSV."""
    from gotrain.review.build import load_review
    from gotrain.review.notes import run_identity, to_csv, validate_import
    from gotrain.util.io import read_json

    review = load_review(Path(args.review_dir))
    payload = read_json(Path(args.notes))
    entries = validate_import(payload, run_identity(review["provenance"]))
    known = {it["sample_id"] for it in review["items"]}
    unknown = [e["sample_id"] for e in entries if e["sample_id"] not in known]
    decided = sum(1 for e in entries if e["decision"])
    if args.csv:
        Path(args.csv).write_text(to_csv(payload), encoding="utf-8")
    _print({"valid": True, "entries": len(entries), "with_decision": decided, "unknown_sample_ids": unknown[:10], "csv": args.csv})
    return 0 if not unknown else 1


def register_extra(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("agent", help="run the autonomous Sonnet agent (Bedrock) on one or more tasks")
    s.add_argument("--task", default="all", help="task, comma list, 'all', or 'synthetic' (offline test)")
    s.add_argument("--session-minutes", type=float, default=None, help="override configs/budget.json agent_session_minutes")
    s.add_argument("--max-experiments", type=int, default=None)
    s.add_argument("--run-max-minutes", type=float, default=None)
    s.add_argument("--max-rounds", type=int, default=8)
    s.add_argument("--turns-per-round", type=int, default=16)
    s.add_argument("--model-id", default="eu.anthropic.claude-sonnet-5-5")
    s.add_argument("--region", default="eu-central-1")
    s.add_argument("--resume-session", default=None, help="continue a previous session id")
    s.add_argument("--scripted", action="store_true", help="use the offline scripted fake model instead of Bedrock (tests/demo only)")
    s.add_argument("--scripted-epochs", type=int, default=2)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_agent)

    sub.add_parser("sessions", help="list agent sessions").set_defaults(fn=cmd_sessions)

    s = sub.add_parser("store-model", help="store an experiment's best checkpoint as an immutable model version")
    s.add_argument("experiment_id")
    s.add_argument("--note", default="")
    s.add_argument("--parent", default=None, help="parent version id")
    s.set_defaults(fn=cmd_store_model)

    s = sub.add_parser("list-versions", help="list stored model versions")
    s.add_argument("--task", default=None)
    s.set_defaults(fn=cmd_list_versions)

    s = sub.add_parser("retrieve-model", help="checksum-verified retrieval of a model version")
    s.add_argument("version_id")
    s.add_argument("--dest", default=None, help="copy the version files to this directory")
    s.set_defaults(fn=cmd_retrieve_model)

    s = sub.add_parser("verify-model", help="rebuild inference from a stored version and reproduce its reference predictions")
    s.add_argument("version_id", help="version id or 'all'")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_verify_model)

    s = sub.add_parser("promote-model", help="mark a version as the promoted one for its task")
    s.add_argument("version_id")
    s.add_argument("--rationale", required=True)
    s.set_defaults(fn=cmd_promote_model)

    s = sub.add_parser("register-dataset", help="register a downloaded task in the artifact store (reference or copy)")
    s.add_argument("--task", required=True)
    s.add_argument("--mode", default="reference", choices=["reference", "copy"])
    s.set_defaults(fn=cmd_register_dataset)

    s = sub.add_parser("retrieve-dataset", help="locate (and optionally copy) a registered dataset after identity verification")
    s.add_argument("task")
    s.add_argument("--dest", default=None)
    s.set_defaults(fn=cmd_retrieve_dataset)

    s = sub.add_parser("freeze", help="freeze the selected experiment per task into selection/manifest.json")
    s.add_argument("--task", default="all")
    s.add_argument("--experiment", action="append", help="override: task=experiment_id (repeatable)")
    s.add_argument("--by", default="manual")
    s.add_argument("--replace", action="store_true", help="replace an existing manifest (refused once test results exist)")
    s.set_defaults(fn=cmd_freeze)

    s = sub.add_parser("final-eval", help="evaluate each frozen task on its TEST split exactly once")
    s.add_argument("--manifest", default=None)
    s.add_argument("--task", default=None, help="subset of frozen tasks (default: all in manifest)")
    s.add_argument("--i-understand-this-uses-the-test-split", action="store_true")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_final_eval)

    s = sub.add_parser("report", help="render the self-contained HTML dashboard from saved records")
    s.add_argument("--out", default=None, help="output path (default <home>/reports/dashboard.html)")
    s.add_argument("--fixture", default=None, help="build a clearly-labelled synthetic fixture home at this path and render it")
    s.set_defaults(fn=cmd_report)


    s = sub.add_parser("review-build", help="run validation inference with the selected model and build the review page (valid split only)")
    s.add_argument("--task", default=None, help="segmentation task, e.g. m-pv4ger-seg (uses its selected experiment)")
    s.add_argument("--experiment", default=None, help="explicit experiment id instead of the selected pointer")
    s.add_argument("--bundle", default=None, help="portable model bundle directory (manifest.json, checkpoint.pt, normalization.json, reference.npz)")
    s.add_argument("--device", default=None, help="cuda or cpu (default cpu)")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--limit", type=int, default=None, help="smoke mode: only the first N validation images (output is labelled PARTIAL)")
    s.add_argument("--out", default=None, help="output directory (default <home>/review/<task>/<version>[_limitN])")
    s.add_argument("--thumb-size", type=int, default=256)
    s.add_argument("--max-embedded", type=int, default=None, help="embed only the first N queue images in the page (disclosed on the page)")
    s.add_argument("--force", action="store_true", help="overwrite an output directory built for a different model/limit")
    s.set_defaults(fn=cmd_review_build)

    s = sub.add_parser("review-page", help="re-render review.html from an existing review output directory")
    s.add_argument("review_dir")
    s.add_argument("--out", default=None)
    s.add_argument("--max-embedded", type=int, default=None)
    s.set_defaults(fn=cmd_review_page)

    s = sub.add_parser("review-notes", help="validate an exported reviewer notes JSON against a review output and convert to CSV")
    s.add_argument("review_dir")
    s.add_argument("--notes", required=True)
    s.add_argument("--csv", default=None)
    s.set_defaults(fn=cmd_review_notes)


    s = sub.add_parser("s3-smoke", help="optional: round-trip a tiny object through S3 to prove bucket access")
    s.add_argument("--bucket", required=True)
    s.add_argument("--prefix", default="gotrain")
    s.set_defaults(fn=cmd_s3_smoke)
