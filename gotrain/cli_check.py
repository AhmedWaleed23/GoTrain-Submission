"""The read-only submission-check subcommand (kept separate so it can be reverted alone)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gotrain.paths import Home

REQUIRED_TASKS = ("m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake")


def _tasks(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    from gotrain.util.ids import validate_task_name

    return [t.strip() if t.strip() == "synthetic" else validate_task_name(t.strip()) for t in arg.split(",") if t.strip()]


def cmd_submission_check(args) -> int:
    from gotrain.submission_check import format_report, run_check
    from gotrain.util.io import write_json

    try:
        home = Home.from_env(getattr(args, "home", None))
        result = run_check(home, tasks=_tasks(args.task), presentation_path=Path(args.presentation) if args.presentation else None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    if args.json:
        write_json(Path(args.json), result)
        print(f"json written to {args.json}", file=sys.stderr)
    return int(result["exit_code"])



def register_check(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("submission-check", help="read-only readiness checklist (exit 0 = no blocking problems, 1 = blocking problems, 2 = usage error)")
    s.add_argument("--task", default=None, help="comma list of tasks (default: the four required tasks)")
    s.add_argument("--json", default=None, help="also write the machine-readable result to this path")
    s.add_argument("--presentation", default=None, help="path to the presentation/demo file to check for existence")
    s.set_defaults(fn=cmd_submission_check)
