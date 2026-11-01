"""Identifier generation and validation."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_RE = re.compile(r"^m-[A-Za-z0-9-]{1,40}$")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}-{utc_stamp()}-{secrets.token_hex(3)}"


def validate_id(value: str, kind: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value) or ".." in value:
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def validate_task_name(value: str) -> str:
    if not isinstance(value, str) or not _TASK_RE.match(value):
        raise ValueError(f"invalid task name: {value!r}")
    return value
