"""Time helpers."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start
