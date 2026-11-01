"""Storage layout. Everything GoTrain writes lives under one home directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Home:
    root: Path

    @staticmethod
    def from_env(explicit: str | None = None) -> "Home":
        raw = explicit or os.environ.get("GOTRAIN_HOME") or "gotrain_runs"
        return Home(Path(raw).expanduser().resolve())

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def normalization(self) -> Path:
        return self.root / "normalization"

    @property
    def agent(self) -> Path:
        return self.root / "agent"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def selection(self) -> Path:
        return self.root / "selection"

    @property
    def final_eval(self) -> Path:
        return self.root / "final_eval"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    def ensure(self) -> "Home":
        for p in (
            self.experiments,
            self.normalization,
            self.agent,
            self.registry,
            self.selection,
            self.final_eval,
            self.reports,
            self.locks,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def geobench_dir(explicit: str | None = None) -> Path:
    raw = explicit or os.environ.get("GEO_BENCH_DIR") or str(Path("~/dataset/geobench").expanduser())
    return Path(raw).expanduser()
