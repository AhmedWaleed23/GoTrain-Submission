"""Configuration objects, JSON loading and validation.

Task configs live in configs/tasks/<task>.json, allow-lists and ranges in
configs/limits.json, budgets in configs/budget.json. Validation happens in Python,
never in the LLM prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from gotrain.util.io import read_json

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
TASK_TYPES = ("classification", "segmentation")


@dataclass
class TaskConfig:
    name: str
    task_type: str
    benchmark: str
    hf_filename: str
    selection_metric: str
    n_classes: int
    class_names: list[str]
    in_channels: int
    patch_size: list[int]
    rgb_indices: list[int]
    ignore_index: int | None = None
    diagnostic_exclude_classes: list[int] = field(default_factory=list)
    input_size: int | None = None
    band_order_note: str = ""
    notes: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"{self.name}: task_type must be one of {TASK_TYPES}")
        if len(self.class_names) != self.n_classes:
            raise ValueError(f"{self.name}: {len(self.class_names)} class names for {self.n_classes} classes")
        if len(self.rgb_indices) != 3 or any(i < 0 or i >= self.in_channels for i in self.rgb_indices):
            raise ValueError(f"{self.name}: rgb_indices must be 3 valid channel positions")
        if self.ignore_index is not None and 0 <= self.ignore_index < self.n_classes:
            raise ValueError(f"{self.name}: ignore_index must lie outside [0, n_classes)")

    @property
    def dataset_dir_name(self) -> str:
        return self.name


def _from_dict(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def load_task_config(name: str, config_dir: Path | None = None) -> TaskConfig:
    path = (config_dir or CONFIG_DIR) / "tasks" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no task config for {name!r} at {path}")
    return _from_dict(TaskConfig, read_json(path))


def list_task_configs(config_dir: Path | None = None) -> list[str]:
    d = (config_dir or CONFIG_DIR) / "tasks"
    return sorted(p.stem for p in d.glob("m-*.json"))


@dataclass
class Limits:
    """Hard bounds enforced in code for every run and every agent tool call."""

    allowed_tasks: list[str]
    allowed_models: list[str]
    lr_min: float = 1e-5
    lr_max: float = 1e-1
    batch_size_min: int = 2
    batch_size_max: int = 256
    epochs_min: int = 1
    epochs_max: int = 200
    max_minutes_per_run: float = 120.0
    weight_decay_max: float = 0.1
    subset_fraction_min: float = 0.02
    max_tool_calls_per_session: int = 60
    max_experiments_per_session: int = 12
    max_session_minutes: float = 240.0
    max_retries_per_run: int = 2
    allowed_input_sizes: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512])

    @staticmethod
    def load(config_dir: Path | None = None) -> "Limits":
        return _from_dict(Limits, read_json((config_dir or CONFIG_DIR) / "limits.json"))


@dataclass
class RunConfig:
    """One training run. Validated against Limits before anything is launched."""

    task: str
    model: str
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    max_minutes: float = 20.0
    weight_decay: float = 1e-4
    seed: int = 0
    augment: bool = True
    precision: str = "auto"
    subset_fraction: float = 1.0
    screening: bool = False
    input_size: int | None = None
    num_workers: int = 2
    early_stopping_patience: int = 5
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    tag: str = "manual"
    hypothesis: str = ""
    rationale: str = ""
    expected_tradeoff: str = ""
    parent_experiment: str | None = None
    val_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RunConfig":
        return _from_dict(RunConfig, dict(data))

    def validate(self, limits: Limits, task_cfg: TaskConfig | None = None) -> list[str]:
        """Return a list of problems (empty when valid)."""
        problems = []
        if self.task not in limits.allowed_tasks:
            problems.append(f"task {self.task!r} not in allowed tasks {limits.allowed_tasks}")
        if self.model not in limits.allowed_models:
            problems.append(f"model {self.model!r} not in allowed models {limits.allowed_models}")
        if not (limits.lr_min <= self.lr <= limits.lr_max):
            problems.append(f"lr {self.lr} outside [{limits.lr_min}, {limits.lr_max}]")
        if not (limits.batch_size_min <= self.batch_size <= limits.batch_size_max):
            problems.append(f"batch_size {self.batch_size} outside [{limits.batch_size_min}, {limits.batch_size_max}]")
        if not (limits.epochs_min <= self.epochs <= limits.epochs_max):
            problems.append(f"epochs {self.epochs} outside [{limits.epochs_min}, {limits.epochs_max}]")
        if not (0 < self.max_minutes <= limits.max_minutes_per_run):
            problems.append(f"max_minutes {self.max_minutes} outside (0, {limits.max_minutes_per_run}]")
        if not (0 <= self.weight_decay <= limits.weight_decay_max):
            problems.append(f"weight_decay {self.weight_decay} outside [0, {limits.weight_decay_max}]")
        if not (limits.subset_fraction_min <= self.subset_fraction <= 1.0):
            problems.append(f"subset_fraction {self.subset_fraction} outside [{limits.subset_fraction_min}, 1]")
        if self.subset_fraction < 1.0 and not self.screening:
            problems.append("subset_fraction < 1 requires screening=true (partial-data runs are never ranked with full runs)")
        if self.precision not in ("auto", "fp32", "fp16"):
            problems.append(f"precision {self.precision!r} must be auto, fp32 or fp16")
        if self.scheduler not in ("cosine", "none"):
            problems.append(f"scheduler {self.scheduler!r} must be cosine or none")
        if self.input_size is not None and self.input_size not in limits.allowed_input_sizes:
            problems.append(f"input_size {self.input_size} not in {limits.allowed_input_sizes}")
        if not (0 <= self.label_smoothing < 0.5):
            problems.append("label_smoothing must be in [0, 0.5)")
        if self.early_stopping_patience < 1:
            problems.append("early_stopping_patience must be >= 1")
        if self.num_workers < 0 or self.num_workers > 16:
            problems.append("num_workers must be in [0, 16]")
        if self.val_limit is not None and self.val_limit < 1:
            problems.append("val_limit must be >= 1 when set")
        if task_cfg is not None:
            from gotrain.models.factory import MODEL_SPECS

            spec = MODEL_SPECS.get(self.model)
            if spec and spec["task_type"] != task_cfg.task_type:
                problems.append(f"model {self.model!r} is for {spec['task_type']} but {self.task} is {task_cfg.task_type}")
        return problems


@dataclass
class Budget:
    """Default compute budgets. Edit configs/budget.json after measuring with `smoke`."""

    baseline_epochs: int = 8
    baseline_max_minutes: float = 15.0
    agent_session_minutes: float = 60.0
    agent_max_experiments: int = 6
    agent_run_max_minutes: float = 15.0
    screening_subset_fraction: float = 0.25
    screening_epochs: int = 3
    finalist_epochs: int = 12
    finalist_max_minutes: float = 25.0

    @staticmethod
    def load(config_dir: Path | None = None) -> "Budget":
        return _from_dict(Budget, read_json((config_dir or CONFIG_DIR) / "budget.json"))
