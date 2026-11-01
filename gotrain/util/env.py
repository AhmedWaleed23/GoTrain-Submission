"""Environment capture for experiment records."""

from __future__ import annotations

import platform
import sys
from importlib import metadata


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "packages": {
            name: _version(name)
            for name in ("numpy", "rasterio", "geobench", "huggingface_hub", "strands-agents", "h5py")
        },
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "total_memory_gib": round(props.total_memory / 1024**3, 2)}
    return info


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
