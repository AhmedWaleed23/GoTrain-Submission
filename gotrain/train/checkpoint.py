"""Checkpoint files: atomic writes, SHA-256, and enough metadata to rebuild inference."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from gotrain.util.io import atomic_write_bytes, sha256_file


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    buf = io.BytesIO()
    torch.save(payload, buf)
    atomic_write_bytes(path, buf.getvalue())
    return sha256_file(path)


def load_checkpoint(path: Path, map_location: str = "cpu", expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(f"checkpoint hash mismatch for {path}: expected {expected_sha256[:12]}..., got {actual[:12]}...")
    return torch.load(path, map_location=map_location, weights_only=False)
