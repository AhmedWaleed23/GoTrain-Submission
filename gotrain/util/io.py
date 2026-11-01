"""Atomic file helpers, JSON helpers and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def dumps_json(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, default=_json_default)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, dumps_json(obj) + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = dumps_json(obj, indent=None) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def iter_jsonl(path: Path) -> Iterator[Any]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn trailing line from an interrupted write is skipped, not fatal.
                continue


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(dumps_json(obj, indent=None).encode("utf-8"))


def ensure_within(root: Path, candidate: Path) -> Path:
    """Return the resolved candidate if it is inside root, else raise ValueError."""
    root_r = Path(root).resolve()
    cand_r = Path(candidate).resolve()
    try:
        cand_r.relative_to(root_r)
    except ValueError as exc:
        raise ValueError(f"path {candidate} escapes storage root {root}") from exc
    return cand_r
