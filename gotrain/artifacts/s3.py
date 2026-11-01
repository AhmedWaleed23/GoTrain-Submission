"""Optional S3 backend. Not executed in this handoff (no bucket, no credentials here).

It reuses the filesystem store for local materialization and mirrors each version
directory to s3://<bucket>/<prefix>/models/<version_id>/ with the manifest's SHA-256
recorded as object metadata. Nothing here creates buckets or asks for keys; boto3 is
imported lazily and credentials come from the SageMaker role.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.paths import Home
from gotrain.util.io import sha256_file
from gotrain.util.timing import utc_now_iso

VERSION_FILES = ("checkpoint.pt", "manifest.json", "normalization.json", "reference.npz")


def _client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on target env
        raise RuntimeError("boto3 is not installed; S3 backend unavailable") from exc
    return boto3.client("s3")


def _safe_prefix(prefix: str) -> str:
    prefix = prefix.strip("/")
    if not prefix or ".." in prefix or prefix.startswith("/"):
        raise ValueError("prefix must be a non-empty relative key prefix")
    return prefix


class S3ArtifactStore(FilesystemArtifactStore):
    backend = "s3"

    def __init__(self, home: Home, bucket: str, prefix: str = "gotrain") -> None:
        super().__init__(home)
        if not bucket or "/" in bucket:
            raise ValueError("bucket must be a bare bucket name")
        self.bucket = bucket
        self.prefix = _safe_prefix(prefix)
        self.s3 = _client()

    def _key(self, version_id: str, name: str) -> str:
        return f"{self.prefix}/models/{version_id}/{name}"

    def store_model(self, experiment_id: str, note: str = "", parent_version: str | None = None) -> dict:
        manifest = super().store_model(experiment_id, note=note, parent_version=parent_version)
        vdir = self.version_dir(manifest["version_id"])
        for name in VERSION_FILES:
            path = vdir / name
            if path.exists():
                self.s3.upload_file(str(path), self.bucket, self._key(manifest["version_id"], name), ExtraArgs={"Metadata": {"sha256": sha256_file(path)}})
        manifest["storage_uri"] = f"s3://{self.bucket}/{self._key(manifest['version_id'], 'checkpoint.pt')}"
        idx = self._index()
        idx["versions"][manifest["version_id"]]["storage_uri"] = manifest["storage_uri"]
        idx["versions"][manifest["version_id"]]["s3_uploaded_at"] = utc_now_iso()
        self._write_index(idx)
        self._event("s3_upload", version_id=manifest["version_id"], uri=manifest["storage_uri"])
        return manifest

    def retrieve_model(self, version_id: str, dest: Path | None = None) -> dict:
        manifest = self.get_version(version_id)
        vdir = self.version_dir(version_id)
        if not (vdir / "checkpoint.pt").exists():
            vdir.mkdir(parents=True, exist_ok=True)
            for name in VERSION_FILES:
                try:
                    self.s3.download_file(self.bucket, self._key(version_id, name), str(vdir / name))
                except Exception as exc:
                    if name in ("checkpoint.pt", "manifest.json"):
                        raise RuntimeError(f"S3 download failed for {name}: {exc}") from exc
            if sha256_file(vdir / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                raise ValueError("downloaded checkpoint hash mismatch; refusing to use it")
            self._event("s3_download", version_id=version_id, uri=f"s3://{self.bucket}/{self._key(version_id, 'checkpoint.pt')}")
        return super().retrieve_model(version_id, dest=dest)


def s3_smoke(bucket: str, prefix: str = "gotrain") -> dict:
    """Round-trip a tiny object: put, get, compare hashes, delete. Proves bucket access."""
    prefix = _safe_prefix(prefix)
    s3 = _client()
    key = f"{prefix}/_smoke/{utc_now_iso().replace(':', '')}.txt"
    payload = f"gotrain s3 smoke {utc_now_iso()}\n".encode()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "up.txt"
        src.write_bytes(payload)
        s3.upload_file(str(src), bucket, key)
        dst = Path(td) / "down.txt"
        s3.download_file(bucket, key, str(dst))
        ok = sha256_file(src) == sha256_file(dst)
        s3.delete_object(Bucket=bucket, Key=key)
    return {"bucket": bucket, "key": key, "roundtrip_ok": ok, "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")}
