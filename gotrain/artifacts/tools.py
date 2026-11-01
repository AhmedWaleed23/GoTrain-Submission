"""Registry tools exposed to the agent: bounded, validated, within the storage root."""

from __future__ import annotations

import json

try:
    from strands import tool
except ImportError:  # pragma: no cover
    def tool(fn=None, **_kw):  # type: ignore[misc]
        return fn if fn is not None else (lambda f: f)


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def build_registry_tools(rt) -> list:
    store = rt.registry
    session = rt.session

    @tool
    def store_model_version(experiment_id: str, note: str = "") -> str:
        """Store the best checkpoint of a completed experiment as an immutable model version in the registry (copies the checkpoint, records provenance and a reference prediction batch).

        Args:
            experiment_id: completed experiment id
            note: short note on why this version matters
        """
        try:
            m = store.store_model(experiment_id, note=note.strip()[:500])
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            return _json({"error": str(exc)})
        session.record_decision("store_model_version", version_id=m["version_id"], experiment_id=experiment_id, note=note.strip()[:500])
        return _json({"version_id": m["version_id"], "task": m["task"], "metric": m["metric"]["name"], "value": m["metric"]["value"], "checkpoint_sha256": m["checkpoint_sha256"][:16]})

    @tool
    def list_model_versions(task: str) -> str:
        """List stored model versions for a task with validation scores, training time and the promoted version.

        Args:
            task: task name
        """
        rows = store.list_versions(task=task)
        return _json({"task": task, "versions": rows, "promoted": store.promoted(task)})

    @tool
    def verify_model_version(version_id: str) -> str:
        """Retrieve a stored version, check its checksum, rebuild the model and confirm it reproduces the stored reference predictions on a fixed validation batch.

        Args:
            version_id: model version id
        """
        from gotrain.artifacts.verify import verify_version

        try:
            res = verify_version(store, version_id, device="cpu", geobench_root=rt.geobench_root)
        except (ValueError, FileNotFoundError) as exc:
            return _json({"error": str(exc)})
        session.record_decision("verify_model_version", version_id=version_id, verified=res.get("verified"), max_abs_diff=res.get("max_abs_diff"))
        return _json({k: res.get(k) for k in ("version_id", "verified", "sha256_ok", "max_abs_diff", "argmax_agreement", "preprocessing_identity", "error")})

    @tool
    def promote_model_version(version_id: str, rationale: str) -> str:
        """Mark a stored version as the promoted (selected) version for its task. Earlier versions stay stored and unchanged.

        Args:
            version_id: model version id
            rationale: why this version is promoted (cite validation scores)
        """
        if len(rationale.strip()) < 12:
            return _json({"error": "rationale must be a real sentence"})
        try:
            p = store.promote_model(version_id, rationale.strip(), by="agent")
        except (ValueError, FileNotFoundError) as exc:
            return _json({"error": str(exc)})
        session.record_decision("promote_model_version", **p)
        return _json(p)

    return [store_model_version, list_model_versions, verify_model_version, promote_model_version]
