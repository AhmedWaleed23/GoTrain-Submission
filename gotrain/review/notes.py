"""Reviewer notes: schema, validation and CSV conversion (the page's JavaScript
implements the same rules for export and import)."""

from __future__ import annotations

import csv
import io
from typing import Any

from gotrain.util.timing import utc_now_iso

NOTES_FORMAT = "gotrain-review-notes-v1"
DECISIONS = ("looks_correct", "needs_correction", "unsure")
DECISION_LABELS = {"looks_correct": "Looks correct", "needs_correction": "Needs correction", "unsure": "Unsure"}
IDENTITY_KEYS = ("task", "experiment_id", "model_version", "checkpoint_sha256", "split", "identity_hash")


def run_identity(provenance: dict) -> dict:
    model = provenance["model"]
    return {
        "task": model["task"],
        "experiment_id": model["experiment_id"],
        "model_version": model.get("model_version"),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "split": provenance["data"]["split"],
        "identity_hash": provenance["identity_hash"],
    }


def new_export(identity: dict, reviews: dict[str, dict], reviewer: str = "") -> dict:
    entries = []
    for sample_id, entry in sorted(reviews.items()):
        decision = entry.get("decision")
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f"bad decision {decision!r} for {sample_id}")
        entries.append({"sample_id": str(sample_id), "decision": decision, "note": str(entry.get("note", "")), "updated_at": entry.get("updated_at")})
    return {
        "format": NOTES_FORMAT,
        "exported_at": utc_now_iso(),
        "reviewer": reviewer,
        "identity": {k: identity.get(k) for k in IDENTITY_KEYS},
        "meaning": "human review labels for inspection; not training annotations, corrected masks or certification",
        "reviews": entries,
    }


def validate_import(payload: Any, identity: dict) -> list[dict]:
    """Return the review entries if the payload matches the schema and this run's identity."""
    if not isinstance(payload, dict):
        raise ValueError("notes file must be a JSON object")
    if payload.get("format") != NOTES_FORMAT:
        raise ValueError(f"unsupported notes format {payload.get('format')!r} (expected {NOTES_FORMAT})")
    ident = payload.get("identity")
    if not isinstance(ident, dict):
        raise ValueError("notes file lacks identity")
    for key in ("identity_hash", "checkpoint_sha256", "experiment_id", "split"):
        if ident.get(key) != identity.get(key):
            raise ValueError(f"notes were exported from a different run ({key}: {ident.get(key)!r} vs {identity.get(key)!r})")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews must be a list")
    out = []
    for i, entry in enumerate(reviews):
        if not isinstance(entry, dict) or "sample_id" not in entry:
            raise ValueError(f"review entry {i} is malformed")
        decision = entry.get("decision")
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f"review entry {i} has an unknown decision {decision!r}")
        note = entry.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"review entry {i} note must be text")
        out.append({"sample_id": str(entry["sample_id"]), "decision": decision, "note": note, "updated_at": entry.get("updated_at")})
    return out


def to_csv(payload: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    ident = payload.get("identity", {})
    writer.writerow(["sample_id", "decision", "decision_label", "note", "updated_at", "task", "experiment_id", "model_version", "checkpoint_sha256", "split", "exported_at"])
    for e in payload.get("reviews", []):
        writer.writerow([e.get("sample_id"), e.get("decision") or "", DECISION_LABELS.get(e.get("decision"), ""), e.get("note", ""), e.get("updated_at") or "", ident.get("task"), ident.get("experiment_id"), ident.get("model_version") or "", ident.get("checkpoint_sha256"), ident.get("split"), payload.get("exported_at")])
    return buf.getvalue()
