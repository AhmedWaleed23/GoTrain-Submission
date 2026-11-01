# Production design: from this repository to GoPilot

This document separates what is implemented in this repository from what is
proposed. "Implemented" means code that exists here, has tests, and ran on CPU
during the handoff. "Proposed" means a concrete design that was not built.

## Implemented today (local, single instance)

```
+------------------------------------------------------------------+
| SageMaker instance (one T4)                                       |
|                                                                   |
|  Strands Agent (Claude Sonnet on Bedrock)                         |
|     | tool calls (JSON in / JSON out)                             |
|     v                                                             |
|  BudgetGuard hook --> tools: inspect_dataset, list_experiments,   |
|   (limits.json)        get_budget_status, run_experiment,         |
|                        get_experiment, select_best_model,         |
|                        record_note, finish_task,                  |
|                        store/list/verify/promote model versions   |
|     |                                                             |
|     v                                                             |
|  Experiment runner --> Trainer (AMP, early stop, OOM/NaN recovery)|
|     |                     |                                       |
|     |                     v                                       |
|     |         gotrain_runs/experiments/<id>/ (record, events,     |
|     |                     checkpoints, curves, predictions)       |
|     v                                                             |
|  FilesystemArtifactStore: registry/models.json, models/<version>/ |
|     (immutable copies + manifest + reference batch)               |
|                                                                   |
|  freeze -> selection/manifest.json -> final-eval (test, once)     |
|  report -> reports/dashboard.html (static, self-contained)        |
+------------------------------------------------------------------+
            ^                                   |
            | geobench HDF5 (selective download) | optional S3 mirror
      ~/dataset/geobench                   (S3ArtifactStore, not executed)
```

Interface that production components must keep (from `gotrain/artifacts/store.py`):

```python
class ArtifactStore(ABC):
    def store_dataset(self, task, source_dir, mode="reference") -> dict
    def retrieve_dataset(self, task, dest=None) -> dict
    def store_model(self, experiment_id, note="", parent_version=None) -> dict   # immutable version
    def retrieve_model(self, version_id, dest=None) -> dict                      # checksum-verified
    def list_versions(self, task=None) -> list[dict]
    def promote_model(self, version_id, rationale, by="manual") -> dict           # pointer only
    def get_version(self, version_id) -> dict
```

A version manifest (`registry/models/<version>/manifest.json`) contains: version id,
task, experiment id, checkpoint SHA-256 and size, storage URI, metric name, value,
direction and definition, preprocessing (normalization hash and copy, input size,
band order), architecture (name, channels, classes, class names, parameters), run
config, seed, environment, parent version, note, reference batch description, and
the last verification result. This is enough to rebuild inference anywhere.

## Proposed extension 1: S3 object store (partially coded, not executed)

`gotrain/artifacts/s3.py` subclasses the filesystem store and mirrors each version
directory to `s3://<bucket>/<prefix>/models/<version>/` with SHA-256 as object
metadata; `retrieve_model` downloads missing files into the local cache and
verifies the hash before use. `python -m gotrain s3-smoke --bucket B` proves
bucket access with a put/get/delete round trip. Production changes:

- bucket and prefix from configuration, never created by code; IAM role grants
  `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on the prefix only;
- multipart uploads for checkpoints above 100 MB, server-side encryption on;
- datasets mirrored once per task as the original zip plus its manifest, so a
  worker can rebuild `~/dataset/geobench/<benchmark>/<task>` from S3 with one call;
- lifecycle rule: experiment checkpoints expire after 30 days, registry versions
  never expire.

## Proposed extension 2: version index service

Today the index is `registry/models.json`, written atomically and reconciled from
manifests. For more than one writer:

- move the index to DynamoDB (partition key `task`, sort key `version_id`) with a
  conditional put (`attribute_not_exists(version_id)`) to keep versions immutable;
- promotion is a single-row update on a `promoted#<task>` item carrying the previous
  version, so history is append-only in a stream;
- `list_versions`, `get_version`, `promote_model` keep the same signatures; only
  the backend class changes (`DynamoVersionIndex`).

## Proposed extension 3: training workers

The agent today calls `run_experiment` synchronously on the same instance. In
production the tool enqueues a job and polls:

```
agent (Strands) --run_experiment--> job queue (SQS) --> worker pool (SageMaker
   ^                                                     training jobs or ECS GPU)
   |                                                         |
   +-------- get_experiment / list_experiments <---- records in S3 + index
```

- `run_experiment` returns a job id immediately; `get_experiment(job_id)` returns
  status and the same compact summary used now; the round loop already tolerates
  "running" states because every summary is rebuilt from records;
- workers run `python -m gotrain run --config <json>` with `GOTRAIN_HOME` pointing
  at a mounted or synced directory and upload the experiment directory at the end;
- the GPU lock becomes a queue concurrency limit (one job per worker);
- budgets move from a session-local timer to a shared ledger keyed by session id,
  so several agents cannot overspend a team quota.

## Proposed extension 4: GoPilot integration through MCP

The tool layer is already JSON-in/JSON-out with Python-side validation, which is
what an MCP server needs. Proposed server (`gotrain-mcp`):

```
tools:
  gotrain.inspect_dataset(task)                 -> same JSON as today
  gotrain.list_experiments(task, include_screening)
  gotrain.run_experiment(task, model, lr, batch_size, epochs, max_minutes,
                         hypothesis, rationale, expected_tradeoff, ...)
  gotrain.get_experiment(experiment_id)
  gotrain.list_model_versions(task) / get_model_version(version_id)
  gotrain.verify_model_version(version_id)
  gotrain.promote_model_version(version_id, rationale)
resources:
  gotrain://experiments/{id}/record.json
  gotrain://registry/{version_id}/manifest.json
  gotrain://reports/dashboard.html
```

- the server imports `gotrain.agent.tools.build_tools` and wraps each tool with the
  MCP schema derived from the same signatures (Strands and MCP both consume JSON
  schema);
- the budget guard runs inside the server, so GoPilot cannot exceed limits
  regardless of its prompt;
- test-split evaluation is not exposed as an MCP tool; it stays an operator command;
- authentication: the server runs next to the data with the SageMaker role; GoPilot
  reaches it over an authenticated MCP transport (streamable HTTP behind IAM or an
  API gateway token).

Example interaction (JSON-RPC payload abbreviated):

```json
{"method": "tools/call", "params": {"name": "gotrain.run_experiment",
 "arguments": {"task": "m-eurosat", "model": "resnet_small", "lr": 0.002,
   "batch_size": 64, "epochs": 12, "max_minutes": 15,
   "hypothesis": "Residual blocks improve accuracy over the plain CNN at equal budget.",
   "rationale": "cnn_small baseline reached 0.93 and was still improving at epoch 8.",
   "expected_tradeoff": "about 2x the time per epoch."}}}
```

## What stays the same across all extensions

- Frozen metric definitions per task (`configs/tasks/*.json`).
- Validation-only tuning and the token-gated test split.
- Immutable versions, promotion as a pointer, SHA-256 everywhere.
- Records as the single source of truth; the dashboard renders records only.
