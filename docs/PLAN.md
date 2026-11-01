# GoTrain architecture and risk plan

Written before implementation; kept short on purpose. See README.md for usage,
PRODUCTION_DESIGN.md for the production extension, VERIFIED.md for evidence.

## Goal

An autonomous training agent (Strands + Claude Sonnet on Bedrock) that trains
GEO-Bench models on one Tesla T4, with an auditable experiment history,
compute-aware decisions, validation-only tuning, and a single final test pass.

## Shape of the solution

```
python -m gotrain <command>        CLI, no install needed (run from repo root)
gotrain/
  data/        geobench adapter, selective download, normalization (train only),
               paired augmentation, synthetic datasets for CPU tests
  models/      cnn_small / resnet_small (classification), unet_small / unet_medium
               (segmentation); channels and classes come from task metadata
  train/       trainer (AMP on CUDA, FP32 fallback, early stopping, OOM + nonfinite
               recovery, resume), checkpoints (atomic + sha256), metrics
  experiments/ experiment store (atomic record.json, append-only events.jsonl,
               best pointer), selection freeze
  artifacts/   ArtifactStore interface, filesystem backend, model registry with
               immutable versions, roundtrip verification, optional S3 backend
  agent/       Strands tools, budget guard hook, session runner, fake model for
               offline orchestration tests, fixed-config baseline runner
  final_eval/  one-shot test evaluation from a frozen manifest
  report/      self-contained HTML dashboard from saved records (no CDN)
configs/       task specs, model/lr allow-lists, budgets (editable JSON)
tests/         pytest, CPU-only, synthetic data; real-data tests skip if absent
```

Storage root (`GOTRAIN_HOME`, default `./gotrain_runs`): `experiments/`,
`normalization/`, `agent/`, `registry/`, `selection/`, `final_eval/`, `reports/`.

## Frozen metric conventions

GEO-Bench's published experiment code (`geo-bench-experiments`,
`torch_toolbox/model.py`) scores classification with torchmetrics
`Accuracy(task="multiclass")` (overall accuracy) and segmentation with
`JaccardIndex(task="multiclass")` (macro mean IoU over classes, confusion
matrix accumulated over the whole split, classes absent from both target and
prediction excluded, no ignore index). GoTrain re-implements exactly these as
`accuracy` and `miou`, verifies them against torchmetrics in tests, and freezes
them per task in `configs/tasks/*.json`. Macro-F1, confusion matrices,
per-class IoU and pixel accuracy are reported as diagnostics only.

## Test-split discipline

The data layer refuses `split="test"` unless given a token that only the
`final-eval` command creates. Agent tools, smoke, baseline, inspect, report and
normalization fitting never hold that token. Every test-split open is logged.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Unknown T4 runtime per epoch | `smoke` measures it first; budgets are config, not promises |
| Strands version on target differs | depend only on `Agent`, `@tool`, hooks, `BedrockModel`; feature-detect extras |
| CUDA OOM / nonfinite loss | bounded retries with halved batch size, recorded in the run record |
| Browser disconnects | background launch script with nohup and a lock file against duplicate jobs |
| Context growth in the agent loop | short rounds with compact state summaries; state persisted on disk |
| Accidental test evaluation | token-gated split access; final results persisted; re-evaluation refused |
| EuroSAT wrong band names | channels used positionally; RGB positions 3,2,1; names recorded as "stored order" |
| chesapeake has a `no-data` class | kept as a class for the frozen metric (benchmark convention); also reported without it |
| Pretrained weights | not required; from-scratch path is the default, pretrained is a backlog item |
