# Automatic post-training completion

`agent --complete` runs the existing budgeted agent, then stores or reuses the selected model version, verifies its checkpoint and preprocessing with validation reference data, builds a full validation reviewer for solar/cattle binary segmentation, and refreshes the dashboard. A task or stage failure is recorded and produces a nonzero command exit status. Other requested tasks still get processed.

For a NEW training run with Bedrock access and prepared data:

```bash
python -m gotrain --home ~/gotrain_new_runs agent --task m-eurosat --device cuda --session-minutes 12 --run-max-minutes 3 --max-experiments 2 --max-rounds 4 --complete
```

Use `--completion-batch-size 4 --completion-num-workers 0` for a small review batch. Completion runs AFTER the training/session budget; its inference and rendering time is additional. It makes no further LLM calls. Dataset download and initial train-only normalization preparation follow the existing runbook.

For an EXISTING results home (no new training or Bedrock calls):

```bash
python -m gotrain --home /path/to/existing/runs complete --task m-eurosat,m-pv4ger,m-pv4ger-seg,m-chesapeake --device cuda --batch-size 8 --num-workers 0
python -m gotrain --home /path/to/cattle/runs complete --task m-nz-cattle --device cuda --batch-size 4 --num-workers 0
```

`all` means the four required tasks; cattle must be named explicitly. Missing selections are failures, not silently omitted. Classification and multiclass segmentation explicitly skip the interactive reviewer and remain in the technical dashboard. The supported binary reviews process the full validation split, without a partial-data limit.

Each invocation saves `completion/<run-id>.json` plus `completion/latest.json` with the selected experiment, reused/stored version, reference verification result, review coverage/page path, dashboard path and stage errors. Repeated runs reuse matching versions. They may rebuild reviewer files and refresh verification timestamps in registry metadata. Existing browser notes remain tied to review identity; keep exported notes as a backup.

The workflow never selects a new model, freezes, promotes, trains, or evaluates a test split. It checks that saved best pointers, the frozen selection and final-evaluation evidence remain byte-identical across completion. Freeze and final evaluation remain explicit operator steps; run them only for a genuinely new submission model before its test split has been opened.

Development checks: eight dependency-free orchestration tests cover successful handoff, unsupported reviewers, verification/review/report failures, reruns, preserved evidence and changing selections. Additional SageMaker tests exercise actual synthetic storage/verification/reporting and the `agent --complete` handoff. Standalone real GPU completion subsequently passed for all five submitted models, with both full binary reviewers and both dashboards generated; see `verification/automatic_completion/`. All 103 SageMaker tests passed. A fresh live Bedrock session using `agent --complete` has not been run; its connection is covered by the mocked handoff test.
