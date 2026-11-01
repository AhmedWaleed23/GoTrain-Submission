# GoTrain

A geospatial training agent built for the AUB ACM x RASID hackathon. Give it a supported GEO-Bench task and a compute budget: Claude Sonnet uses Strands tools to inspect data, choose a training configuration, train on a GPU, compare validation scores and select a model. Every experiment leaves saved configurations, scores and agent logs.

## Final results

Two classification tasks, two required segmentation tasks, and one additional segmentation task were completed on the organizer's SageMaker Tesla T4.

| Task | Validation | Final test | Metric | Final selection |
|---|---:|---:|---|---|
| m-chesapeake | 66.83% | **64.05%** | miou | manual |
| m-eurosat | 90.20% | **90.10%** | accuracy | agent |
| m-pv4ger | 94.09% | **95.60%** | accuracy | agent |
| m-pv4ger-seg | 92.30% | **91.44%** | miou | agent |
| m-nz-cattle | 84.77% | **84.32%** | miou | manual |

These are completed test evaluations, not validation estimates. Scores were independently checked against saved confusion matrices during evidence packaging. Segmentation uses macro mean IoU; Chesapeake includes all seven stored labels. Different tasks and metrics should not be averaged into a single accuracy.

Machine-readable results: [results.json](results.json). Original frozen selections, test results, exposure logs, experiment records and agent transcripts are under [evidence/](evidence/). Each task has one recorded final test opening after selection was frozen.

## What runs autonomously

The agent chooses among four implemented model architectures and supported training settings, launches experiments, reads validation results, adjusts its approach and records its selection within enforced time and experiment limits. The runtime uses `eu.anthropic.claude-sonnet-5-5` through Bedrock in `eu-central-1` on SageMaker. All trained networks use weights learned from scratch.

The final EuroSAT, solar classification and solar segmentation models were selected by the agent. Longer Chesapeake training and higher-resolution cattle training were proposed and selected manually using validation results. These manual improvements are recorded explicitly; the final five models are not claimed to be an entirely autonomous outcome. The agent does not generate arbitrary new training code or architectures.

## Extra feature: review before deployment

A GIS analyst can inspect a priority queue of uncertain predictions, toggle overlays, record decisions and notes, and export/import their review. Self-contained pages were built on all 403 solar validation images and all 66 cattle validation images. Ranking uses model uncertainty without access to correct labels; correct labels are used only for retrospective benchmarking and an optional demo overlay.

At a 20% image-review budget, the solar queue contained **51.6% of misclassified pixels**, compared with **19.7%** for the mean random queue. Cattle figures were **47.6% versus 21.1%**. These are validation experiments, not measured analyst time savings or guarantees for deployment. Reviews do not automatically retrain models or correct masks. Production interactive overlays currently support binary segmentation, demonstrated on solar and cattle; the technical dashboard covers all five tasks.

The downloadable evidence ZIP contains the standalone dashboards and reviewers. Structured benchmark evidence is also in `evidence/*/review/`. See [REVIEW_RUNBOOK.md](REVIEW_RUNBOOK.md) and [docs/REVIEW_RATIONALE.md](docs/REVIEW_RATIONALE.md).

## Storage and model versions

The filesystem artifact store saves and retrieves datasets and immutable model versions with architecture, preprocessing, configuration, hashes and validation references. All five selected versions passed GPU reference verification. A EuroSAT store/retrieve demonstration compared all 4,012 files successfully. S3 support is implemented but has not been exercised; demonstrated storage is filesystem storage.

Two release assets accompany the submission: `gotrain_submission_evidence_20261101T073155Z.zip` (logs, scores, dashboards, reviewers) and `gotrain_selected_models_20261101T074107Z.zip` (five selected model bundles and validation references). They must be attached when publishing the submission release; they are not ordinary Git source files.

## Run the code on SageMaker

From the cloned repository's terminal:

```bash
/opt/conda/bin/python -m gotrain env-check
/opt/conda/bin/python -m gotrain test
```

The tested environment is Python 3.12.15, PyTorch 2.8.0, NumPy 1.26.4, rasterio 1.4.3, geobench 1.1.0, huggingface-hub 0.36.0 and strands-agents 1.63.0. Preserve the supplied CUDA-enabled PyTorch installation. `requirements-constraints.txt` constrains data dependencies; Bedrock access uses the SageMaker role, not an API key.

For a NEW training demonstration, use a new results folder:

```bash
/opt/conda/bin/python -m gotrain --home ~/gotrain_demo_runs download --task m-eurosat
/opt/conda/bin/python -m gotrain --home ~/gotrain_demo_runs inspect --task m-eurosat --save --fit-normalization
/opt/conda/bin/python -m gotrain --home ~/gotrain_demo_runs agent --task m-eurosat --device cuda --session-minutes 12 --run-max-minutes 3 --max-experiments 2 --max-rounds 4
/opt/conda/bin/python -m gotrain --home ~/gotrain_demo_runs report
```

This launches new training and Bedrock calls. It does not reproduce a score deterministically or use the test split. Do not rerun final evaluation for the submitted models. The detailed [RUNBOOK.md](RUNBOOK.md) describes a fresh end-to-end experiment lifecycle; historical submitted evidence should be read as saved files, not treated as a restored runnable results home.

The evidence ZIP and model ZIP use two groups: `main_tasks` and `cattle`. Keep their frozen manifests separate. Bundle directories contain `manifest.json`, `checkpoint.pt`, `normalization.json` and `reference.npz`. For a fresh binary review build, `review-build --bundle /path/to/version --device cuda` loads bundle-local preprocessing; download the corresponding validation dataset first. Original metadata preserves historical absolute SageMaker paths and must not be silently rewritten. The lightweight evidence folder intentionally omits model weights, image caches and HTML pages; standalone pages are in the evidence release asset.

## Automatic delivery after training

Add `--complete` to an `agent` command to store/reuse and verify selected models, build supported binary reviewers and refresh the dashboard after training. For existing selections, `python -m gotrain --home /path/to/runs complete --task all --device cuda` performs those steps without training or LLM calls. Cattle is explicit: `--task m-nz-cattle`. Each invocation saves per-stage success/failure evidence under `completion/`; incomplete output yields a nonzero exit code. Completion time is additional to the agent's training budget. Freeze and final test evaluation remain separate operator commands. See [AUTOMATION.md](AUTOMATION.md).

## Verification and limitations

The latest combined source passed **93 tests** on SageMaker. Tests include synthetic training, metrics, budget guards, artifact integrity, test isolation, review behavior and reporting; they do not prove arbitrary-dataset generalization. The optional S3 backend and a production GoPilot/MCP integration remain future work. See [VERIFIED.md](VERIFIED.md), [PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md) and [DEMO.md](DEMO.md).
