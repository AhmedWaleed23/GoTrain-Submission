> Development reference: this document describes the original implementation checks or fresh-run procedure. For final SageMaker status and scores, use README.md, VERIFIED.md and evidence/. Do not repeat submitted test evaluations.

# GoTrain runbook (beginner edition)

This is the step-by-step guide for running GoTrain on the organizer SageMaker
instance. Each step says what to type, what you should see, and where files land.
Timings are deliberately not promised: step 5 measures them on your GPU.

Conventions: `$` lines are typed in a JupyterLab terminal from the repository root
(`cd ~/GoTrain`). Python is `/opt/conda/bin/python`; run `export PATH=/opt/conda/bin:$PATH`
once per terminal so `python` means that one. All outputs go under
`gotrain_runs/` inside the repository (override with `GOTRAIN_HOME=/path`).

## 0. Get the code onto SageMaker

Upload the ZIP through the JupyterLab file browser (or `git clone` your repository),
then in a terminal:

```
$ cd ~ && unzip -q GoTrain.zip && cd GoTrain
$ export PATH=/opt/conda/bin:$PATH
```

## 1. Environment check

```
$ python -m gotrain env-check
```

Expected: a JSON block with `"torch": "2.8.0"`, `"cuda_available": true`,
`"gpu": {"name": "Tesla T4", ...}`, `"strands_import": "ok"`, `"geobench_import": "ok"`
and `"datasets_present"` with `m-eurosat: true`. Warnings print to the terminal if
CUDA or strands are unavailable. If `geobench_import` fails:

```
$ python -m pip install -c requirements-constraints.txt numpy rasterio geobench huggingface_hub
```

Never reinstall torch and never run an unconstrained `pip install geobench`
(it upgrades NumPy and breaks rasterio).

## 2. Tests (CPU, no data needed)

```
$ python -m gotrain test
```

Expected: all tests pass. A line like `53 passed` (the exact count is in
VERIFIED.md). They take a few minutes because they train tiny models on CPU.

## 3. Data

```
$ python -m gotrain download --task all
```

Downloads `m-pv4ger` (about 1.0 GB), `m-pv4ger-seg` (0.3 GB) and `m-chesapeake`
(1.25 GB) one zip at a time into `~/dataset/geobench/<benchmark>/`, verifies each
zip's SHA-256 against the value Hugging Face publishes, extracts, checks the
extracted layout, and writes `.gotrain_download.json` next to the data. EuroSAT is
detected as already present and skipped. Add `--delete-zip` to free about 2.5 GB
afterwards. Re-running is safe: present datasets are not downloaded again.

```
$ python -m gotrain inspect --task all --save --fit-normalization
```

Expected per task: split sizes (train/valid/test counts), class names, channels,
image shapes, label fractions on train and valid, `invalid_labels: 0`, value ranges
and `warnings: []`. Saved to `gotrain_runs/inspect/<task>.json`. If a task reports
a metadata mismatch, stop and read the message: the task config in
`configs/tasks/` disagrees with the data and nothing will train until that is
resolved.

## 4. Fitting normalization

Happens automatically on the first run per task (train split only) and is cached
in `gotrain_runs/normalization/<task>.json` with a hash. Nothing to do.

## 5. Smoke run: measure speed

```
$ python -m gotrain smoke --task all
```

One epoch on 10 percent of the training data per task, validation limited to 100
samples, labelled `screening` so it never enters rankings. Expected: four JSON
summaries with `"status": "completed"`. Look at `wall_minutes` and, in
`gotrain_runs/experiments/<id>/log.txt`, the `(NNs+MMs)` figures after each epoch:
training seconds plus validation seconds. Multiply the training seconds by 10 for a
full epoch. Use that to set `configs/budget.json`:

- `baseline_epochs`, `baseline_max_minutes`: enough for a few full epochs per task
- `agent_session_minutes`: how long one agent session may run (all four tasks)
- `agent_run_max_minutes`: cap for a single experiment
- `finalist_epochs`, `finalist_max_minutes`: the common budget for finalists

Keep at least 45 minutes before the deadline free for freeze, final evaluation
and the report; the agent does not know about the deadline.

Rough resource notes (not timings): the T4 has 14.5 GiB. `cnn_small` on 64x64x13
EuroSAT is tiny at any batch size up to 128. `unet_small` on 320x320 RGB at batch
16 and `unet_medium` at batch 8 are comfortable; larger batches may trigger the
automatic batch halving, which is recorded, not fatal.

## 6. Baselines (background)

```
$ scripts/run_background.sh baseline baseline --task all
$ scripts/tail_logs.sh baseline
```

The first command prints a pid and a log path under `gotrain_runs/logs/`. The job
keeps running if the browser disconnects. Expected at the end of the log: one
`completed` line per task with `best accuracy=...` or `best miou=...`. Then:

```
$ python -m gotrain list
```

shows the experiments and a `best[<task>] = ...` line per task.

## 7. Autonomous agent (background)

```
$ scripts/run_background.sh agent agent --task all --session-minutes 90
$ scripts/tail_logs.sh agent
```

The agent inspects each task, reads the baseline, proposes experiments with a
hypothesis and rationale, runs them, reacts to the results, selects the best
completed full run and finishes the task. Watch `[agent] launching ...` lines in
the log. Everything it does is in `gotrain_runs/agent/sessions/<session_id>/`:

- `decisions.jsonl`: proposals, results, guard blocks, selections, summaries
- `transcript.jsonl`: compact model/tool messages per round
- `state.json`: counters and budget

If the session ends with a task `left unfinished`, the best pointer still holds
the best completed run (set automatically). You can give it more time by
resuming with a larger TOTAL session budget (elapsed time carries over, so a
90-minute session resumed with 135 gets 45 more minutes):

```
$ python -m gotrain agent --task all --resume-session <session_id> --session-minutes 135
```

## 8. If something was interrupted

- A training run: `python -m gotrain list` shows `interrupted`; resume with
  `python -m gotrain resume <experiment_id>`. Finished epochs are not repeated.
- The lock file `gotrain_runs/locks/training.lock` prevents two jobs at once. If a
  job died hard, the lock is ignored automatically (dead pid). Check with
  `scripts/status.sh`.
- Never start a second baseline or agent while `scripts/status.sh` shows a running
  `python -m gotrain` process; the launcher refuses when the lock is live.

## 8b. Review page for the solar segmentation model

See REVIEW_RUNBOOK.md (`python -m gotrain review-build --task m-pv4ger-seg --device cuda`).
`python -m gotrain submission-check` prints a read-only readiness checklist at any time.

## 9. Dashboard

```
$ python -m gotrain report
```

Writes `gotrain_runs/reports/dashboard.html`. Open it from the JupyterLab file
browser (double-click) or download it. It is a single file with no external
resources.

## 10. Freeze the selection

```
$ python -m gotrain freeze --task all
```

Writes `gotrain_runs/selection/manifest.json` with the chosen experiment,
checkpoint hash and normalization hash per task, stores each chosen checkpoint as
an immutable model version in `gotrain_runs/registry/` and promotes it. Refuses if
any task has no completed full run. To override a task's choice:
`--experiment m-eurosat=<experiment_id>`. Existing manifests are only replaced
with `--replace`, and never after test results exist.

## 11. Final test evaluation (once)

```
$ python -m gotrain final-eval --i-understand-this-uses-the-test-split
```

Evaluates each frozen task on its test split once and writes
`gotrain_runs/final_eval/results.json`, `events.jsonl` and `exposure.jsonl`.
Running it again prints `already evaluated ... refusing` for completed tasks and
only retries failed ones. Then `python -m gotrain report` shows the test scores.

## Troubleshooting

- `dataset m-pv4ger not found`: run step 3; check `GEO_BENCH_DIR` if you moved data.
- `metadata mismatch`: the data differs from `configs/tasks/<task>.json`; inspect
  and fix the config deliberately, do not guess.
- `CUDA out of memory` in a log: the trainer already halved the batch size and
  retried (see `adjustments` in the record). If it still fails, lower
  `--batch-size` or use the smaller model.
- `another training job is running`: wait, or confirm with `scripts/status.sh`
  that the pid is dead and delete `gotrain_runs/locks/training.lock`.
- Agent round errors in the log (Bedrock throttling or network): the loop retries
  the next round and stops after two consecutive failures; resume the session.
- `strands not importable`: the agent cannot run, but baselines, registry, freeze,
  final evaluation and report do not need it.
- Browser disconnected: nothing is lost; jobs started with
  `scripts/run_background.sh` keep running. Reconnect and `scripts/tail_logs.sh`.
- To look at a single experiment: `python -m gotrain show <id>` (add `--events`).
