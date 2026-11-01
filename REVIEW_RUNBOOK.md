# Review workflow runbook (beginner edition)

Everything below runs on the organizer SageMaker instance unless the heading says
"laptop". Python on SageMaker is `/opt/conda/bin/python`; the existing run home
`/home/sagemaker-user/GoTrain/gotrain_runs` holds your experiments, registry and
selection and must be preserved (nothing here writes into it except a new
`review/` folder).

## 1. Get the branch without losing local work

In a JupyterLab terminal:

```
cd ~/GoTrain
git status                      # should be clean; if not, commit or stash first
git fetch origin
git checkout feature/review-workbench
git pull --ff-only origin feature/review-workbench
```

If `git status` shows local changes you want to keep, run `git stash` before the
checkout and `git stash pop` after it. Do not run the preflight patch again; the
branch already contains its effects (`gotrain_preflight_patch.py` is unchanged).

## 2. Run the tests (about 3 to 6 minutes on CPU, no GPU needed)

```
/opt/conda/bin/python -m gotrain test
```

Expected: all tests pass (86 in the author's environment: 61 existing plus 25
new). If `pytest` is missing: `/opt/conda/bin/python -m pip install -c requirements-constraints.txt pytest`.

## 3. Smoke build on 16 validation images (under a minute)

```
cd ~/GoTrain
/opt/conda/bin/python -m gotrain review-build --task m-pv4ger-seg --device cuda --batch-size 8 --num-workers 0 --limit 16
```

Expected on stderr: the model line (`exp-20261031T182357Z-34155d`, version
`mv-m-pv4ger-seg-0001-9f0e0bd9`, input size 160), a reference-check line with a
tiny max abs diff, `PARTIAL build` warning, progress lines, and a summary with the
three budgets. Expected on stdout: a JSON block with `"partial": true`,
`"n_images": 16` and the page path
`gotrain_runs/review/m-pv4ger-seg/mv-m-pv4ger-seg-0001-9f0e0bd9_limit16/review.html`.
The page shows a yellow PARTIAL badge; do not read conclusions from 16 images.

## 4. Full build on all 403 validation images

```
/opt/conda/bin/python -m gotrain review-build --task m-pv4ger-seg --device cuda --batch-size 8 --num-workers 0
```

On the author's laptop CPU this took about 10 seconds of inference plus a few
seconds of thumbnail encoding; on the T4 it should be similar or faster. Expected
stdout: `"partial": false`, `"n_images": 403`,
`"split_miou_fp32"` close to 0.9230 and `"metric_comparison"` with
`"within_tolerance": true` (the review recomputes validation mIoU in fp32 and
compares it with the recorded 0.92296; a difference above 0.005 means something is
wrong with preprocessing and should be sent back). Output directory:
`gotrain_runs/review/m-pv4ger-seg/mv-m-pv4ger-seg-0001-9f0e0bd9/` with
`review.html` (about 14 MB, all 403 images embedded), `items.json`, `queue.json`,
`benchmark.json`, `provenance.json` and `cache/` (per-image thumbnails and masks).

Re-running the same command rebuilds in place. Building for a different model
into the same folder is refused unless `--force`.

## 5. Open and use the page

Download: in the JupyterLab file browser, navigate to
`GoTrain/gotrain_runs/review/m-pv4ger-seg/mv-m-pv4ger-seg-0001-9f0e0bd9/`,
right-click `review.html`, Download. Open the downloaded file in Chrome, Firefox
or Safari by double-clicking. It needs no internet connection and no server.
(JupyterLab's own HTML viewer may block the page's script; use a normal browser.)

Using it: the left list is the review queue in priority order. Click an image
(or press `j`/`k`), look at the three panels, tick "Show correct mask" when you
want the validation truth, press `1` (looks correct), `2` (needs correction) or
`3` (unsure), type a note, move on. Decisions stay in the browser for this model
only. Click "Export notes (JSON)" or "(CSV)" to save a file; "Import notes JSON"
restores an export (files from another model or build are rejected). The
collapsed "Technical evidence and ranking benchmark" section holds the benchmark
table, the capture curve, checksums and limits.

Validate or convert an exported file on the command line:

```
/opt/conda/bin/python -m gotrain review-notes gotrain_runs/review/m-pv4ger-seg/mv-m-pv4ger-seg-0001-9f0e0bd9 --notes ~/Downloads/review-notes-m-pv4ger-seg-<stamp>.json --csv notes.csv
```

## 6. Technical report link

```
/opt/conda/bin/python -m gotrain report
```

The dashboard (`gotrain_runs/reports/dashboard.html`) now lists review pages
under "Review pages" with their coverage; the link only appears when a review
output exists.

## 7. Optional owner checks for the allocation fix and the readiness command

No GPU or Bedrock needed:

```
/opt/conda/bin/python -m gotrain agent --task synthetic --scripted --max-experiments 3 --session-minutes 10 --device cpu
/opt/conda/bin/python -m gotrain submission-check
/opt/conda/bin/python -m gotrain submission-check --json gotrain_runs/submission_check.json
```

`submission-check` is read-only. Exit code 0 means no blocking problems
(pending final scores are expected before the final run), 1 means blocking
problems (for example a checkpoint whose hash no longer matches), 2 means a bad
home path. A live multi-task Sonnet session re-check is pending unless you run
one; the allocation behaviour is covered by scripted tests.

## 8. If something fails

Send back the full terminal output (both stderr and stdout) of the failing
command, plus:

- `ls -la gotrain_runs/review/m-pv4ger-seg/*/` and the `provenance.json` inside;
- for a metric mismatch: the `metric_comparison` block of the JSON output;
- for a page problem: the browser name/version and whether the page was opened
  from a downloaded file or through JupyterLab.

Common cases: `no selected experiment for m-pv4ger-seg` means the run home is not
the one with your experiments (check `GOTRAIN_HOME` or run from `~/GoTrain`);
`checkpoint on disk does not match` means the experiment folder changed since
training; `dataset m-pv4ger-seg not found` means `GEO_BENCH_DIR` does not point at
`/home/sagemaker-user/dataset/geobench`.

## Laptop: build from the portable bundle (no SageMaker data needed for the bundle check)

```
gh release download review-handoff-v1 --repo AhmedWaleed23/GoTrain --pattern gotrain_review_handoff.zip --dir .local-review-assets
cd .local-review-assets && unzip -q gotrain_review_handoff.zip -d bundle && cd ..
python -m gotrain --home .local-review-assets/scratch_home review-build --bundle .local-review-assets/bundle/solar_version --device cpu --limit 16
```

The bundle check (checksum, normalization hash, reference logits) runs before any
inference. A full laptop build additionally needs the validation data:
`python -m gotrain download --task m-pv4ger-seg` (300 MB, train/valid/test files
are downloaded, only `valid` is ever opened by the review), then drop `--limit`.
Outputs go under `.local-review-assets/scratch_home/review/`, which git ignores.
