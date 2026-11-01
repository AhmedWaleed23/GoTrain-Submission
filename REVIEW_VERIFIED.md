> Development reference: this document describes the original implementation checks or fresh-run procedure. For final SageMaker status and scores, use README.md, VERIFIED.md and evidence/. Do not repeat submitted test evaluations.

# REVIEW_VERIFIED.md: what actually ran for the review feature, allocation fix and readiness check

Author environment: macOS arm64 laptop, CPU only, no AWS access. Python 3.12.3
virtualenv with torch 2.8.0 (CPU), numpy 1.26.4, rasterio 1.4.3, geobench 1.1.0,
huggingface_hub 0.36.0, strands-agents 1.63.0, Pillow 13.0.0, pytest. Browser
checks used Chromium through Playwright. Nothing below ran on SageMaker or the T4.

## Executed

| What | Result |
|---|---|
| Baseline suite on `origin/main` (`a63a5fd` in ancestry) | 61 passed |
| Full suite on the branch (`python -m pytest tests -q`) | 86 passed, 0 failed: 61 existing + 12 review + 7 allocation + 6 readiness |
| Bundle load (`.local-review-assets/bundle/solar_version`, version `mv-m-pv4ger-seg-0001-9f0e0bd9`) | checkpoint SHA-256 `9f0e0bd9...` matches the manifest; normalization hash `9dbf0929...` matches; `run_config.input_size` 160 used |
| Reference reproduction on CPU | max abs logit diff 4.59e-06 on the 4 stored validation inputs, within tolerance (atol 1e-3), argmax agreement 1.0. Scope: stored reference batch only |
| Real validation smoke (`--limit 16`, real m-pv4ger-seg validation data downloaded with the repo downloader) | 16 of 403 images in 0.44 s, labelled PARTIAL everywhere |
| Full real validation build on CPU (all 403 images, fp32, batch 8) | inference 9.2 s, build 9.6 s, thumbnails/page 3.8 s; `review.html` 13.6 MB with all 403 images embedded |
| fp32 recomputed validation mIoU vs the recorded selection score | 0.9229558 vs 0.9229633, abs diff 7.5e-06 (tolerance 5e-3) |
| Error-capture benchmark (403 images, 1,065,638 misclassified pixels, 0 both-empty images) | 10% budget (41 images): priority order 37.3% vs random mean 9.9% (p5 6.9%, p95 14.1%); 20% (81): 51.6% vs 19.7% (15.2%, 25.3%); 30% (121): 64.0% vs 29.4% (23.8%, 34.9%). Random: 200 orders, seed 42. Exploratory validation evidence only |
| Browser (Chromium, page served from a local folder) | 16-image and 403-image pages load with 0 console errors/warnings; queue click and `j` key navigation; decision buttons; note with quotes, newline and HTML kept as text; overlay toggles; reload keeps decisions (localStorage namespaced by identity hash); JSON export produced a real download (validated by `review-notes`, converted to CSV); import of a wrong-run file rejected with a visible message; import of a valid file applied 2 entries; 390 px width: horizontal overflow found and fixed (375 px after the fix) |
| Screenshots (real renders of the 16-image PARTIAL build and the 403 build) | docs/review_screenshots/ |
| `review-build`, `review-page`, `review-notes`, `submission-check` CLI | exercised directly and through tests (`tests/test_review.py`, `tests/test_submission_check.py`, `tests/test_cli.py`) |
| Allocation fix | scripted multi-task sessions: 3 tasks x 6 slots gives 2/2/2 with a greedy first task blocked by the guard; single task keeps 3/3; 2 slots over 3 tasks ends `partial` with the third task unfinished; resume keeps allowances and counts; launch-only sessions end `partial` |
| Readiness command | exit 0 with "ready to begin final evaluation ... NOT ready to submit" on a complete pre-eval home; exit 1 on a mismatched checkpoint hash, on a failed final-eval record with exposure, on a missing pointer; exit 2 on a missing home; JSON output |

## Not executed (pending for the owners)

- Any run on the T4: `review-build --device cuda` (the CPU numbers above are the reference; GPU fp32 results should match within the 5e-3 tolerance).
- The native SageMaker mode (`--task m-pv4ger-seg` using the run home's selected pointer and normalization file). Locally only the bundle mode could use the real model; the native mode is covered by the synthetic-fixture tests.
- A live multi-task Sonnet session with the new allowances (scripted tests only).
- `submission-check` against the real SageMaker run home.
- Opening `review.html` by double-click from a downloaded file in the owners' browser (local checks used a local HTTP folder because the automation blocks `file://`; the page has no external resources and no server dependency).

## Versions and scope statements

- Signal: normalized predictive entropy per pixel, mean of the top 10 percent per image; ties by sample id; labels never used for ranking. Stored in `benchmark.json` and `queue.json`.
- Foreground IoU with both target and prediction empty counts as 1.0; the number of such images is reported (0 in this validation set).
- The review never opens the test split (`assert_review_split` runs before any dataset access; tests cover `test` and `val`).
- `tests/test_review.py::test_no_training_or_fitting_calls_in_review_package` asserts the review package contains no training, fitting or final-evaluation calls.
- Chesapeake labels, the preflight patch, freeze and final-eval code paths were not changed; `freeze`/`final-eval` were not run.
