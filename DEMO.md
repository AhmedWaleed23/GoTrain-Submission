# Demo sequence (about five minutes)

1. Explain the goal: a supported dataset and a compute budget become logged training experiments, validation comparisons and a selected model.
2. Show the README result table. Distinguish classification accuracy from segmentation mean IoU. State that Chesapeake and cattle were improved manually after agent runs, before final testing.
3. Open a saved agent transcript under evidence/main_tasks/agent/sessions/. Show its hypothesis, actual tool invocation, returned score and next decision. Use the solar classification learning-rate change as the example. These are recorded real Bedrock runs, not a live training animation.
4. Open the full solar review.html from the evidence release asset. Navigate the queue, toggle uncertainty and correct-mask demo overlays, add a note and export it. Explain the 20% review-budget comparison and its validation-only scope. Show the cattle reviewer to demonstrate reuse.
5. Show a registry manifest, frozen selection, final_eval/results.json and exposure.jsonl. Explain train-only normalization, validation-only selection, immutable hashes, model reference verification and one final test opening per task.
6. End with the production path: GoPilot tool interface, remote storage and training workers. Clearly identify S3 and production MCP integration as unverified/future work.

Use saved pages and logs for the judging demo. Do not rerun final-eval to demonstrate its guard, corrupt artifacts, launch unnecessary training or overwrite the original run homes. Keep a screenshot backup for browser or network problems. Presentation slides are a separate deliverable.
