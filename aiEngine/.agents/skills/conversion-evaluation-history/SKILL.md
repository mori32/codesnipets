---
name: conversion-evaluation-history
description: Append completed coltIme corpus-conversion evaluation metrics to a dated CSV history, including dictionary hashes, character accuracy, N-best coverage, exact-match rates, and raw score summaries.
---

# Conversion Evaluation History

Record each completed corpus conversion evaluation as one row in the ongoing CSV at `C:\lmwork\int\coltime-conversion-history.csv`.

## Workflow

1. Locate the completed evaluation `summary.json`, its preparation metadata, and `raw-score-summary.json` if one was produced.
2. Confirm the summary status is `completed`. Do not add interrupted, partial, or failed runs. Conversion errors within a completed run remain part of that result and are recorded in the error count.
3. Run the `scripts/append_history.py` helper next to this `SKILL.md`. For a personal install, the command is:

   ```powershell
   python "$HOME\.codex\skills\conversion-evaluation-history\scripts\append_history.py" `
     --summary "C:\path\to\results\summary.json"
   ```

   Pass `--preparation` or `--raw-scores` only when the metadata is not at its normal location. Use `--history-csv` when the user requests a different destination, such as a repository-tracked history. The default history is the local `C:\lmwork\int\coltime-conversion-history.csv` file. In the aiEngine repository, the tracked history is `aiEngine/docs/conversion-evaluation-history.csv`; pass that path with `--history-csv` when updating the repository copy.
4. Verify the script reports `ok: true`, then inspect the CSV header and newest row. Re-running on the same summary is safe: the summary SHA-256 prevents a duplicate row.

## Recorded fields

Each row stores the completion date and timestamp, dictionary SHA-256 and path, source sample/cases/harness hashes, segmentation policy, held-out marker, evaluation counts, character accuracy and error rate, N-best oracle character accuracy and exact inclusion rate, top-1 exact rates, raw Viterbi and reranked N-best score summaries, top1-top2 margins, conversion latency percentiles, and conversion error count. Rate columns are percentages on a 0–100 scale.

Raw path scores are summed descriptive values, not calibrated probabilities. Keep the interpretation in mind when comparing evaluations with different target lengths. Preserve the evaluation's `held_out` status; do not describe a run as independent test accuracy unless corpus/dictionary disjointness is established.

## Helper

`scripts/append_history.py` validates completion and matching case hashes, refuses to rewrite a CSV with an unexpected schema, writes atomically, and is idempotent for an identical summary file. Use `--help` for arguments.
