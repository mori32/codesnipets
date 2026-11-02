from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_HISTORY = Path(r"C:\lmwork\int\coltime-conversion-history.csv")
FIELDS = [
    "evaluation_date",
    "completed_at",
    "dictionary_sha256",
    "source_sample_sha256",
    "cases_sha256",
    "harness_sha256",
    "segmentation_policy",
    "held_out",
    "cases",
    "reference_scalars",
    "final_edits",
    "final_character_accuracy_pct",
    "final_character_error_rate_pct",
    "nbest_oracle_character_accuracy_pct",
    "nbest_exact_inclusion_rate_pct",
    "final_top_exact_rate_pct",
    "reranked_top_exact_rate_pct",
    "viterbi_top_exact_rate_pct",
    "mean_unique_nbest_surfaces",
    "viterbi_bigram_top1_mean_raw_score",
    "viterbi_bigram_top1_p50_raw_score",
    "viterbi_bigram_top1_p95_raw_score",
    "reranked_nbest_trigram_top1_mean_raw_score",
    "reranked_nbest_trigram_top1_p50_raw_score",
    "reranked_nbest_trigram_top1_p95_raw_score",
    "reranked_top1_minus_top2_mean_raw_score",
    "reranked_top1_minus_top2_p50_raw_score",
    "reranked_top1_minus_top2_p95_raw_score",
    "conversion_time_mean_ms",
    "conversion_time_p50_ms",
    "conversion_time_p95_ms",
    "conversion_time_p99_ms",
    "conversion_errors",
    "summary_sha256",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.6f}"


def number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def error_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(error_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def preparation_path(summary: dict[str, Any], summary_path: Path, override: Path | None) -> Path | None:
    if override:
        return override
    named = summary.get("preparation_file")
    if named:
        candidate = Path(named)
        if candidate.exists():
            return candidate
    cases_file = summary.get("cases_file")
    if cases_file:
        candidate = Path(cases_file).parent / "preparation.json"
        if candidate.exists():
            return candidate
    candidate = summary_path.parent.parent / "cases" / "preparation.json"
    return candidate if candidate.exists() else None


def build_row(summary_path: Path, prep_override: Path | None, raw_override: Path | None) -> dict[str, str]:
    summary = read_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError(f"Evaluation is not completed (status={summary.get('status')!r})")
    required = (
        "dictionary_sha256",
        "cases_sha256",
        "harness_sha256",
        "cases",
        "reference_scalars",
        "final_character_accuracy",
        "nbest_exact_inclusion_rate",
        "final_top_exact_rate",
    )
    missing = [key for key in required if summary.get(key) is None or summary.get(key) == ""]
    if missing:
        raise ValueError(f"Evaluation summary is missing required history fields: {', '.join(missing)}")

    run_file = summary_path.parent / "run.json"
    if run_file.exists():
        run = read_json(run_file)
        if run.get("status") != "completed":
            raise ValueError(f"Run metadata is not completed (status={run.get('status')!r})")

    completed_at = summary.get("completed_at")
    if not completed_at:
        raise ValueError("summary.json has no completed_at timestamp")
    completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

    prep_file = preparation_path(summary, summary_path, prep_override)
    prep = read_json(prep_file) if prep_file else {}
    if prep and prep.get("status") != "completed":
        raise ValueError(f"Preparation metadata is not completed (status={prep.get('status')!r})")
    if prep.get("cases_sha256") and summary.get("cases_sha256") != prep["cases_sha256"]:
        raise ValueError("Preparation and evaluation cases_sha256 do not match")

    raw_file = raw_override or summary_path.parent / "raw-score-summary.json"
    if raw_override and not raw_file.exists():
        raise FileNotFoundError(f"Raw score summary not found: {raw_file}")
    raw = read_json(raw_file) if raw_file.exists() else {}
    if raw.get("cases") is not None and raw.get("cases") != summary.get("cases"):
        raise ValueError("Raw score summary case count does not match the evaluation")

    viterbi = raw.get("viterbi_bigram_top1", {})
    reranked = raw.get("reranked_nbest_trigram_top1", {})
    margin = raw.get("reranked_top1_minus_top2", {})
    timing = summary.get("conversion_time", {})
    counts = summary.get("counts", {})
    source_sample_hash = prep.get("source_sample_sha256", "")

    values: dict[str, Any] = {
        "evaluation_date": completed.date().isoformat(),
        "completed_at": completed_at,
        "dictionary_sha256": summary.get("dictionary_sha256", ""),
        "source_sample_sha256": source_sample_hash,
        "cases_sha256": summary.get("cases_sha256", ""),
        "harness_sha256": summary.get("harness_sha256", ""),
        "segmentation_policy": prep.get("segmentation_policy", ""),
        "held_out": str(prep.get("held_out", summary.get("measurement", {}).get("held_out", ""))).lower(),
        "cases": summary.get("cases", ""),
        "reference_scalars": summary.get("reference_scalars", ""),
        "final_edits": counts.get("final_edits", ""),
        "final_character_accuracy_pct": pct(summary.get("final_character_accuracy")),
        "final_character_error_rate_pct": pct(summary.get("final_character_error_rate")),
        "nbest_oracle_character_accuracy_pct": pct(summary.get("nbest_oracle_character_accuracy")),
        "nbest_exact_inclusion_rate_pct": pct(summary.get("nbest_exact_inclusion_rate")),
        "final_top_exact_rate_pct": pct(summary.get("final_top_exact_rate")),
        "reranked_top_exact_rate_pct": pct(summary.get("reranked_top_exact_rate")),
        "viterbi_top_exact_rate_pct": pct(summary.get("viterbi_top_exact_rate")),
        "mean_unique_nbest_surfaces": number(summary.get("mean_unique_nbest_surfaces")),
        "viterbi_bigram_top1_mean_raw_score": number(viterbi.get("mean")),
        "viterbi_bigram_top1_p50_raw_score": number(viterbi.get("p50")),
        "viterbi_bigram_top1_p95_raw_score": number(viterbi.get("p95")),
        "reranked_nbest_trigram_top1_mean_raw_score": number(reranked.get("mean")),
        "reranked_nbest_trigram_top1_p50_raw_score": number(reranked.get("p50")),
        "reranked_nbest_trigram_top1_p95_raw_score": number(reranked.get("p95")),
        "reranked_top1_minus_top2_mean_raw_score": number(margin.get("mean")),
        "reranked_top1_minus_top2_p50_raw_score": number(margin.get("p50")),
        "reranked_top1_minus_top2_p95_raw_score": number(margin.get("p95")),
        "conversion_time_mean_ms": number(timing.get("mean_ms")),
        "conversion_time_p50_ms": number(timing.get("p50_ms")),
        "conversion_time_p95_ms": number(timing.get("p95_ms")),
        "conversion_time_p99_ms": number(timing.get("p99_ms")),
        "conversion_errors": error_count(summary.get("errors", {})),
        "summary_sha256": digest(summary_path),
    }
    return {key: str(values.get(key, "")) for key in FIELDS}


def append_row(history_path: Path, row: dict[str, str]) -> bool:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing history CSV has a different header; refusing to rewrite it")
            existing = list(reader)
        if any(item.get("summary_sha256") == row["summary_sha256"] for item in existing):
            return False

    fd, temp_name = tempfile.mkstemp(prefix=history_path.name + ".", suffix=".tmp", dir=history_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, history_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a completed coltIme corpus conversion evaluation to CSV history.")
    parser.add_argument("--summary", required=True, type=Path, help="Completed evaluation summary.json")
    parser.add_argument("--preparation", type=Path, help="Optional preparation.json override")
    parser.add_argument("--raw-scores", type=Path, help="Optional raw-score-summary.json override")
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY, help=f"History CSV (default: {DEFAULT_HISTORY})")
    args = parser.parse_args()

    row = build_row(args.summary, args.preparation, args.raw_scores)
    added = append_row(args.history_csv, row)
    print(json.dumps({"ok": True, "added": added, "history_csv": str(args.history_csv), "evaluation_date": row["evaluation_date"], "dictionary_sha256": row["dictionary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
