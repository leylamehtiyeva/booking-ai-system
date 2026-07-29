# evaluation/tasks/end_to_end/report.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(
    *,
    data: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_jsonl(
    *,
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    for row in rows:
        is_failure = (
            row["status"] == "error"
            or row["decision_correct"] is False
            or row["selection_correct"] is False
        )

        if is_failure:
            failures.append(row)

    return failures


def save_evaluation_report(
    *,
    evaluation_result: dict[str, Any],
    report_path: str | Path,
    failures_path: str | Path,
) -> None:
    metrics = evaluation_result["metrics"]
    rows = evaluation_result["rows"]

    failures = build_failures(rows)

    report = {
        "metrics": metrics,
        "failure_count": len(failures),
        "failures_path": str(failures_path),
    }

    save_json(
        data=report,
        path=report_path,
    )

    save_jsonl(
        rows=failures,
        path=failures_path,
    )