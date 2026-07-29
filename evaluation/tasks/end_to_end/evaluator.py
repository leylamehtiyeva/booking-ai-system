# evaluation/tasks/end_to_end/evaluator.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.tasks.end_to_end.comparator import compare_case
from evaluation.tasks.end_to_end.dataset import load_end_to_end_dataset
from evaluation.tasks.end_to_end.metrics import compute_metrics


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    predictions_by_case_id: dict[str, dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            predictions_by_case_id[record["case_id"]] = record

    return predictions_by_case_id


def evaluate_predictions(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    cases = load_end_to_end_dataset(dataset_path)

    if limit is not None:
        cases = cases[:limit]

    predictions_by_case_id = _load_predictions(predictions_path)

    rows: list[dict[str, Any]] = []

    for case in cases:
        prediction_record = predictions_by_case_id.get(case.case_id)

        if prediction_record is None:
            prediction_record = {
                "case_id": case.case_id,
                "status": "error",
                "runtime_ms": None,
                "error": {
                    "type": "MissingPrediction",
                    "message": "Prediction record was not found.",
                },
            }

        row = compare_case(
            case=case,
            prediction_record=prediction_record,
        )

        rows.append(row)

    metrics = compute_metrics(rows)

    return {
        "metrics": metrics,
        "rows": rows,
    }