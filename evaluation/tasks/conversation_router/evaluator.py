from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from evaluation.core.io import load_jsonl
from evaluation.tasks.conversation_router.metrics import (
    calculate_profile_metrics,
)
from evaluation.tasks.conversation_router.runner import (
    RouterEvalPrediction,
)


def load_router_predictions(
    path: str | Path,
) -> list[RouterEvalPrediction]:
    rows = load_jsonl(path)

    return [
        RouterEvalPrediction.model_validate(row)
        for row in rows
    ]


def evaluate_predictions(
    predictions: list[RouterEvalPrediction],
) -> dict:
    if not predictions:
        raise ValueError(
            "No router predictions to evaluate"
        )

    grouped = defaultdict(list)

    for prediction in predictions:
        grouped[prediction.profile].append(
            prediction
        )

    profile_metrics = {
        profile: calculate_profile_metrics(rows)
        for profile, rows in sorted(grouped.items())
    }

    failures = [
        prediction
        for prediction in predictions
        if not prediction.correct
    ]

    return {
        "total_prediction_rows": len(predictions),
        "profiles": profile_metrics,
        "failures_count": len(failures),
        "failures": failures,
    }