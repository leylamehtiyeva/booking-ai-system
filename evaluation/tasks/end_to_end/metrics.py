# evaluation/tasks/end_to_end/metrics.py

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _get_allowed_ids(row: dict[str, Any]) -> set[str]:
    allowed_ids = set(row.get("acceptable_ids") or [])

    expected_selected_id = row.get("expected_selected_id")
    if expected_selected_id:
        allowed_ids.add(expected_selected_id)

    return allowed_ids


def _top_k_contains_acceptable(row: dict[str, Any]) -> bool:
    allowed_ids = _get_allowed_ids(row)
    predicted_ids = row.get("predicted_selected_ids") or []

    return any(listing_id in allowed_ids for listing_id in predicted_ids)


def _reciprocal_rank(row: dict[str, Any]) -> float:
    allowed_ids = _get_allowed_ids(row)
    predicted_ids = row.get("predicted_selected_ids") or []

    for idx, listing_id in enumerate(predicted_ids, start=1):
        if listing_id in allowed_ids:
            return 1 / idx

    return 0.0


def _extract_cost(row: dict[str, Any]) -> float:
    raw_response = row.get("raw_response") or {}
    telemetry = raw_response.get("telemetry") or {}
    cost = telemetry.get("cost") or {}

    return float(cost.get("estimated_total_usd") or 0.0)


def _extract_llm_calls(row: dict[str, Any]) -> int:
    raw_response = row.get("raw_response") or {}
    telemetry = raw_response.get("telemetry") or {}
    scenario = telemetry.get("scenario") or {}

    return int(scenario.get("llm_calls_count") or 0)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)

    ok_rows = [
        row for row in rows
        if row["status"] == "ok"
    ]

    error_rows = [
        row for row in rows
        if row["status"] == "error"
    ]

    decision_correct_count = sum(
        row["decision_correct"]
        for row in ok_rows
    )

    yes_rows = [
        row for row in ok_rows
        if row["expected_decision"] == "YES"
    ]

    top1_selection_correct_count = sum(
        row["selection_correct"] is True
        for row in yes_rows
    )

    topk_contains_acceptable_count = sum(
        _top_k_contains_acceptable(row)
        for row in yes_rows
    )

    mrr_sum = sum(
        _reciprocal_rank(row)
        for row in yes_rows
    )

    rejection_rows = [
        row for row in ok_rows
        if row["expected_decision"] in ("NO", "UNCERTAIN")
    ]

    critical_false_yes_count = sum(
        row["critical_false_yes"]
        for row in ok_rows
    )

    confusion_matrix = Counter(
        (
            row["expected_decision"],
            row["predicted_decision"],
        )
        for row in ok_rows
    )

    error_types = Counter(
        row["error"]["type"]
        for row in error_rows
        if row.get("error")
    )

    runtimes = [
        row["runtime_ms"]
        for row in ok_rows
        if row.get("runtime_ms") is not None
    ]

    costs = [
        _extract_cost(row)
        for row in ok_rows
    ]

    llm_calls = [
        _extract_llm_calls(row)
        for row in ok_rows
    ]

    return {
        "total": total,

        "ok_total": len(ok_rows),
        "error_total": len(error_rows),
        "error_rate": _safe_rate(len(error_rows), total),
        "error_types": dict(error_types),

        "decision_correct_count": decision_correct_count,
        "decision_accuracy": _safe_rate(
            decision_correct_count,
            len(ok_rows),
        ),

        "yes_total": len(yes_rows),

        "top1_selection_correct_count": top1_selection_correct_count,
        "top1_selection_accuracy": _safe_rate(
            top1_selection_correct_count,
            len(yes_rows),
        ),

        "topk_contains_acceptable_count": topk_contains_acceptable_count,
        "topk_contains_acceptable_rate": _safe_rate(
            topk_contains_acceptable_count,
            len(yes_rows),
        ),

        "mrr": _safe_rate(
            mrr_sum,
            len(yes_rows),
        ),

        "rejection_total": len(rejection_rows),
        "critical_false_yes_count": critical_false_yes_count,
        "critical_false_yes_rate": _safe_rate(
            critical_false_yes_count,
            len(rejection_rows),
        ),

        "runtime_ms": {
            "mean": mean(runtimes) if runtimes else 0.0,
            "max": max(runtimes) if runtimes else 0,
        },

        "cost": {
            "total_estimated_usd": sum(costs),
            "mean_estimated_usd": mean(costs) if costs else 0.0,
        },

        "llm_calls": {
            "total": sum(llm_calls),
            "mean": mean(llm_calls) if llm_calls else 0.0,
        },

        "confusion_matrix": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in confusion_matrix.items()
        },
    }