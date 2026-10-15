# evaluation/tasks/end_to_end/metrics.py

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from evaluation.tasks.end_to_end.contract import CONTRACT_VERSION, LEGACY_CONTRACT_VERSION


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


def _confusion(rows: list[dict[str, Any]], predicted_key: str) -> dict[str, int]:
    counts = Counter((row["expected_decision"], row[predicted_key]) for row in rows)
    return {f"{expected}->{predicted}": n for (expected, predicted), n in sorted(counts.items())}


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    decision (current contract):
    - accuracy: strict three-way match with the golden label;
    - critical_false_yes: golden NO/UNCERTAIN, system claimed YES - the
      primary safety metric (rate over NO+UNCERTAIN cases);
    - false_no: golden YES, system showed nothing (rate over YES cases);
    - conservative: system answered UNCERTAIN where the golden label is YES
      or NO - wrong, but it did not claim anything false.
    legacy: the pre-v2 "any result = YES" contract, for comparison only.
    """
    total = len(rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    error_rows = [row for row in rows if row["status"] == "error"]

    yes_rows = [row for row in ok_rows if row["expected_decision"] == "YES"]
    rejection_rows = [row for row in ok_rows if row["expected_decision"] in ("NO", "UNCERTAIN")]

    decision_correct = sum(row["decision_correct"] for row in ok_rows)
    critical_false_yes = sum(row["critical_false_yes"] for row in ok_rows)
    false_no = sum(row["predicted_decision"] == "NO" for row in yes_rows)
    yes_to_uncertain = sum(row["predicted_decision"] == "UNCERTAIN" for row in yes_rows)
    no_to_uncertain = sum(
        row["expected_decision"] == "NO" and row["predicted_decision"] == "UNCERTAIN"
        for row in ok_rows
    )

    legacy_correct = sum(row["legacy_decision_correct"] for row in ok_rows)
    legacy_critical_false_yes = sum(row["legacy_critical_false_yes"] for row in ok_rows)

    top1 = sum(row["selection_correct"] is True for row in yes_rows)
    topk = sum(_top_k_contains_acceptable(row) for row in yes_rows)
    mrr_sum = sum(_reciprocal_rank(row) for row in yes_rows)

    runtimes = [row["runtime_ms"] for row in ok_rows if row.get("runtime_ms") is not None]
    costs = [_extract_cost(row) for row in ok_rows]
    llm_calls = [_extract_llm_calls(row) for row in ok_rows]

    return {
        "total": total,
        "ok_total": len(ok_rows),
        "error_total": len(error_rows),
        "error_rate": _safe_rate(len(error_rows), total),
        "error_types": dict(Counter(row["error"]["type"] for row in error_rows if row.get("error"))),

        "decision": {
            "contract_version": CONTRACT_VERSION,
            "accuracy": _safe_rate(decision_correct, len(ok_rows)),
            "correct_count": decision_correct,
            "yes_total": len(yes_rows),
            "rejection_total": len(rejection_rows),
            "critical_false_yes_count": critical_false_yes,
            "critical_false_yes_rate": _safe_rate(critical_false_yes, len(rejection_rows)),
            "false_no_count": false_no,
            "false_no_rate": _safe_rate(false_no, len(yes_rows)),
            "conservative": {
                "yes_to_uncertain": yes_to_uncertain,
                "no_to_uncertain": no_to_uncertain,
            },
            "basis_counts": dict(Counter(row["decision_basis"] for row in ok_rows)),
            "confusion_matrix": _confusion(ok_rows, "predicted_decision"),
        },

        "selection": {
            "yes_total": len(yes_rows),
            "top1_correct_count": top1,
            "top1_accuracy": _safe_rate(top1, len(yes_rows)),
            "topk_contains_acceptable_count": topk,
            "topk_contains_acceptable_rate": _safe_rate(topk, len(yes_rows)),
            "mrr": _safe_rate(mrr_sum, len(yes_rows)),
        },

        "legacy": {
            "contract_version": LEGACY_CONTRACT_VERSION,
            "note": "Any non-empty result counted as YES. For comparison with pre-v2 reports only.",
            "decision_accuracy": _safe_rate(legacy_correct, len(ok_rows)),
            "critical_false_yes_count": legacy_critical_false_yes,
            "critical_false_yes_rate": _safe_rate(legacy_critical_false_yes, len(rejection_rows)),
            "confusion_matrix": _confusion(ok_rows, "legacy_predicted_decision"),
        },

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
    }
