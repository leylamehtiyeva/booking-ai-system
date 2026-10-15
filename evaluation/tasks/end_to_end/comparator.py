# evaluation/tasks/end_to_end/comparator.py

from __future__ import annotations

from typing import Any

from evaluation.tasks.end_to_end.contract import (
    infer_decision,
    infer_legacy_decision,
    selected_listing_ids,
)
from evaluation.tasks.end_to_end.dataset import EndToEndEvalCase


def _is_selected_id_acceptable(
    *,
    predicted_id: str | None,
    expected_selected_id: str | None,
    acceptable_ids: list[str],
) -> bool:
    if predicted_id is None:
        return False

    allowed_ids = set(acceptable_ids)

    if expected_selected_id:
        allowed_ids.add(expected_selected_id)

    return predicted_id in allowed_ids


def _error_row(case: EndToEndEvalCase, prediction_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "query_style": case.query_style,
        "user_query": case.user_query,

        "status": "error",
        "error": prediction_record.get("error"),

        "expected_decision": case.expected_decision,
        "predicted_decision": None,
        "decision_basis": None,
        "decision_correct": False,
        "legacy_predicted_decision": None,
        "legacy_decision_correct": False,

        "expected_selected_id": case.expected_selected_id,
        "acceptable_ids": case.acceptable_ids,
        "predicted_selected_id": None,
        "predicted_selected_ids": [],

        "should_check_selection": case.expected_decision == "YES",
        "selection_correct": False if case.expected_decision == "YES" else None,

        "critical_false_yes": False,
        "legacy_critical_false_yes": False,

        "runtime_ms": prediction_record.get("runtime_ms"),
        "hard_constraints": case.hard_constraints,
        "gold_reason": case.reason,
        "raw_response": None,
    }


def compare_case(
    *,
    case: EndToEndEvalCase,
    prediction_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Decisions are re-derived from the saved raw_response with the current
    contract (and the legacy one) rather than trusted from the stored
    "decision" field, so predictions from any run can be re-scored.
    """
    if prediction_record["status"] != "ok":
        return _error_row(case, prediction_record)

    raw_response = prediction_record["prediction"].get("raw_response") or {}

    predicted_decision, decision_basis = infer_decision(raw_response)
    if predicted_decision is None:
        return _error_row(
            case,
            {
                **prediction_record,
                "error": {"type": "TurnFailure", "message": decision_basis},
            },
        )

    legacy_decision = infer_legacy_decision(raw_response)
    selected_ids = selected_listing_ids(raw_response)
    predicted_id = selected_ids[0] if selected_ids else None

    expected_decision = case.expected_decision
    should_check_selection = expected_decision == "YES"

    selection_correct = _is_selected_id_acceptable(
        predicted_id=predicted_id,
        expected_selected_id=case.expected_selected_id,
        acceptable_ids=case.acceptable_ids,
    )

    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "query_style": case.query_style,
        "user_query": case.user_query,

        "status": "ok",

        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "decision_basis": decision_basis,
        "decision_correct": predicted_decision == expected_decision,
        "legacy_predicted_decision": legacy_decision,
        "legacy_decision_correct": legacy_decision == expected_decision,

        "expected_selected_id": case.expected_selected_id,
        "acceptable_ids": case.acceptable_ids,
        "predicted_selected_id": predicted_id,
        "predicted_selected_ids": selected_ids,

        "should_check_selection": should_check_selection,
        "selection_correct": selection_correct if should_check_selection else None,

        "critical_false_yes": (
            expected_decision in {"NO", "UNCERTAIN"} and predicted_decision == "YES"
        ),
        "legacy_critical_false_yes": (
            expected_decision in {"NO", "UNCERTAIN"} and legacy_decision == "YES"
        ),

        "runtime_ms": prediction_record.get("runtime_ms"),
        "hard_constraints": case.hard_constraints,
        "gold_reason": case.reason,
        "raw_response": raw_response,
    }
