# evaluation/tasks/end_to_end/comparator.py

from __future__ import annotations

from typing import Any

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


def compare_case(
    *,
    case: EndToEndEvalCase,
    prediction_record: dict[str, Any],
) -> dict[str, Any]:
    if prediction_record["status"] != "ok":
        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "query_style": case.query_style,
            "user_query": case.user_query,

            "status": "error",
            "error": prediction_record.get("error"),

            "expected_decision": case.expected_decision,
            "predicted_decision": None,
            "decision_correct": False,

            "expected_selected_id": case.expected_selected_id,
            "acceptable_ids": case.acceptable_ids,
            "predicted_selected_id": None,
            "predicted_selected_ids": [],

            "should_check_selection": case.expected_decision == "YES",
            "selection_correct": False if case.expected_decision == "YES" else None,

            "critical_false_yes": False,

            "runtime_ms": prediction_record.get("runtime_ms"),
            "hard_constraints": case.hard_constraints,
            "gold_reason": case.reason,
            "raw_response": None,
        }

    prediction = prediction_record["prediction"]

    expected_decision = case.expected_decision
    predicted_decision = prediction["decision"]

    decision_correct = predicted_decision == expected_decision

    should_check_selection = expected_decision == "YES"

    selection_correct = _is_selected_id_acceptable(
        predicted_id=prediction.get("selected_listing_id"),
        expected_selected_id=case.expected_selected_id,
        acceptable_ids=case.acceptable_ids,
    )

    critical_false_yes = (
        expected_decision in {"NO", "UNCERTAIN"}
        and predicted_decision == "YES"
    )

    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "query_style": case.query_style,
        "user_query": case.user_query,

        "status": "ok",

        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "decision_correct": decision_correct,

        "expected_selected_id": case.expected_selected_id,
        "acceptable_ids": case.acceptable_ids,
        "predicted_selected_id": prediction.get("selected_listing_id"),
        "predicted_selected_ids": prediction.get("selected_listing_ids", []),

        "should_check_selection": should_check_selection,
        "selection_correct": selection_correct if should_check_selection else None,

        "critical_false_yes": critical_false_yes,

        "runtime_ms": prediction_record.get("runtime_ms"),
        "hard_constraints": case.hard_constraints,
        "gold_reason": case.reason,
        "raw_response": prediction.get("raw_response"),
    }