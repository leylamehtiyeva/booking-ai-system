"""
End-to-end evaluation contract (P0-7): decision inference, scoring,
metrics, run metadata and the LLM-free replay - all on synthetic data,
no network.
"""

from __future__ import annotations

import json

import pytest

from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from evaluation.tasks.end_to_end import adapter
from evaluation.tasks.end_to_end.comparator import compare_case
from evaluation.tasks.end_to_end.contract import (
    CONTRACT_VERSION,
    infer_decision,
    infer_legacy_decision,
)
from evaluation.tasks.end_to_end.dataset import EndToEndEvalCase
from evaluation.tasks.end_to_end.evaluator import evaluate_predictions
from evaluation.tasks.end_to_end.metrics import compute_metrics
from evaluation.tasks.end_to_end.replay import replay_predictions
from evaluation.tasks.end_to_end.run_config import EvalRunConfig, build_run_metadata


def _result(result_id: str, tier: str = "strong", eligibility: str = "eligible") -> dict:
    return {"result_id": result_id, "title": result_id, "score": 1.0, "match_tier": tier, "eligibility_status": eligibility}


def _search(*results: dict) -> dict:
    return {"status": "results" if results else "no_results", "results": list(results)}


# --- decision contract ------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "decision", "basis"),
    [
        (_search(_result("a", "strong"), _result("b", "partial")), "YES", "strong_match"),
        (_search(_result("a", "partial")), "UNCERTAIN", "partial_match"),
        (_search(_result("a", "weak")), "UNCERTAIN", "weak_match"),
        (_search(), "NO", "no_results"),
        ({"need_clarification": True, "questions": ["Which dates?"]}, "UNCERTAIN", "clarification"),
        (_search(_result("a", "strong", eligibility="ineligible")), "UNCERTAIN", "ineligible_result_shown"),
        (_search({"result_id": "a", "title": "a", "score": 1.0}), "UNCERTAIN", "unclassified_result"),
        ({"response_type": "processing_unavailable", "failure": {"stage": "retrieval", "code": "timeout"}}, None, "turn_failure"),
        ({"response_type": "routing_unavailable"}, None, "turn_failure"),
    ],
)
def test_decision_contract(response, decision, basis):
    assert infer_decision(response) == (decision, basis)


@pytest.mark.parametrize(
    ("response", "legacy"),
    [
        (_search(_result("a", "partial")), "YES"),
        (_search(), "NO"),
        ({"need_clarification": True, "questions": ["Which dates?"]}, "UNCERTAIN"),
        ({"need_clarification": True, "questions": [], "debug_notes": ["No listings remained"]}, "NO"),
    ],
)
def test_legacy_contract_is_unchanged(response, legacy):
    assert infer_legacy_decision(response) == legacy


# --- comparator + metrics ------------------------------------------------------------


def _case(case_id: str, expected: str, acceptable: list[str] | None = None) -> EndToEndEvalCase:
    return EndToEndEvalCase(
        case_id=case_id,
        case_type="t",
        query_style="t",
        user_query="q",
        expected_decision=expected,
        expected_selected_id=(acceptable or [None])[0],
        acceptable_ids=acceptable or [],
        hard_constraints={},
    )


def _ok(response: dict) -> dict:
    # A stored "decision" that disagrees on purpose: the comparator must
    # re-derive decisions from raw_response, never trust the stored field.
    return {"status": "ok", "runtime_ms": 10, "prediction": {"decision": "NO", "raw_response": response}}


CASES_AND_RESPONSES = [
    (_case("yes-strong-right", "YES", ["a"]), _ok(_search(_result("a", "strong")))),
    (_case("yes-partial", "YES", ["a"]), _ok(_search(_result("a", "partial")))),
    (_case("yes-none", "YES", ["a"]), _ok(_search())),
    (_case("no-partial", "NO"), _ok(_search(_result("x", "partial")))),
    (_case("no-strong", "NO"), _ok(_search(_result("x", "strong")))),
    (_case("unc-strong", "UNCERTAIN"), _ok(_search(_result("x", "strong")))),
    (_case("unc-weak", "UNCERTAIN"), _ok(_search(_result("x", "weak")))),
    (_case("no-none", "NO"), _ok(_search())),
    (
        _case("failure", "YES", ["a"]),
        _ok({"response_type": "processing_unavailable", "failure": {"stage": "retrieval", "code": "timeout"}}),
    ),
]


@pytest.fixture
def metrics():
    rows = [compare_case(case=case, prediction_record=record) for case, record in CASES_AND_RESPONSES]
    return compute_metrics(rows)


def test_decision_metrics(metrics):
    decision = metrics["decision"]
    assert decision["contract_version"] == CONTRACT_VERSION
    assert (metrics["ok_total"], metrics["error_total"]) == (8, 1)
    assert decision["correct_count"] == 3  # yes-strong-right, unc-weak, no-none
    assert decision["accuracy"] == pytest.approx(3 / 8)
    assert decision["critical_false_yes_count"] == 2  # no-strong, unc-strong
    assert decision["rejection_total"] == 5
    assert decision["false_no_count"] == 1  # yes-none
    assert decision["conservative"] == {"yes_to_uncertain": 1, "no_to_uncertain": 1}
    assert decision["confusion_matrix"]["YES->UNCERTAIN"] == 1
    assert decision["basis_counts"]["strong_match"] == 3


def test_selection_metrics(metrics):
    selection = metrics["selection"]
    assert selection["yes_total"] == 3
    assert selection["top1_correct_count"] == 2  # strong-right + partial (both top "a")


def test_legacy_metrics_are_reported_separately(metrics):
    legacy = metrics["legacy"]
    # any result = YES: right only for yes-strong-right, yes-partial, no-none
    assert legacy["decision_accuracy"] == pytest.approx(3 / 8)
    assert legacy["critical_false_yes_count"] == 4  # every NO/UNCERTAIN case with results


def test_turn_failure_is_an_error_not_a_decision(metrics):
    assert metrics["error_types"] == {"TurnFailure": 1}
    assert sum(metrics["decision"]["confusion_matrix"].values()) == metrics["ok_total"] == 8


def test_evaluate_rescores_saved_predictions_from_raw_response(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "case_type": "t",
                    "query_style": "t",
                    "user_query": "q",
                    "expected_decision": case.expected_decision,
                    "expected_selected_id": case.expected_selected_id,
                    "acceptable_ids": ", ".join(case.acceptable_ids),
                    "hard_constraints": "{}",
                }
            )
            for case, _ in CASES_AND_RESPONSES
        )
    )
    predictions.write_text(
        "\n".join(json.dumps({"case_id": case.case_id, **record}) for case, record in CASES_AND_RESPONSES)
    )

    result = evaluate_predictions(dataset_path=dataset, predictions_path=predictions)

    assert result["metrics"]["decision"]["critical_false_yes_count"] == 2
    assert result["metrics"]["legacy"]["critical_false_yes_count"] == 4


# --- adapter ---------------------------------------------------------------------------


def _route(action):
    async def fake(**kwargs):
        return ConversationActionDecision(
            action=action,
            reason="t",
            decision_status=ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.NONE,
            result_scope=ResultScope.NOT_APPLICABLE,
            target_result_id="",
        )

    return fake


async def test_adapter_uses_production_settings_and_records_both_decisions(monkeypatch):
    captured = {}

    async def fake_handle_user_message(**kwargs):
        captured.update(kwargs)
        return _search(_result("a", "partial"))

    monkeypatch.setattr(adapter, "handle_user_message", fake_handle_user_message)

    prediction = await adapter.run_case(_case("c", "YES", ["a"]))

    config = EvalRunConfig()
    assert captured["top_n"] == config.top_n
    assert captured["max_items"] == config.max_items
    assert captured["fallback_policy"] == config.fallback_policy()
    assert (prediction.decision, prediction.decision_basis, prediction.legacy_decision) == (
        "UNCERTAIN",
        "partial_match",
        "YES",
    )
    assert prediction.contract_version == CONTRACT_VERSION


async def test_adapter_raises_on_typed_turn_failure(monkeypatch):
    async def failing(**kwargs):
        return {"response_type": "processing_unavailable", "failure": {"stage": "search_intent_extraction", "code": "rate_limited"}}

    monkeypatch.setattr(adapter, "handle_user_message", failing)

    with pytest.raises(adapter.EvalTurnFailure, match="search_intent_extraction:rate_limited"):
        await adapter.run_case(_case("c", "YES", ["a"]))


# --- run metadata -----------------------------------------------------------------------


def test_run_metadata_describes_backend_models_and_settings(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("{}\n")

    metadata = build_run_metadata(config=EvalRunConfig(top_n=3), dataset_path=dataset, case_limit=10)

    assert metadata["contract_version"] == CONTRACT_VERSION
    assert set(metadata["git"]) == {"commit", "dirty"}
    assert len(metadata["dataset"]["sha256"]) == 64
    assert metadata["dataset"]["case_limit"] == 10
    assert metadata["reference_date"] == "2026-03-01"
    assert metadata["run_config"]["top_n"] == 3
    assert metadata["run_config"]["fallback_policy"]["enabled"] is True
    assert set(metadata["models"]) == {"conversation_router_profile", "conversation_router", "intent_extraction", "intent_update", "textual_fallback"}
    json.dumps(metadata)


# --- replay ------------------------------------------------------------------------------


async def test_replay_runs_saved_intents_through_the_current_backend_without_llm(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "baku",
                "case_type": "t",
                "query_style": "t",
                "user_query": "apartment in Baku with WiFi",
                "expected_decision": "YES",
                "expected_selected_id": "baku-apt-1",
                "acceptable_ids": "baku-apt-1, baku-apt-3",
                "hard_constraints": "{}",
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "no-intent",
                "case_type": "t",
                "query_style": "t",
                "user_query": "q",
                "expected_decision": "NO",
                "hard_constraints": "{}",
            }
        )
    )
    state = {
        "city": "Baku",
        "check_in": "2026-04-10",
        "check_out": "2026-04-15",
        "property_types": ["apartment"],
        "constraints": [
            {
                "raw_text": "WiFi",
                "normalized_text": "wifi",
                "priority": "must",
                "category": "amenity",
                "mapping_status": "known",
                "mapped_fields": ["wifi"],
                "evidence_strategy": "structured",
            }
        ],
    }
    predictions.write_text(
        json.dumps({"case_id": "baku", "status": "ok", "prediction": {"raw_response": {"state": state}}})
        + "\n"
        + json.dumps({"case_id": "no-intent", "status": "error", "error": {"type": "X"}})
    )

    result = await replay_predictions(dataset_path=dataset, predictions_path=predictions)

    assert result["skipped_case_ids"] == ["no-intent"]
    (row,) = result["rows"]
    assert row["predicted_decision"] == "YES"
    assert row["predicted_selected_id"] in {"baku-apt-1", "baku-apt-3"}
    assert row["selection_correct"] is True
    assert result["replay"]["textual_fallback"] == "disabled"


# --- spend guard ----------------------------------------------------------------------


async def test_runner_stops_starting_cases_once_the_llm_call_budget_is_reached(tmp_path, monkeypatch):
    from evaluation.tasks.end_to_end import runner

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": f"c{i}",
                    "case_type": "t",
                    "query_style": "t",
                    "user_query": "q",
                    "expected_decision": "NO",
                    "hard_constraints": "{}",
                }
            )
            for i in range(5)
        )
    )
    calls_made = []

    async def fake_run_case(case, *, config):
        calls_made.append(case.case_id)
        return adapter.EndToEndPrediction(
            decision="NO",
            decision_basis="no_results",
            legacy_decision="NO",
            raw_response={"results": [], "telemetry": {"llm": {"calls_count": 3}}},
        )

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    output = tmp_path / "out" / "predictions.jsonl"

    await runner.run_predictions(dataset_path=dataset, output_path=output, max_llm_calls=7)

    assert calls_made == ["c0", "c1", "c2"]  # 3 + 3 < 7, then 9 >= 7 stops
    assert len(output.read_text().splitlines()) == 3
    execution = json.loads((output.parent / runner.RUN_METADATA_FILENAME).read_text())["execution"]
    assert execution["status"] == "stopped"
    assert (execution["cases_completed"], execution["llm_calls_used"]) == (3, 9)
