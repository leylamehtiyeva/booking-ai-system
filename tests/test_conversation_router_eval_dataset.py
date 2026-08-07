from __future__ import annotations
from pathlib import Path
from evaluation.tasks.conversation_router.evaluator import evaluate_predictions
from evaluation.tasks.conversation_router.metrics import (
    calculate_profile_metrics,
)
from app.schemas.conversation_route import ConversationAction
from evaluation.tasks.conversation_router.dataset import (
    load_conversation_router_dataset,
)


DATASET_PATH = Path(
    "evaluation/datasets/conversation_router/router_eval_dev.jsonl"
)


def test_load_conversation_router_dataset():
    cases = load_conversation_router_dataset(DATASET_PATH)

    assert len(cases) == 70

    first_case = cases[0]

    assert first_case.id == "router-001"
    assert first_case.user_message == "Find an apartment in Baku"
    assert first_case.has_current_search is False
    assert first_case.has_shown_results is False
    assert first_case.expected_action == ConversationAction.START_SEARCH
    
    
from app.schemas.query import SearchRequest

from evaluation.tasks.conversation_router.adapter import (
    build_router_input,
)
from evaluation.tasks.conversation_router.dataset import (
    ConversationRouterEvalCase,
)


def test_build_router_input_without_existing_state():
    case = ConversationRouterEvalCase(
        id="router-test-1",
        user_message="Find me a hotel in Rome",
        has_current_search=False,
        has_shown_results=False,
        expected_action="start_search",
        category="first_search",
        notes="Test case",
    )

    router_input = build_router_input(case)

    assert router_input.user_message == case.user_message
    assert router_input.current_search is None
    assert router_input.latest_result_context is None


def test_build_router_input_with_existing_state():
    case = ConversationRouterEvalCase(
        id="router-test-2",
        user_message="Does the second one have parking?",
        has_current_search=True,
        has_shown_results=True,
        expected_action="listing_question",
        category="listing_question",
        notes="Test case",
    )

    router_input = build_router_input(case)

    assert isinstance(router_input.current_search, SearchRequest)
    assert router_input.latest_result_context == {
        "has_shown_results": True,
    }
    
    
import asyncio

from app.logic.conversation_router import ConversationRoutingError
from app.observability.trace import LLMCallTrace
from app.schemas.conversation_route import (
    ConversationAction,
    ConversationActionDecision,
)
from evaluation.tasks.conversation_router.dataset import (
    ConversationRouterEvalCase,
)
from evaluation.tasks.conversation_router.runner import (
    run_router_eval_case,
)


def _make_case() -> ConversationRouterEvalCase:
    return ConversationRouterEvalCase(
        id="router-test-1",
        user_message="Use Tbilisi instead",
        has_current_search=True,
        has_shown_results=False,
        expected_action="update_search",
        category="change_city",
        notes="Test case",
    )


def test_run_router_eval_case_success(monkeypatch):
    async def fake_route_conversation_async(
        *,
        router_input,
        trace,
        llm_profile_name,
    ):
        assert router_input.current_search is not None
        assert llm_profile_name == "groq_gpt_oss_20b"

        trace.add_llm_call(
            LLMCallTrace(
                step="conversation_routing",
                model="groq/openai/gpt-oss-20b",
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                estimated_cost_usd=0.0000135,
                success=True,
            )
        )

        return ConversationActionDecision(
            action=ConversationAction.UPDATE_SEARCH,
            reason="The user changes the current city.",
        )

    monkeypatch.setattr(
        "evaluation.tasks.conversation_router.runner."
        "route_conversation_async",
        fake_route_conversation_async,
    )

    prediction = asyncio.run(
        run_router_eval_case(
            case=_make_case(),
            profile="groq_gpt_oss_20b",
            repeat_number=1,
        )
    )

    assert prediction.case_id == "router-test-1"
    assert prediction.profile == "groq_gpt_oss_20b"

    assert (
        prediction.expected_action
        == ConversationAction.UPDATE_SEARCH
    )
    assert (
        prediction.predicted_action
        == ConversationAction.UPDATE_SEARCH
    )

    assert prediction.correct is True
    assert prediction.success is True
    assert prediction.error is None

    assert prediction.model == "groq/openai/gpt-oss-20b"
    assert prediction.llm_attempts == 1

    assert prediction.prompt_tokens == 100
    assert prediction.completion_tokens == 20
    assert prediction.total_tokens == 120
    assert prediction.estimated_cost_usd == 0.0000135

    assert prediction.latency_ms >= 0


def test_run_router_eval_case_records_failed_attempts(
    monkeypatch,
):
    async def fake_route_conversation_async(
        *,
        router_input,
        trace,
        llm_profile_name,
    ):
        trace.add_llm_call(
            LLMCallTrace(
                step="conversation_routing",
                model="groq/openai/gpt-oss-20b",
                prompt_tokens=100,
                completion_tokens=0,
                total_tokens=100,
                estimated_cost_usd=0.0000075,
                success=False,
                error="rate_limited",
            )
        )

        trace.add_llm_call(
            LLMCallTrace(
                step="conversation_routing",
                model="groq/openai/gpt-oss-20b",
                prompt_tokens=100,
                completion_tokens=0,
                total_tokens=100,
                estimated_cost_usd=0.0000075,
                success=False,
                error="timeout",
            )
        )

        raise ConversationRoutingError(
            code="timeout",
        )

    monkeypatch.setattr(
        "evaluation.tasks.conversation_router.runner."
        "route_conversation_async",
        fake_route_conversation_async,
    )

    prediction = asyncio.run(
        run_router_eval_case(
            case=_make_case(),
            profile="groq_gpt_oss_20b",
            repeat_number=1,
        )
    )

    assert prediction.success is False
    assert prediction.correct is False

    assert prediction.predicted_action is None
    assert prediction.predicted_reason is None

    assert prediction.error == "timeout"

    assert prediction.llm_attempts == 2

    assert prediction.prompt_tokens == 200
    assert prediction.completion_tokens == 0
    assert prediction.total_tokens == 200
    assert prediction.estimated_cost_usd == 0.000015
    
    


import asyncio
import json

from app.schemas.conversation_route import ConversationAction
from evaluation.tasks.conversation_router.runner import (
    RouterEvalPrediction,
    run_predictions,
)


def test_run_predictions_writes_all_profile_case_combinations(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    rows = [
        {
            "id": "router-001",
            "user_message": "Find a hotel",
            "has_current_search": False,
            "has_shown_results": False,
            "expected_action": "start_search",
            "category": "first_search",
            "critical_if_wrong": False,
            "notes": "Test case",
        },
        {
            "id": "router-002",
            "user_message": "Thanks",
            "has_current_search": True,
            "has_shown_results": False,
            "expected_action": "general_chat",
            "category": "thanks",
            "critical_if_wrong": True,
            "notes": "Test case",
        },
    ]

    with dataset_path.open(
        "w",
        encoding="utf-8",
    ) as dataset_file:
        for row in rows:
            dataset_file.write(
                json.dumps(row) + "\n"
            )

    calls: list[
        tuple[str, str, int]
    ] = []

    async def fake_run_router_eval_case(
        *,
        case,
        profile,
        repeat_number,
    ):
        calls.append(
            (
                case.id,
                profile,
                repeat_number,
            )
        )

        return RouterEvalPrediction(
            case_id=case.id,
            user_message=case.user_message,
            category=case.category,
            profile=profile,
            repeat_number=repeat_number,
            expected_action=case.expected_action,
            predicted_action=case.expected_action,
            predicted_reason="Test prediction",
            correct=True,
            success=True,
            latency_ms=100.0,
            model=f"model-for-{profile}",
            llm_attempts=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.001,
            error=None,
        )

    monkeypatch.setattr(
        "evaluation.tasks.conversation_router.runner."
        "run_router_eval_case",
        fake_run_router_eval_case,
    )

    completed = asyncio.run(
        run_predictions(
            dataset_path=dataset_path,
            output_path=output_path,
            profiles=[
                "gemini_default",
                "groq_gpt_oss_20b",
            ],
            repeats=2,
        )
    )

    assert completed == 8
    assert len(calls) == 8

    saved_rows = []

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as predictions_file:
        for line in predictions_file:
            saved_rows.append(
                json.loads(line)
            )

    assert len(saved_rows) == 8

    combinations = {
        (
            row["case_id"],
            row["profile"],
            row["repeat_number"],
        )
        for row in saved_rows
    }

    assert len(combinations) == 8

    assert {
        row["predicted_action"]
        for row in saved_rows
    } == {
        ConversationAction.START_SEARCH.value,
        ConversationAction.GENERAL_CHAT.value,
    }


def test_run_predictions_does_not_overwrite_existing_run(
    tmp_path,
):
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    dataset_path.write_text(
        "",
        encoding="utf-8",
    )

    output_path.write_text(
        "existing results",
        encoding="utf-8",
    )

    try:
        asyncio.run(
            run_predictions(
                dataset_path=dataset_path,
                output_path=output_path,
                profiles=["gemini_default"],
            )
        )

    except FileExistsError:
        pass

    else:
        raise AssertionError(
            "Existing predictions should not be overwritten"
        )
        
        
def test_calculate_profile_metrics():
    rows = [
        RouterEvalPrediction(
            case_id="router-001",
            user_message="Find a hotel",
            category="first_search",
            profile="test_profile",
            repeat_number=1,
            expected_action="start_search",
            predicted_action="start_search",
            predicted_reason="Correct",
            correct=True,
            success=True,
            latency_ms=100.0,
            model="test-model",
            llm_attempts=1,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            estimated_cost_usd=0.001,
        ),
        RouterEvalPrediction(
            case_id="router-002",
            user_message="Hello",
            category="greeting",
            profile="test_profile",
            repeat_number=1,
            expected_action="general_chat",
            predicted_action="start_search",
            predicted_reason="Wrong",
            correct=False,
            success=True,
            latency_ms=300.0,
            model="test-model",
            llm_attempts=1,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            estimated_cost_usd=0.003,
        ),
    ]

    metrics = calculate_profile_metrics(rows)

    assert metrics["runs"] == 2
    assert metrics["successful_runs"] == 2
    assert metrics["accuracy"] == 0.5

    assert metrics["latency_ms"]["mean"] == 200.0
    assert metrics["latency_ms"]["median"] == 200.0

    assert metrics["cost_usd"]["total"] == 0.004
    assert metrics["cost_usd"]["mean_per_call"] == 0.002
    assert (
        metrics["cost_usd"]["estimated_per_1000_calls"]
        == 2.0
    )

    assert (
        metrics["confusion_matrix"]["general_chat"]
        ["start_search"]
        == 1
    )


def test_evaluate_predictions_groups_profiles():
    predictions = [
        RouterEvalPrediction(
            case_id="router-001",
            user_message="Find a hotel",
            category="first_search",
            profile="gemini_default",
            repeat_number=1,
            expected_action="start_search",
            predicted_action="start_search",
            predicted_reason="Correct",
            correct=True,
            success=True,
            latency_ms=100.0,
            model="gemini-test",
            llm_attempts=1,
        ),
        RouterEvalPrediction(
            case_id="router-001",
            user_message="Find a hotel",
            category="first_search",
            profile="groq_gpt_oss_20b",
            repeat_number=1,
            expected_action="start_search",
            predicted_action="general_chat",
            predicted_reason="Wrong",
            correct=False,
            success=True,
            latency_ms=50.0,
            model="groq-test",
            llm_attempts=1,
        ),
    ]

    result = evaluate_predictions(predictions)

    assert result["total_prediction_rows"] == 2

    assert set(result["profiles"]) == {
        "gemini_default",
        "groq_gpt_oss_20b",
    }

    assert (
        result["profiles"]["gemini_default"]["accuracy"]
        == 1.0
    )

    assert (
        result["profiles"]["groq_gpt_oss_20b"]["accuracy"]
        == 0.0
    )

    assert result["failures_count"] == 1
    assert len(result["failures"]) == 1