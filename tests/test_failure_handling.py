"""
Fault-injection tests for graceful failure handling (P0-1).

Failure policy under test:
- fatal stage failures (router, intent extraction, intent update, factual
  extraction, retrieval) -> typed user-safe failure outcome, previous
  SearchRequest / ShownResultSet untouched;
- recoverable failures (textual evidence fallback) -> UNCERTAIN, the
  search or listing question still completes;
- internal programming errors -> propagate, never relabelled as a
  provider failure;
- only rate limits / provider unavailability are retried; timeouts,
  auth errors and invalid responses are not.

Every provider is faked at its SDK boundary (ADK Runner, google-genai
client, retriever) - no network.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import httpx
import litellm
import pytest
from google.genai import errors as genai_errors

from app.config import settings
from app.llm import call_policy
from app.llm.call_policy import LlmCallError, call_llm_stage, classify_provider_exception
from app.llm.model_factory import LlmModelConfigurationError
from app.logic import (
    constraint_evidence_resolution,
    conversation_flow,
    conversation_router,
    factual_question_extraction,
    intent_router,
    intent_update,
)
from app.logic.conversation_response_adapter import build_conversation_response_input
from app.logic.conversation_response_llm import generate_conversation_response_with_llm
from app.logic.result_candidate_selection import CandidateSelectionResult
from app.logic.result_query_matching import match_result_query_against_candidates
from app.observability.trace import RequestTrace
from app.retrieval.errors import RetrievalError
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_response import ConversationFailureOutcome
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.result_query import ResultQuery
from app.schemas.shown_result_set import ShownResult, ShownResultSet


def _genai_error(code: int) -> genai_errors.APIError:
    error_class = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return error_class(code, {"error": {"code": code, "message": "provider says no"}})


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(settings, "LLM_STAGE_RETRY_BASE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(settings, "LLM_STAGE_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(settings, "LLM_STAGE_TIMEOUT_SECONDS", 5.0)


# --- error classification ---------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (TimeoutError(), "timeout"),
        (_genai_error(429), "rate_limited"),
        (_genai_error(401), "authentication_error"),
        (_genai_error(403), "authentication_error"),
        (_genai_error(503), "provider_unavailable"),
        (_genai_error(400), "provider_error"),
        (litellm.RateLimitError("rl", "groq", "m"), "rate_limited"),
        (litellm.AuthenticationError("auth", "groq", "m"), "authentication_error"),
        (litellm.Timeout("t", "m", "groq"), "timeout"),
        (litellm.ServiceUnavailableError("down", "groq", "m"), "provider_unavailable"),
        (httpx.ConnectError("refused"), "provider_unavailable"),
        (LlmModelConfigurationError("missing key"), "configuration_error"),
    ],
)
def test_provider_exceptions_are_classified(exc, code):
    assert classify_provider_exception(exc) == code


@pytest.mark.parametrize("exc", [TypeError("bug"), KeyError("x"), ValueError("bug"), RuntimeError("bug")])
def test_programming_errors_are_not_classified_as_provider_failures(exc):
    assert classify_provider_exception(exc) is None


# --- call_llm_stage: retry / timeout policy --------------------------------------


def _failing_then(results: list):
    """Attempt function that raises/returns the queued items in order."""
    calls = {"count": 0}

    async def attempt():
        item = results[calls["count"]]
        calls["count"] += 1
        if isinstance(item, BaseException):
            raise item
        return item

    return attempt, calls


@pytest.mark.parametrize("retryable", [_genai_error(429), _genai_error(503), httpx.ConnectError("x")])
async def test_retryable_failure_is_retried_then_succeeds(retryable):
    trace = RequestTrace()
    attempt, calls = _failing_then([retryable, "ok"])

    result = await call_llm_stage(attempt, stage="intent_extraction", model="m", trace=trace)

    assert result == "ok"
    assert calls["count"] == 2
    assert [(c.success, c.step) for c in trace.llm_calls] == [(False, "intent_extraction")]


async def test_rate_limit_is_retried_up_to_max_attempts_then_fails():
    trace = RequestTrace()
    attempt, calls = _failing_then([_genai_error(429)] * 3)

    with pytest.raises(LlmCallError) as exc_info:
        await call_llm_stage(attempt, stage="intent_extraction", model="m", trace=trace)

    assert exc_info.value.code == "rate_limited"
    assert exc_info.value.retryable is True
    assert calls["count"] == settings.LLM_STAGE_MAX_ATTEMPTS
    assert len(trace.llm_calls) == settings.LLM_STAGE_MAX_ATTEMPTS


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (_genai_error(401), "authentication_error"),
        (_genai_error(400), "provider_error"),
        (LlmCallError("invalid_response", stage="s"), "invalid_response"),
        (LlmModelConfigurationError("missing key"), "configuration_error"),
    ],
)
async def test_non_retryable_failure_is_attempted_once(exc, code):
    attempt, calls = _failing_then([exc, "never reached"])

    with pytest.raises(LlmCallError) as exc_info:
        await call_llm_stage(attempt, stage="s", model="m", trace=None)

    assert exc_info.value.code == code
    assert calls["count"] == 1


async def test_timeout_is_not_retried(monkeypatch):
    monkeypatch.setattr(settings, "LLM_STAGE_TIMEOUT_SECONDS", 0.01)
    calls = {"count": 0}

    async def slow_attempt():
        calls["count"] += 1
        await asyncio.sleep(1)

    with pytest.raises(LlmCallError) as exc_info:
        await call_llm_stage(slow_attempt, stage="s", model="m", trace=None)

    assert exc_info.value.code == "timeout"
    assert calls["count"] == 1


async def test_programming_error_propagates_unchanged_and_is_not_retried():
    attempt, calls = _failing_then([TypeError("bug in our code"), "never reached"])

    with pytest.raises(TypeError, match="bug in our code"):
        await call_llm_stage(attempt, stage="s", model="m", trace=None)

    assert calls["count"] == 1


async def test_failure_telemetry_carries_only_the_error_code():
    trace = RequestTrace()
    secret_payload = "api_key=sk-secret-123 user said: my passport is 1234"
    attempt, _ = _failing_then([genai_errors.ClientError(401, {"error": {"message": secret_payload}})])

    with pytest.raises(LlmCallError):
        await call_llm_stage(attempt, stage="s", model="m", trace=trace)

    recorded = trace.summary()["llm"]["calls"]
    assert recorded[0]["error"] == "authentication_error"
    assert recorded[0]["success"] is False
    assert "secret" not in repr(trace.summary())
    assert "passport" not in repr(trace.summary())


def test_retry_delay_backs_off_exponentially(monkeypatch):
    monkeypatch.setattr(settings, "LLM_STAGE_RETRY_BASE_DELAY_SECONDS", 0.5)

    assert [call_policy._retry_delay_seconds(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]


# --- ADK-based stages (intent extraction, intent update, factual extraction) -------


def _text_event(text: str):
    return SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))


def _fake_runner(behaviours: list):
    """
    Replaces google.adk Runner: each run_async call consumes one behaviour
    - an exception to raise, or the response text to stream back.
    """
    calls = {"count": 0}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_async(self, **kwargs):
            behaviour = behaviours[calls["count"]]
            calls["count"] += 1

            async def events():
                if isinstance(behaviour, BaseException):
                    raise behaviour
                yield _text_event(behaviour)

            return events()

    return FakeRunner, calls


ADK_STAGES = {
    "intent_extraction": (
        intent_router,
        lambda: intent_router.route_intent_adk_async("Baku April 10-15", trace=TRACE_HOLDER["trace"]),
        '{"city": "Baku"}',
    ),
    "intent_update": (
        intent_update,
        lambda: intent_update.route_intent_update_patch_async(
            SearchRequest(city="Baku"), "with parking", trace=TRACE_HOLDER["trace"]
        ),
        "{}",
    ),
    "factual_extraction": (
        factual_question_extraction,
        lambda: factual_question_extraction.extract_factual_constraints_async(
            "does it have parking?", trace=TRACE_HOLDER["trace"]
        ),
        '{"constraints": []}',
    ),
}
TRACE_HOLDER: dict = {}


@pytest.fixture
def stage_trace():
    TRACE_HOLDER["trace"] = RequestTrace()
    yield TRACE_HOLDER["trace"]
    TRACE_HOLDER.clear()


@pytest.mark.parametrize("stage", ADK_STAGES)
async def test_adk_stage_retries_rate_limit_then_succeeds(monkeypatch, stage_trace, stage):
    module, call, valid_response = ADK_STAGES[stage]
    runner, calls = _fake_runner([_genai_error(429), valid_response])
    monkeypatch.setattr(module, "Runner", runner)

    await call()

    assert calls["count"] == 2
    assert [c.success for c in stage_trace.llm_calls] == [False, True]


@pytest.mark.parametrize("stage", ADK_STAGES)
@pytest.mark.parametrize(
    ("behaviour", "code"),
    [
        (_genai_error(401), "authentication_error"),
        ("this is not json", "invalid_response"),
        ("", "empty_response"),
    ],
)
async def test_adk_stage_non_retryable_failure_raises_typed_error_once(
    monkeypatch, stage_trace, stage, behaviour, code
):
    module, call, _ = ADK_STAGES[stage]
    runner, calls = _fake_runner([behaviour, "never reached"])
    monkeypatch.setattr(module, "Runner", runner)

    with pytest.raises(LlmCallError) as exc_info:
        await call()

    assert exc_info.value.code == code
    assert calls["count"] == 1
    assert [(c.success, c.error) for c in stage_trace.llm_calls] == [(False, code)]


@pytest.mark.parametrize("stage", ADK_STAGES)
async def test_adk_stage_timeout(monkeypatch, stage_trace, stage):
    monkeypatch.setattr(settings, "LLM_STAGE_TIMEOUT_SECONDS", 0.01)
    module, call, _ = ADK_STAGES[stage]

    class HangingRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_async(self, **kwargs):
            async def events():
                await asyncio.sleep(1)
                yield _text_event("{}")

            return events()

    monkeypatch.setattr(module, "Runner", HangingRunner)

    with pytest.raises(LlmCallError) as exc_info:
        await call()

    assert exc_info.value.code == "timeout"


async def test_intent_update_no_longer_records_success_before_the_call(monkeypatch, stage_trace):
    runner, _ = _fake_runner([_genai_error(401)])
    monkeypatch.setattr(intent_update, "Runner", runner)

    with pytest.raises(LlmCallError):
        await ADK_STAGES["intent_update"][1]()

    assert [c.success for c in stage_trace.llm_calls] == [False]


# --- textual evidence fallback (recoverable) -----------------------------------


def _constraint(field: Field, text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _textual_constraint(text: str = "quiet street") -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


def _fake_genai_client(monkeypatch, behaviours: list):
    calls = {"count": 0}

    class Models:
        def generate_content(self, **kwargs):
            behaviour = behaviours[min(calls["count"], len(behaviours) - 1)]
            calls["count"] += 1
            raise behaviour

    monkeypatch.setattr(
        constraint_evidence_resolution,
        "_gemini_client",
        lambda: SimpleNamespace(models=Models()),
    )
    return calls


def _listing(listing_id: str) -> ListingRaw:
    return ListingRaw(id=listing_id, name=listing_id, city="Baku", description="A cozy flat.")


async def test_textual_fallback_provider_failure_degrades_to_uncertain(monkeypatch):
    calls = _fake_genai_client(monkeypatch, [_genai_error(503)])
    trace = RequestTrace()

    results = await constraint_evidence_resolution.resolve_listing_constraints_with_fallback(
        listing=_listing("L1"),
        constraints=[_textual_constraint()],
        structured_matches_by_field={},
        policy=FallbackPolicy(enabled=True, must_only=False),
        trace=trace,
    )

    assert [r.decision for r in results] == ["UNCERTAIN"]
    assert "unavailable" in results[0].reason
    assert calls["count"] == settings.LLM_STAGE_MAX_ATTEMPTS
    assert all(c.step == "constraint_textual_fallback" and not c.success for c in trace.llm_calls)


async def test_textual_fallback_auth_failure_is_not_retried(monkeypatch):
    calls = _fake_genai_client(monkeypatch, [_genai_error(401)])

    results = await constraint_evidence_resolution.resolve_listing_constraints_with_fallback(
        listing=_listing("L1"),
        constraints=[_textual_constraint()],
        structured_matches_by_field={},
        policy=FallbackPolicy(enabled=True, must_only=False),
    )

    assert [r.decision for r in results] == ["UNCERTAIN"]
    assert calls["count"] == 1


async def test_textual_fallback_programming_error_is_not_hidden(monkeypatch):
    _fake_genai_client(monkeypatch, [TypeError("bug in request building")])

    with pytest.raises(TypeError):
        await constraint_evidence_resolution.resolve_listing_constraints_with_fallback(
            listing=_listing("L1"),
            constraints=[_textual_constraint()],
            structured_matches_by_field={},
            policy=FallbackPolicy(enabled=True, must_only=False),
        )


async def test_listing_question_fallback_failure_keeps_uncertain_verdict(monkeypatch):
    _fake_genai_client(monkeypatch, [_genai_error(429)])
    shown = ShownResult(result_id="H1", listing=_listing("H1"))

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=CandidateSelectionResult(candidates=[shown], status="ok"),
    )

    assert outcome.status == "ok"
    assert outcome.results[0].constraint_results[0].value == Ternary.UNCERTAIN


# --- conversation flow: typed failure outcome + state preservation ---------------


PREVIOUS_STATE = SearchRequest(city="Baku", check_in=date(2026, 4, 10), check_out=date(2026, 4, 15))
PREVIOUS_SHOWN = ShownResultSet(
    result_set_id="rs-previous",
    items=[ShownResult(result_id="H1", listing=_listing("H1"))],
)


def _route(monkeypatch, action: ConversationAction, *, scope=ResultScope.NOT_APPLICABLE, target=""):
    async def fake_route(**kwargs):
        return ConversationActionDecision(
            action=action,
            reason="test",
            decision_status=ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.NONE,
            result_scope=scope,
            target_result_id=target,
        )

    monkeypatch.setattr(conversation_flow, "route_conversation_async", fake_route)


def _raise(exc):
    async def fake(*args, **kwargs):
        raise exc

    return fake


def _assert_failure_turn(result: dict, *, stage: str, code: str) -> None:
    assert result["need_clarification"] is False
    assert result["response_type"] == "processing_unavailable"
    assert result["failure"] == {"stage": stage, "code": code}
    assert "shown_result_set" not in result
    assert SearchRequest.model_validate(result["state"]) == PREVIOUS_STATE
    assert result["telemetry"]["trace_id"]

    response_input = build_conversation_response_input(user_message="msg", result=result)
    assert isinstance(response_input.outcome, ConversationFailureOutcome)
    assert response_input.outcome.user_safe_message == result["answer"]


async def test_start_search_intent_failure_returns_typed_failure(monkeypatch):
    _route(monkeypatch, ConversationAction.START_SEARCH)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", _raise(LlmCallError("rate_limited", stage="x", retryable=True)))

    result = await conversation_flow.handle_user_message(
        "new search in Tbilisi", previous_state=PREVIOUS_STATE, shown_result_set=PREVIOUS_SHOWN
    )

    _assert_failure_turn(result, stage="search_intent_extraction", code="rate_limited")


async def test_update_failure_keeps_previous_state(monkeypatch):
    _route(monkeypatch, ConversationAction.UPDATE_SEARCH)
    monkeypatch.setattr(intent_update, "route_intent_update_patch_async", _raise(LlmCallError("timeout", stage="x")))

    result = await conversation_flow.handle_user_message(
        "add parking", previous_state=PREVIOUS_STATE, shown_result_set=PREVIOUS_SHOWN
    )

    _assert_failure_turn(result, stage="search_state_update", code="timeout")


async def test_listing_question_extraction_failure(monkeypatch):
    _route(monkeypatch, ConversationAction.LISTING_QUESTION, scope=ResultScope.SPECIFIC_RESULT, target="H1")
    monkeypatch.setattr(
        conversation_flow,
        "extract_factual_constraints_async",
        _raise(LlmCallError("authentication_error", stage="x")),
    )

    result = await conversation_flow.handle_user_message(
        "does it have parking?", previous_state=PREVIOUS_STATE, shown_result_set=PREVIOUS_SHOWN
    )

    _assert_failure_turn(result, stage="listing_question_factual_extraction", code="authentication_error")


async def test_retrieval_failure_keeps_previous_state_not_the_new_one(monkeypatch):
    _route(monkeypatch, ConversationAction.UPDATE_SEARCH)

    async def fake_update(previous_state, user_message, **kwargs):
        return previous_state.model_copy(update={"city": "Tbilisi"})

    monkeypatch.setattr(conversation_flow, "update_search_state_async", fake_update)
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        _raise(RetrievalError("provider_unavailable")),
    )

    result = await conversation_flow.handle_user_message(
        "Tbilisi instead", previous_state=PREVIOUS_STATE, shown_result_set=PREVIOUS_SHOWN
    )

    _assert_failure_turn(result, stage="retrieval", code="provider_unavailable")
    assert "couldn't search" in result["answer"]


async def test_programming_error_in_a_stage_is_not_converted_to_failure_outcome(monkeypatch):
    _route(monkeypatch, ConversationAction.START_SEARCH)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", _raise(KeyError("bug")))

    with pytest.raises(KeyError):
        await conversation_flow.handle_user_message("Baku April 10-15")


async def test_search_completes_when_every_textual_fallback_fails(monkeypatch):
    """Recoverable: provider down for the fallback only - results still come back."""
    _route(monkeypatch, ConversationAction.START_SEARCH)
    _fake_genai_client(monkeypatch, [_genai_error(503)])

    async def fake_intent(user_text, **kwargs):
        from app.agents.intent_router_agent import IntentRoute

        return IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            constraints=[
                UserConstraint(
                    raw_text="quiet street",
                    normalized_text="quiet street",
                    priority=ConstraintPriority.MUST,
                    category=ConstraintCategory.OTHER,
                    mapping_status=ConstraintMappingStatus.UNRESOLVED,
                    mapped_fields=[],
                    evidence_strategy=EvidenceStrategy.TEXTUAL,
                )
            ],
        )

    monkeypatch.setattr(intent_router, "route_intent_adk_async", fake_intent)

    result = await conversation_flow.handle_user_message(
        "quiet apartment in Baku April 10-15",
        fallback_policy=FallbackPolicy(enabled=True, top_k=2),
    )

    assert result["status"] == "results"
    assert "shown_result_set" in result
    decisions = {
        r["decision"]
        for item in result["results"]
        for r in item.get("constraint_resolution_results", [])
    }
    assert decisions == {"UNCERTAIN"}
    failed = [c for c in result["telemetry"]["llm"]["calls"] if c["step"] == "constraint_textual_fallback"]
    assert failed and not any(c["success"] for c in failed)


# --- router --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (litellm.ServiceUnavailableError("down", "groq", "m"), "provider_unavailable"),
        (litellm.Timeout("t", "m", "groq"), "timeout"),
        (httpx.ConnectError("refused"), "provider_unavailable"),
    ],
)
async def test_router_classifies_provider_errors_it_used_to_reraise(monkeypatch, exc, code):
    runner, _ = _fake_runner([exc] * 5)
    monkeypatch.setattr(conversation_router, "Runner", runner)
    monkeypatch.setattr(conversation_router, "CONVERSATION_ROUTER_RETRY_DELAY_SECONDS", 0)

    result = await conversation_flow.handle_user_message(
        "hello", previous_state=PREVIOUS_STATE, shown_result_set=PREVIOUS_SHOWN
    )

    assert result["response_type"] == "routing_unavailable"
    assert result["parsed_intent"]["router"]["error_code"] == code
    assert SearchRequest.model_validate(result["state"]) == PREVIOUS_STATE


async def test_router_programming_error_still_propagates(monkeypatch):
    runner, _ = _fake_runner([TypeError("bug")])
    monkeypatch.setattr(conversation_router, "Runner", runner)

    with pytest.raises(TypeError):
        await conversation_flow.handle_user_message("hello")


# --- response generation ----------------------------------------------------------


async def test_failure_outcome_is_rendered_deterministically_without_llm(monkeypatch):
    def no_llm(*args, **kwargs):
        raise AssertionError("response LLM must not be called for a failure outcome")

    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_llm)
    _route(monkeypatch, ConversationAction.UPDATE_SEARCH)
    monkeypatch.setattr(intent_update, "route_intent_update_patch_async", _raise(LlmCallError("timeout", stage="x")))

    result = await conversation_flow.handle_user_message("add parking", previous_state=PREVIOUS_STATE)
    response_input = build_conversation_response_input(user_message="add parking", result=result)

    generated = await generate_conversation_response_with_llm(response_input)

    assert generated.source == "deterministic"
    assert generated.text == result["answer"]


async def test_response_generation_failure_falls_back_with_code_only(monkeypatch):
    def failing_model(*args, **kwargs):
        raise _genai_error(503)

    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", failing_model)
    _route(monkeypatch, ConversationAction.GENERAL_CHAT)
    trace = RequestTrace()

    result = await conversation_flow.handle_user_message("hello", trace=trace)
    response_input = build_conversation_response_input(user_message="hello", result=result)
    generated = await generate_conversation_response_with_llm(response_input, trace=trace)

    assert generated.source == "deterministic_fallback"
    response_calls = [c for c in trace.llm_calls if c.step == "conversation_response_generation"]
    assert [(c.success, c.error) for c in response_calls] == [(False, "provider_unavailable")]


# --- retrieval (Apify) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (lambda: HTTPError("https://api.apify.com/?token=SECRET", 401, "Unauthorized", {}, None), "authentication_error"),
        (lambda: HTTPError("https://api.apify.com/?token=SECRET", 429, "Too Many", {}, None), "rate_limited"),
        (lambda: HTTPError("https://api.apify.com/?token=SECRET", 502, "Bad Gateway", {}, None), "provider_unavailable"),
        (lambda: URLError("connection refused"), "provider_unavailable"),
        (lambda: TimeoutError("read timed out"), "timeout"),
    ],
)
async def test_apify_failures_become_retrieval_errors_without_secrets(monkeypatch, raised, code):
    from app.retrieval import apify

    monkeypatch.setenv("APIFY_TOKEN", "SECRET")

    def fail(*args, **kwargs):
        raise raised()

    monkeypatch.setattr(apify, "_post_json_sync", fail)
    trace = RequestTrace()

    with pytest.raises(RetrievalError) as exc_info:
        await apify.ApifyRetriever().get_candidates(
            SearchRequest(city="Baku", check_in=date(2026, 4, 10), check_out=date(2026, 4, 15)),
            max_items=5,
            trace=trace,
        )

    assert exc_info.value.code == code
    assert "SECRET" not in str(exc_info.value)
    external = trace.summary()["external"]["calls"]
    assert external and all(c["success"] is False and c["error"] == code for c in external)
    assert "SECRET" not in repr(trace.summary())


async def test_apify_missing_token_is_a_configuration_failure(monkeypatch):
    from app.retrieval import apify

    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    with pytest.raises(RetrievalError) as exc_info:
        await apify.ApifyRetriever().get_candidates(
            SearchRequest(city="Baku", check_in=date(2026, 4, 10), check_out=date(2026, 4, 15)),
            max_items=5,
        )

    assert exc_info.value.code == "configuration_error"
