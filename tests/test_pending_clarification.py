"""
PendingClarification V1: continuity after an AMBIGUOUS_ACTION abstention.

Whether a reply answers, cancels or replaces a pending clarification is a
semantic judgement of the routing layer (pending_resolution), made with
PendingClarification as input. These tests fake that judgement - the
fake router returns a scripted semantic outcome per message - and check
what the application does with it: state transitions, recovery of the
original request, result-set validity, persistence.

Limitation: a passing suite proves the deterministic contract only. It
says nothing about whether the production model (Gemini) actually
returns the right pending_resolution for real paraphrases - that needs a
live router eval with pending-context cases.

Hermetic: the first search runs on real fixture retrieval; later searches
are captured by a fake orchestrator so a parking requirement never needs
the textual LLM fallback.
"""

from __future__ import annotations

import inspect
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agents import conversation_router_agent as router_agent_module
from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import (
    clarification_continuation,
    conversation_flow,
    conversation_router,
    intent_router,
    intent_update,
)
from app.logic.clarification_continuation import (
    PendingTransition,
    build_pending_clarification,
    resolve_pending_transition,
)
from app.logic.conversation_flow import _REQUEST_NOT_COMPLETED_MESSAGE
from app.logic.conversation_router import ConversationRoutingError
from app.logic.date_normalization import InvalidStayDatesError, StayDatesIssue
from app.logic.factual_question_extraction import _build_factual_question_message
from app.logic.result_query_matching import ResultQueryMatchOutcome
from app.retrieval.errors import RetrievalError
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    PendingAwareConversationActionDecision,
    PendingClarification,
    PendingResolution,
    ResultScope,
    RouterInput,
)
from app.schemas.fields import Field
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.result_query import ResultQuery
from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
from app.schemas.search_response import NormalizedSearchResponse, SearchStatus
from app.schemas.session import ConversationSession
from app.schemas.shown_result_set import ShownResultSet
from app.schemas.turn import TurnConfig, TurnResult
from app.services.conversation_service import process_turn

CONFIG = TurnConfig(persist_telemetry=False, include_debug=True)
TODAY = date(2026, 3, 1)

ORIGINAL = "Only the ones with parking?"

RESOLVED = ConversationDecisionStatus.RESOLVED
NEEDS = ConversationDecisionStatus.NEEDS_CLARIFICATION


def _decision(
    action,
    *,
    status=RESOLVED,
    reason=ClarificationReason.NONE,
    scope=ResultScope.NOT_APPLICABLE,
    target="",
):
    return ConversationActionDecision(
        action=action,
        reason="test",
        decision_status=status,
        clarification_reason=reason,
        result_scope=scope,
        target_result_id=target,
    )


def _pending_aware(
    resolution,
    action,
    *,
    status=RESOLVED,
    reason=ClarificationReason.NONE,
    scope=ResultScope.NOT_APPLICABLE,
):
    return PendingAwareConversationActionDecision(
        **_decision(action, status=status, reason=reason, scope=scope).model_dump(),
        pending_resolution=resolution,
    )


AMBIGUOUS_ACTION = _decision(
    ConversationAction.LISTING_QUESTION,
    status=NEEDS,
    reason=ClarificationReason.AMBIGUOUS_ACTION,
    scope=ResultScope.CURRENT_RESULTS,
)

CONTINUE_UPDATE = _pending_aware(PendingResolution.CONTINUE_PENDING, ConversationAction.UPDATE_SEARCH)
CONTINUE_LISTING = _pending_aware(
    PendingResolution.CONTINUE_PENDING,
    ConversationAction.LISTING_QUESTION,
    scope=ResultScope.CURRENT_RESULTS,
)
CANCEL = _pending_aware(PendingResolution.CANCEL_PENDING, ConversationAction.GENERAL_CHAT)
UNRESOLVED = _pending_aware(
    PendingResolution.UNRESOLVED,
    ConversationAction.UPDATE_SEARCH,
    status=NEEDS,
    reason=ClarificationReason.AMBIGUOUS_ACTION,
)


def _constraint(text, field, priority=ConstraintPriority.MUST) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _parking(priority=ConstraintPriority.MUST) -> UserConstraint:
    return _constraint("parking", Field.PARKING, priority)


def _async_return(value):
    async def fake(*args, **kwargs):
        return value

    return fake


class Fakes:
    """
    Scriptable LLM boundaries that record every call. The router fake is
    the semantic stand-in: route() scripts its judgement per message.
    """

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.router_inputs: list[RouterInput] = []
        # (state, message, confirmed_search_change, answered_request)
        self.patch_calls: list[tuple[SearchRequest, str, bool, str | None]] = []
        # (message, original_question)
        self.factual_calls: list[tuple[str, str | None]] = []
        self.facts: dict[str, ResultQuery] = {}
        self.matched_queries: list[ResultQuery] = []
        self.candidate_selections: list[dict] = []
        self.searched: list[SearchRequest] = []
        self.patches: dict[str, SearchIntentPatch] = {}
        self.routes: dict[str, object] = {}

        def no_model(*args, **kwargs):
            raise LlmCallError("configuration_error", stage="conversation_response_generation")

        monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_model)

        async def fake_route(*, router_input, **kwargs):
            self.router_inputs.append(router_input)
            outcome = self.routes[router_input.user_message]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(conversation_flow, "route_conversation_async", fake_route)

        async def fake_patch(
            previous_state,
            user_message,
            *,
            trace=None,
            confirmed_search_change=False,
            answered_request=None,
        ):
            self.patch_calls.append(
                (previous_state, user_message, confirmed_search_change, answered_request)
            )
            outcome = self.patches.get(user_message, SearchIntentPatch())
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(intent_update, "route_intent_update_patch_async", fake_patch)

        async def fake_factual(user_message, *, trace=None, original_question=None):
            # Semantic stand-in for the extractor: scripted per message.
            self.factual_calls.append((user_message, original_question))
            outcome = self.facts.get(user_message, ResultQuery(constraints=[_constraint("WiFi", Field.WIFI)]))
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(conversation_flow, "extract_factual_constraints_async", fake_factual)

        real_select = conversation_flow.select_shown_candidates

        def spy_select(**kwargs):
            self.candidate_selections.append(kwargs)
            return real_select(**kwargs)

        monkeypatch.setattr(conversation_flow, "select_shown_candidates", spy_select)

    def route(self, message: str, outcome) -> None:
        self.routes[message] = outcome

    def capture_searches(self):
        async def fake_orchestrate(state, **kwargs):
            self.searched.append(state)
            return (
                NormalizedSearchResponse(status=SearchStatus.NO_RESULTS, results=[]),
                ShownResultSet(result_set_id="rs-after-update", items=[]),
            )

        self.monkeypatch.setattr(conversation_flow, "orchestrate_search_request", fake_orchestrate)

    def fail_searches(self, exc: BaseException):
        async def failing_orchestrate(state, **kwargs):
            self.searched.append(state)
            raise exc

        self.monkeypatch.setattr(conversation_flow, "orchestrate_search_request", failing_orchestrate)

    def fake_matcher(self):
        """Records the ResultQuery; every verdict is UNCERTAIN (hermetic)."""

        async def matcher(*, result_query, candidate_selection, **kwargs):
            self.matched_queries.append(result_query)
            return ResultQueryMatchOutcome(
                results=[
                    ResultQueryMatch(
                        result_id=candidate.result_id,
                        constraint_results=[
                            ConstraintMatchResult(
                                constraint_id=c.id,
                                raw_text=c.raw_text,
                                field=c.mapped_fields[0] if c.mapped_fields else None,
                                value=Ternary.UNCERTAIN,
                            )
                            for c in result_query.constraints
                        ],
                    )
                    for candidate in candidate_selection.candidates
                ],
                status="ok",
            )

        self.monkeypatch.setattr(conversation_flow, "match_result_query_against_candidates", matcher)

    def reset_calls(self):
        for calls in (
            self.router_inputs,
            self.patch_calls,
            self.factual_calls,
            self.candidate_selections,
            self.searched,
            self.matched_queries,
        ):
            calls.clear()


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    return Fakes(monkeypatch)


async def _turn(session: ConversationSession, message: str, config: TurnConfig = CONFIG) -> TurnResult:
    with clock.use_reference_date(TODAY):
        return await process_turn(session, message, config=config)


async def _searched_session(fakes: Fakes) -> ConversationSession:
    message = "apartment in Baku April 10-15 with WiFi"
    fakes.route(message, _decision(ConversationAction.START_SEARCH))
    fakes.monkeypatch.setattr(
        intent_router,
        "route_intent_adk_async",
        _async_return(
            IntentRoute(
                city="Baku",
                check_in="2026-04-10",
                check_out="2026-04-15",
                property_types=["apartment"],
                constraints=[_constraint("WiFi", Field.WIFI)],
            )
        ),
    )
    turn = await _turn(ConversationSession(), message)
    assert turn.session.shown_result_set.items, "fixture search must show results"
    fakes.reset_calls()
    return turn.session


async def _pending_session(fakes: Fakes) -> ConversationSession:
    session = await _searched_session(fakes)
    fakes.route(ORIGINAL, AMBIGUOUS_ACTION)
    turn = await _turn(session, ORIGINAL)
    fakes.reset_calls()
    return turn.session


def _assert_read_only(before: ConversationSession, after: ConversationSession) -> None:
    assert after.search_request == before.search_request
    assert after.shown_result_set == before.shown_result_set


def _patch_messages(fakes: Fakes) -> list[tuple[str, bool]]:
    return [(message, confirmed) for _, message, confirmed, _ in fakes.patch_calls]


def _patch_contexts(fakes: Fakes) -> list[str | None]:
    return [answered for *_, answered in fakes.patch_calls]


def _parking_priorities(session: ConversationSession) -> list[ConstraintPriority]:
    return [c.priority for c in session.search_request.constraints if c.normalized_text == "parking"]


# --- creation -------------------------------------------------------------------


async def test_ambiguous_action_creates_pending_clarification(fakes):
    session = await _searched_session(fakes)
    fakes.route(ORIGINAL, AMBIGUOUS_ACTION)

    turn = await _turn(session, ORIGINAL)

    assert turn.session.pending_clarification == PendingClarification(
        kind=ClarificationReason.AMBIGUOUS_ACTION,
        original_user_message=ORIGINAL,
        possible_actions=[ConversationAction.UPDATE_SEARCH, ConversationAction.LISTING_QUESTION],
        result_set_id=session.shown_result_set.result_set_id,
    )
    assert turn.outcome_kind == "clarification"
    _assert_read_only(session, turn.session)


async def test_ambiguous_reference_does_not_create_pending(fakes):
    # Without the discourse feature (now on by default, so switched off here) a
    # "which one?" clarification (here from a target that is not shown - an
    # ambiguous_reference abstention is answered for all shown results under
    # safe scope widening) opens nothing. With the feature on it opens an
    # ambiguous_reference question: tests/test_discourse_context.py.
    session = await _searched_session(fakes)
    message = "Does it have parking?"
    fakes.route(
        message,
        _decision(
            ConversationAction.LISTING_QUESTION,
            scope=ResultScope.SPECIFIC_RESULT,
            target="not-shown",
        ),
    )

    turn = await _turn(session, message, config=CONFIG.model_copy(update={"discourse_context": False}))

    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification is None


def test_no_pending_when_only_one_reading_can_run():
    empty = ShownResultSet(result_set_id="rs", items=[])
    for previous_state, shown_result_set in (
        (None, empty),
        (SearchRequest(city="Baku"), None),
        (SearchRequest(city="Baku"), empty),
    ):
        assert (
            build_pending_clarification(
                reason=ClarificationReason.AMBIGUOUS_ACTION,
                user_message=ORIGINAL,
                previous_state=previous_state,
                shown_result_set=shown_result_set,
            )
            is None
        )


# --- persistence ----------------------------------------------------------------


async def test_pending_survives_session_serialization(fakes):
    session = await _pending_session(fakes)

    restored = ConversationSession.model_validate_json(session.model_dump_json())

    assert restored.pending_clarification == session.pending_clarification
    assert restored.pending_clarification.original_user_message == ORIGINAL


def test_stored_session_without_pending_field_still_loads():
    legacy = json.loads(ConversationSession(search_request=SearchRequest(city="Baku")).model_dump_json())
    del legacy["pending_clarification"]

    restored = ConversationSession.model_validate(legacy)

    assert restored.pending_clarification is None
    assert restored.search_request.city == "Baku"


# --- 1. pending context reaches the router ------------------------------------------


async def test_pending_clarification_is_part_of_router_input(fakes):
    session = await _pending_session(fakes)
    reply = "I'd rather change the search criteria."
    fakes.route(reply, UNRESOLVED)

    await _turn(session, reply)

    [router_input] = fakes.router_inputs
    assert router_input.user_message == reply
    assert router_input.pending_clarification == session.pending_clarification
    assert router_input.current_search == session.search_request


def test_router_prompt_carries_pending_context():
    pending = PendingClarification(
        kind=ClarificationReason.AMBIGUOUS_ACTION,
        original_user_message=ORIGINAL,
        possible_actions=[ConversationAction.UPDATE_SEARCH, ConversationAction.LISTING_QUESTION],
        result_set_id="rs-1",
    )
    base = RouterInput(user_message="change the criteria please")

    with_pending = conversation_router._build_router_prompt(
        router_input=base.model_copy(update={"pending_clarification": pending})
    )
    without = conversation_router._build_router_prompt(router_input=base)

    section = with_pending.split("Pending clarification:\n", 1)[1].split("\n\nLatest user message:", 1)[0]
    assert json.loads(section) == pending.model_dump(mode="json")
    assert with_pending.endswith("Latest user message:\nchange the criteria please")
    # Turns without pending state keep the exact previous prompt.
    assert without == (
        "Current search:\nnull\n\nLatest shown result context:\n{}\n\n"
        "Latest user message:\nchange the criteria please"
    )


def test_pending_aware_agent_extends_the_router_contract(monkeypatch):
    agent_mock = Mock(return_value=object())
    monkeypatch.setattr(router_agent_module, "Agent", agent_mock)

    router_agent_module.build_pending_aware_conversation_router_agent(model=object())

    kwargs = agent_mock.call_args.kwargs
    assert kwargs["name"] == "conversation_router"
    assert kwargs["output_schema"] is PendingAwareConversationActionDecision
    assert kwargs["instruction"].startswith(router_agent_module.CONVERSATION_ROUTER_INSTRUCTION)
    assert "PENDING CLARIFICATION" in kwargs["instruction"]
    assert "PENDING CLARIFICATION" not in router_agent_module.CONVERSATION_ROUTER_INSTRUCTION


class _OneShotRunner:
    response_text = ""

    def __init__(self, *, agent, app_name, session_service):
        self.agent = agent

    def run_async(self, **kwargs):
        text = self.response_text

        async def events():
            yield SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                is_final_response=lambda: True,
            )

        return events()


class _FakeSessionService:
    async def create_session(self, **kwargs):
        return None


def _patch_router_runtime(monkeypatch, response: dict) -> dict[str, Mock]:
    runner = type("Runner", (_OneShotRunner,), {"response_text": json.dumps(response)})
    builders = {
        "build_conversation_router_agent": Mock(return_value="plain-agent"),
        "build_pending_aware_conversation_router_agent": Mock(return_value="pending-agent"),
    }
    monkeypatch.setattr(conversation_router, "get_llm_profile", Mock(return_value=SimpleNamespace(model="m")))
    monkeypatch.setattr(conversation_router, "build_adk_model", Mock(return_value=object()))
    monkeypatch.setattr(conversation_router, "InMemorySessionService", _FakeSessionService)
    monkeypatch.setattr(conversation_router, "Runner", runner)
    for name, mock in builders.items():
        monkeypatch.setattr(conversation_router, name, mock)
    return builders


PENDING = PendingClarification(
    kind=ClarificationReason.AMBIGUOUS_ACTION,
    original_user_message=ORIGINAL,
    possible_actions=[ConversationAction.UPDATE_SEARCH, ConversationAction.LISTING_QUESTION],
    result_set_id="rs",
)


async def test_router_uses_pending_aware_output_only_with_pending(monkeypatch):
    builders = _patch_router_runtime(monkeypatch, CONTINUE_UPDATE.model_dump(mode="json"))

    decision = await conversation_router.route_conversation_async(
        router_input=RouterInput(user_message="change it", pending_clarification=PENDING)
    )

    assert isinstance(decision, PendingAwareConversationActionDecision)
    assert decision.pending_resolution == PendingResolution.CONTINUE_PENDING
    builders["build_pending_aware_conversation_router_agent"].assert_called_once()
    builders["build_conversation_router_agent"].assert_not_called()


async def test_router_with_pending_rejects_output_without_pending_resolution(monkeypatch):
    _patch_router_runtime(monkeypatch, _decision(ConversationAction.UPDATE_SEARCH).model_dump(mode="json"))

    with pytest.raises(ConversationRoutingError) as exc:
        await conversation_router.route_conversation_async(
            router_input=RouterInput(user_message="change it", pending_clarification=PENDING)
        )

    assert exc.value.code == "invalid_response"


async def test_router_without_pending_is_unchanged(monkeypatch):
    builders = _patch_router_runtime(monkeypatch, _decision(ConversationAction.GENERAL_CHAT).model_dump(mode="json"))

    decision = await conversation_router.route_conversation_async(router_input=RouterInput(user_message="hi"))

    assert type(decision) is ConversationActionDecision
    builders["build_conversation_router_agent"].assert_called_once()
    builders["build_pending_aware_conversation_router_agent"].assert_not_called()


# --- 3/8. continue the original update ----------------------------------------------


async def test_continued_update_applies_original_requirement(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "I'd rather change the search criteria."
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])

    turn = await _turn(session, reply)

    # Original first (as a confirmed change), then the reply on top of it;
    # the reply adds nothing here, so Intent Update returns an empty patch.
    assert _patch_messages(fakes) == [(ORIGINAL, True), (reply, False)]
    assert fakes.patch_calls[0][0] == session.search_request
    assert _parking_priorities(turn.session) == [ConstraintPriority.MUST]
    assert fakes.searched == [turn.session.search_request]
    assert turn.conversation_action == ConversationAction.UPDATE_SEARCH
    assert turn.session.pending_clarification is None
    assert session.pending_clarification is not None  # input session untouched


async def test_continued_update_with_refinement_keeps_parking_as_preference(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Let's update the search, but parking is only a preference."
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    # What Intent Update semantically extracts from the reply.
    fakes.patches[reply] = SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE)])

    turn = await _turn(session, reply)

    assert _patch_messages(fakes) == [(ORIGINAL, True), (reply, False)]
    # The reply is applied on top of the recovered original requirement.
    state_before_reply = fakes.patch_calls[1][0]
    assert [c.priority for c in state_before_reply.constraints if c.normalized_text == "parking"] == [
        ConstraintPriority.MUST
    ]
    assert _parking_priorities(turn.session) == [ConstraintPriority.NICE]
    assert turn.session.pending_clarification is None


async def test_continued_update_without_change_is_a_no_op(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "yes, change the search"
    fakes.route(reply, CONTINUE_UPDATE)  # both patches empty

    turn = await _turn(session, reply)

    assert fakes.searched == []
    assert turn.outcome_kind == "informational"
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


# --- 4. continue the original listing question --------------------------------------


@pytest.mark.parametrize("router_scope", [ResultScope.CURRENT_RESULTS, ResultScope.NOT_APPLICABLE])
async def test_continued_listing_question_is_read_only(fakes, router_scope):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Actually, just tell me which of the hotels you've already found have parking."
    fakes.route(
        reply,
        _pending_aware(
            PendingResolution.CONTINUE_PENDING,
            ConversationAction.LISTING_QUESTION,
            scope=router_scope,
        ),
    )

    turn = await _turn(session, reply)

    assert fakes.patch_calls == [] and fakes.searched == []
    assert fakes.factual_calls == [(reply, ORIGINAL)]
    [selection] = fakes.candidate_selections
    assert selection["result_scope"] == ResultScope.CURRENT_RESULTS
    assert selection["shown_result_set"] == session.shown_result_set
    assert turn.outcome_kind == "listing_query"
    assert {i.match.result_id for i in turn.inspector.listing_query} == {
        item.result_id for item in session.shown_result_set.items
    }
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


# --- 5. new independent request -----------------------------------------------------


async def test_new_request_does_not_inherit_original_requirement(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Actually, forget that. Find me somewhere in Paris instead."
    # Same action an answer would have - only pending_resolution differs.
    fakes.route(
        reply,
        _pending_aware(PendingResolution.NEW_REQUEST, ConversationAction.UPDATE_SEARCH),
    )
    fakes.patches[reply] = SearchIntentPatch(set_city="Paris")

    turn = await _turn(session, reply)

    assert _patch_messages(fakes) == [(reply, False)]
    assert turn.session.search_request.city == "Paris"
    assert _parking_priorities(turn.session) == []
    assert turn.session.pending_clarification is None


async def test_new_ambiguous_request_replaces_pending(fakes):
    session = await _pending_session(fakes)
    reply = "Only the ones with a pool?"
    fakes.route(
        reply,
        _pending_aware(
            PendingResolution.NEW_REQUEST,
            ConversationAction.LISTING_QUESTION,
            status=NEEDS,
            reason=ClarificationReason.AMBIGUOUS_ACTION,
            scope=ResultScope.CURRENT_RESULTS,
        ),
    )

    turn = await _turn(session, reply)

    assert turn.session.pending_clarification.original_user_message == reply
    _assert_read_only(session, turn.session)


# --- 6. unclear reply ---------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome",
    [
        UNRESOLVED,
        # Inconsistent "answers": never executed, never guessed into an action.
        _pending_aware(
            PendingResolution.CONTINUE_PENDING,
            ConversationAction.UPDATE_SEARCH,
            status=NEEDS,
            reason=ClarificationReason.AMBIGUOUS_ACTION,
        ),
        _pending_aware(PendingResolution.CONTINUE_PENDING, ConversationAction.START_SEARCH),
        _pending_aware(PendingResolution.CONTINUE_PENDING, ConversationAction.GENERAL_CHAT),
    ],
)
async def test_unresolved_reply_keeps_pending_and_changes_nothing(fakes, outcome):
    session = await _pending_session(fakes)
    reply = "I'm not really sure which option is better."
    fakes.route(reply, outcome)

    turn = await _turn(session, reply)

    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification == session.pending_clarification
    assert fakes.patch_calls == [] and fakes.factual_calls == []
    _assert_read_only(session, turn.session)


# --- 7. cancellation ----------------------------------------------------------------


async def test_cancel_clears_pending_without_state_mutation(fakes):
    session = await _pending_session(fakes)
    reply = "Never mind, let's leave everything as it is."
    fakes.route(reply, CANCEL)

    turn = await _turn(session, reply)

    assert fakes.patch_calls == [] and fakes.factual_calls == []
    assert turn.outcome_kind == "informational"
    assert turn.conversation_action == ConversationAction.GENERAL_CHAT
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


# --- 9. stale result set ------------------------------------------------------------


async def test_changed_result_set_cannot_satisfy_pending_listing_question(fakes):
    session = await _pending_session(fakes)
    stale = session.model_copy(
        update={
            "pending_clarification": session.pending_clarification.model_copy(
                update={"result_set_id": "rs-some-older-search"}
            )
        }
    )
    reply = "look through the ones you found"
    fakes.route(reply, CONTINUE_LISTING)

    turn = await _turn(stale, reply)

    assert fakes.factual_calls == [] and fakes.candidate_selections == []
    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification is None
    _assert_read_only(stale, turn.session)


async def test_unavailable_result_set_cannot_satisfy_pending_listing_question(fakes):
    session = await _pending_session(fakes)
    gone = session.model_copy(update={"shown_result_set": None})
    reply = "look through the ones you found"
    fakes.route(reply, CONTINUE_LISTING)

    turn = await _turn(gone, reply)

    assert fakes.factual_calls == []
    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification is None


# --- failure lifecycle -----------------------------------------------------------------


async def test_routing_failure_keeps_pending(fakes):
    """Before any semantic decision: the pending request is kept."""
    session = await _pending_session(fakes)
    reply = "change the search"
    fakes.route(reply, ConversationRoutingError("invalid_response"))

    turn = await _turn(session, reply)

    assert turn.failure.stage == "conversation_routing"
    assert turn.session.pending_clarification == session.pending_clarification
    assert fakes.patch_calls == [] and fakes.factual_calls == []
    _assert_read_only(session, turn.session)


async def test_missing_pending_resolution_is_a_typed_routing_failure(fakes):
    """M1: a plain six-field decision on a pending turn is never executed."""
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "change the search"
    fakes.route(reply, _decision(ConversationAction.UPDATE_SEARCH))

    turn = await _turn(session, reply)

    assert turn.failure.stage == "conversation_routing"
    assert turn.failure.code == "missing_pending_resolution"
    assert fakes.patch_calls == [] and fakes.searched == []
    assert turn.session.pending_clarification == session.pending_clarification
    _assert_read_only(session, turn.session)


@pytest.mark.parametrize("failing_call", ["original", "reply"])
async def test_continued_update_failure_clears_pending_and_asks_for_full_request(fakes, failing_call):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Update the search, but parking is only a preference."
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    failing_message = ORIGINAL if failing_call == "original" else reply
    fakes.patches[failing_message] = LlmCallError("timeout", stage="intent_update")

    turn = await _turn(session, reply)

    assert turn.failure is not None
    assert fakes.searched == []  # no intermediate state reached retrieval
    assert turn.session.pending_clarification is None
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message
    _assert_read_only(session, turn.session)


async def test_continued_update_retrieval_failure_clears_pending(fakes):
    session = await _pending_session(fakes)
    reply = "Update the search, but parking is only a preference."
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    fakes.fail_searches(RetrievalError("provider_unavailable"))

    turn = await _turn(session, reply)

    assert len(fakes.searched) == 1  # the update itself succeeded
    assert turn.failure.stage == "retrieval"
    assert turn.session.pending_clarification is None
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message
    _assert_read_only(session, turn.session)


async def test_retrieval_failure_keeps_retriever_message_and_asks_for_full_request(fakes):
    session = await _pending_session(fakes)
    reply = "change the search"
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    fakes.fail_searches(
        RetrievalError("unsupported_request", user_message="I can't search for that here.")
    )

    turn = await _turn(session, reply)

    assert turn.assistant_message.startswith("I can't search for that here.")
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message


async def test_continued_listing_extraction_failure_clears_pending(fakes):
    session = await _pending_session(fakes)
    reply = "just look through the ones you found"
    fakes.route(reply, CONTINUE_LISTING)
    fakes.facts[reply] = LlmCallError("timeout", stage="factual_question_extraction")

    turn = await _turn(session, reply)

    assert turn.failure.stage == "listing_question_factual_extraction"
    assert turn.session.pending_clarification is None
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message
    _assert_read_only(session, turn.session)


async def test_continued_update_with_invalid_dates_clears_pending_and_says_so(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Update the search, and make it check out the day before check-in"
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    fakes.patches[reply] = InvalidStayDatesError(StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN)

    turn = await _turn(session, reply)

    assert turn.outcome_kind == "clarification"
    assert fakes.searched == []
    assert turn.session.pending_clarification is None
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message
    _assert_read_only(session, turn.session)


@pytest.mark.parametrize("failure", ["intent_update", "retrieval"])
async def test_failed_new_request_does_not_revive_abandoned_pending(fakes, failure):
    session = await _pending_session(fakes)
    reply = "Actually, find hotels in Paris instead."
    fakes.route(reply, _pending_aware(PendingResolution.NEW_REQUEST, ConversationAction.UPDATE_SEARCH))
    if failure == "intent_update":
        fakes.patches[reply] = LlmCallError("timeout", stage="intent_update")
    else:
        fakes.patches[reply] = SearchIntentPatch(set_city="Paris")
        fakes.fail_searches(RetrievalError("provider_unavailable"))

    turn = await _turn(session, reply)

    assert turn.failure is not None
    assert turn.session.pending_clarification is None
    assert _REQUEST_NOT_COMPLETED_MESSAGE in turn.assistant_message
    _assert_read_only(session, turn.session)

    # The next turn is routed without the abandoned parking request.
    retry = "try again"
    fakes.route(retry, _decision(ConversationAction.GENERAL_CHAT))
    await _turn(turn.session, retry)
    assert fakes.router_inputs[-1].pending_clarification is None


async def test_failures_without_pending_keep_their_existing_message(fakes):
    session = await _searched_session(fakes)
    message = "make it cheaper"
    fakes.route(message, _decision(ConversationAction.UPDATE_SEARCH))
    fakes.patches[message] = LlmCallError("timeout", stage="intent_update")

    turn = await _turn(session, message)

    assert turn.assistant_message == conversation_flow._PROCESSING_UNAVAILABLE_MESSAGE
    assert turn.session.pending_clarification is None


# --- M2: contradictory cancellation ----------------------------------------------------------


async def test_cancel_with_an_executable_action_changes_nothing(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Never mind that, find me hotels in Paris"
    fakes.route(reply, _pending_aware(PendingResolution.CANCEL_PENDING, ConversationAction.START_SEARCH))

    turn = await _turn(session, reply)

    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification == session.pending_clarification
    assert fakes.patch_calls == [] and fakes.searched == []
    _assert_read_only(session, turn.session)


# --- specific listing reference in a continuation -----------------------------------------


def _continue_specific(target: str):
    return PendingAwareConversationActionDecision(
        **_decision(
            ConversationAction.LISTING_QUESTION,
            scope=ResultScope.SPECIFIC_RESULT,
            target=target,
        ).model_dump(),
        pending_resolution=PendingResolution.CONTINUE_PENDING,
    )


async def test_continued_listing_with_valid_target_checks_only_that_hotel(fakes):
    session = await _pending_session(fakes)
    second = session.shown_result_set.items[1]
    reply = "Actually, check the second hotel."
    fakes.route(reply, _continue_specific(second.result_id))

    turn = await _turn(session, reply)

    assert fakes.factual_calls == [(reply, ORIGINAL)]
    assert [i.match.result_id for i in turn.inspector.listing_query] == [second.result_id]
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


@pytest.mark.parametrize("target", ["", "not-a-shown-result"])
async def test_continued_listing_with_invalid_target_asks_which_hotel(fakes, target):
    session = await _pending_session(fakes)
    reply = "Actually, check the second hotel."
    fakes.route(reply, _continue_specific(target))

    turn = await _turn(session, reply)

    assert fakes.factual_calls == [] and fakes.candidate_selections == []
    assert turn.outcome_kind == "clarification"
    for item in session.shown_result_set.items:
        assert item.listing.name in turn.assistant_message
    assert turn.session.pending_clarification == session.pending_clarification
    _assert_read_only(session, turn.session)


async def test_invalid_target_then_valid_reference_resumes_original_question(fakes):
    session = await _pending_session(fakes)
    first_reply = "Actually, check the second hotel."
    fakes.route(first_reply, _continue_specific("not-a-shown-result"))
    asked = await _turn(session, first_reply)

    second = session.shown_result_set.items[1]
    answer = second.listing.name
    fakes.route(answer, _continue_specific(second.result_id))
    turn = await _turn(asked.session, answer)

    assert fakes.router_inputs[-1].pending_clarification == session.pending_clarification
    assert fakes.factual_calls == [(answer, ORIGINAL)]
    assert [i.match.result_id for i in turn.inspector.listing_query] == [second.result_id]
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


# --- M5: original + reply reach factual extraction ------------------------------------------


def _pool() -> UserConstraint:
    return UserConstraint(
        raw_text="pool",
        normalized_text="pool",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


async def test_listing_continuation_keeps_original_and_added_facts(fakes):
    session = await _pending_session(fakes)
    fakes.fake_matcher()
    reply = "Check the current results, and also tell me which ones have a pool."
    fakes.route(reply, CONTINUE_LISTING)
    # What the extractor semantically returns for (original, reply).
    fakes.facts[reply] = ResultQuery(constraints=[_parking(), _pool()])

    turn = await _turn(session, reply)

    assert fakes.factual_calls == [(reply, ORIGINAL)]
    [query] = fakes.matched_queries
    assert [c.normalized_text for c in query.constraints] == ["parking", "pool"]
    for item in turn.inspector.listing_query:
        assert [r.raw_text for r in item.match.constraint_results] == ["parking", "pool"]
    _assert_read_only(session, turn.session)


async def test_action_only_reply_adds_no_facts(fakes):
    session = await _pending_session(fakes)
    fakes.fake_matcher()
    reply = "Check the current results."
    fakes.route(reply, CONTINUE_LISTING)
    fakes.facts[reply] = ResultQuery(constraints=[_parking()])

    await _turn(session, reply)

    [query] = fakes.matched_queries
    assert [c.normalized_text for c in query.constraints] == ["parking"]


async def test_ordinary_listing_question_has_no_clarification_context(fakes):
    session = await _searched_session(fakes)
    message = "Which of these have WiFi?"
    fakes.route(
        message,
        _decision(ConversationAction.LISTING_QUESTION, scope=ResultScope.CURRENT_RESULTS),
    )

    await _turn(session, message)

    assert fakes.factual_calls == [(message, None)]


def test_factual_extraction_input_is_unchanged_without_context():
    assert _build_factual_question_message("Which of these have parking?", None) == (
        "Which of these have parking?"
    )


def test_factual_extraction_input_labels_both_messages():
    reply = "Check the current results, and also which have a pool?"
    text = _build_factual_question_message(reply, ORIGINAL)

    before_reply, reply_part = text.split("Clarification reply:\n")
    assert before_reply.split("Original question:\n")[1].strip() == json.dumps(ORIGINAL)
    assert reply_part == json.dumps(reply)
    assert "across BOTH" in text


# --- M6: original request as read-only context for the reply update ---------------------------


async def test_reply_update_resolves_reference_to_original_requirement(fakes):
    session = await _pending_session(fakes)
    assert [c.normalized_text for c in session.search_request.constraints] == ["WiFi"]
    fakes.capture_searches()
    reply = "Yes, update the search, but make it optional."
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])
    # Semantic outcome of the reply given the original as context: only the
    # change the reply expresses - parking restated as nice, nothing else.
    fakes.patches[reply] = SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE)])

    turn = await _turn(session, reply)

    assert _patch_messages(fakes) == [(ORIGINAL, True), (reply, False)]
    assert _patch_contexts(fakes) == [None, ORIGINAL]
    priorities = {c.normalized_text: c.priority for c in turn.session.search_request.constraints}
    assert priorities == {"WiFi": ConstraintPriority.MUST, "parking": ConstraintPriority.NICE}


async def test_ordinary_update_has_no_answered_request_context(fakes):
    session = await _searched_session(fakes)
    fakes.capture_searches()
    message = "make it cheaper"
    fakes.route(message, _decision(ConversationAction.UPDATE_SEARCH))

    await _turn(session, message)

    assert _patch_contexts(fakes) == [None]


def test_update_prompt_context_is_optional():
    state = SearchRequest(city="Baku")
    plain = intent_update._build_update_prompt(state, "make it optional")
    with_context = intent_update._build_update_prompt(
        state, "make it optional", answered_request=ORIGINAL
    )

    assert "New user message:\nmake it optional\n\nTask:" in plain
    assert "earlier request" not in plain
    assert json.dumps(ORIGINAL) in with_context
    assert "ALREADY been applied" in with_context
    assert "Do not restate or re-add" in with_context
    context_block = with_context[with_context.index("\nThis message answers") : with_context.index("Task:")]
    assert with_context.replace(context_block, "\n") == plain


# --- 10. no pending: unchanged behaviour ----------------------------------------------


async def test_without_pending_router_input_and_flow_are_unchanged(fakes):
    session = await _searched_session(fakes)
    fakes.capture_searches()
    message = "change the search"
    fakes.route(message, _decision(ConversationAction.UPDATE_SEARCH))

    turn = await _turn(session, message)

    [router_input] = fakes.router_inputs
    assert router_input.pending_clarification is None
    assert _patch_messages(fakes) == [(message, False)]
    assert turn.session.pending_clarification is None


# --- 11. routing-contract boundary ----------------------------------------------------


SIX_FIELDS = [
    "action",
    "reason",
    "decision_status",
    "clarification_reason",
    "result_scope",
    "target_result_id",
]


def test_six_field_contract_is_unchanged():
    assert list(ConversationActionDecision.model_fields) == SIX_FIELDS
    assert list(PendingAwareConversationActionDecision.model_fields) == [*SIX_FIELDS, "pending_resolution"]
    assert list(inspect.signature(conversation_router.route_conversation_async).parameters) == [
        "router_input",
        "trace",
        "llm_profile_name",
    ]


def test_pending_aware_output_reduces_to_the_six_field_decision():
    decision = CONTINUE_LISTING.to_decision()

    assert type(decision) is ConversationActionDecision
    assert decision.model_dump() == {k: v for k, v in CONTINUE_LISTING.model_dump().items() if k != "pending_resolution"}


def test_non_generative_router_output_is_enum_only():
    """A classifier (e.g. Jev) only has to emit one more closed label."""
    decision = PendingAwareConversationActionDecision.model_validate(
        {
            "action": "update_search",
            "reason": "",
            "decision_status": "resolved",
            "clarification_reason": "none",
            "result_scope": "not_applicable",
            "target_result_id": "",
            "pending_resolution": "continue_pending",
        }
    )
    schema = PendingAwareConversationActionDecision.model_json_schema()

    assert decision.pending_resolution == PendingResolution.CONTINUE_PENDING
    assert set(schema["required"]) == {*SIX_FIELDS, "pending_resolution"}
    assert set(schema["$defs"]["PendingResolution"]["enum"]) == {r.value for r in PendingResolution}


# --- 12. no text interpretation in Python ---------------------------------------------


def test_transition_rules_never_read_message_text():
    assert list(inspect.signature(resolve_pending_transition).parameters) == ["pending", "decision"]
    assert not hasattr(clarification_continuation, "interpret_clarification_reply")
    phrase_tables = [name for name in vars(clarification_continuation) if name.endswith("_PHRASES")]
    assert phrase_tables == []


@pytest.mark.parametrize(
    ("outcome", "expected_pending_kept", "expected_patches"),
    [
        (CONTINUE_UPDATE, False, [(ORIGINAL, True), ("Update the search.", False)]),
        (_pending_aware(PendingResolution.NEW_REQUEST, ConversationAction.UPDATE_SEARCH), False, [("Update the search.", False)]),
        (UNRESOLVED, True, []),
    ],
)
async def test_same_words_follow_the_semantic_judgement(fakes, outcome, expected_pending_kept, expected_patches):
    """Identical text, different routing judgements -> different outcomes."""
    session = await _pending_session(fakes)
    fakes.capture_searches()
    fakes.route("Update the search.", outcome)

    turn = await _turn(session, "Update the search.")

    assert _patch_messages(fakes) == expected_patches
    assert (turn.session.pending_clarification is not None) == expected_pending_kept


@pytest.mark.parametrize(
    "reply",
    [
        "Could you change what you're searching for?",
        "Поменяй, пожалуйста, условия поиска",
        "Mejor cambia la búsqueda",
    ],
)
async def test_continuation_is_language_independent_given_the_judgement(fakes, reply):
    """G: Python behaves the same for any language once the router judged it."""
    session = await _pending_session(fakes)
    fakes.capture_searches()
    fakes.route(reply, CONTINUE_UPDATE)
    fakes.patches[ORIGINAL] = SearchIntentPatch(add_constraints=[_parking()])

    turn = await _turn(session, reply)

    assert _patch_messages(fakes)[0] == (ORIGINAL, True)
    assert _parking_priorities(turn.session) == [ConstraintPriority.MUST]


# --- transition rule matrix -------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (CONTINUE_UPDATE, PendingTransition.CONTINUE),
        (CONTINUE_LISTING, PendingTransition.CONTINUE),
        (CANCEL, PendingTransition.CANCEL),
        (_pending_aware(PendingResolution.CANCEL_PENDING, ConversationAction.START_SEARCH), PendingTransition.KEEP),
        (_pending_aware(PendingResolution.CANCEL_PENDING, ConversationAction.UPDATE_SEARCH), PendingTransition.KEEP),
        (_pending_aware(PendingResolution.CANCEL_PENDING, ConversationAction.LISTING_QUESTION), PendingTransition.KEEP),
        (
            _pending_aware(
                PendingResolution.CANCEL_PENDING,
                ConversationAction.GENERAL_CHAT,
                status=NEEDS,
                reason=ClarificationReason.AMBIGUOUS_ACTION,
            ),
            PendingTransition.KEEP,
        ),
        (UNRESOLVED, PendingTransition.KEEP),
        (_pending_aware(PendingResolution.NEW_REQUEST, ConversationAction.START_SEARCH), PendingTransition.SUPERSEDE),
        (_pending_aware(PendingResolution.CONTINUE_PENDING, ConversationAction.START_SEARCH), PendingTransition.KEEP),
        (
            _pending_aware(
                PendingResolution.CONTINUE_PENDING,
                ConversationAction.UPDATE_SEARCH,
                status=NEEDS,
                reason=ClarificationReason.AMBIGUOUS_ACTION,
            ),
            PendingTransition.KEEP,
        ),
    ],
)
def test_transition_rules(decision, expected):
    assert resolve_pending_transition(PENDING, decision) == expected


def test_continue_on_unsupported_kind_is_handled_on_its_own():
    reference = PENDING.model_copy(update={"kind": ClarificationReason.AMBIGUOUS_REFERENCE})

    assert resolve_pending_transition(reference, CONTINUE_UPDATE) == PendingTransition.SUPERSEDE


# --- KEEP with an ambiguous listing reference ----------------------------------------------

ACTION_QUESTION = "update the search, or look at the current results?"

AMBIGUOUS_LISTING_REFERENCE = _pending_aware(
    PendingResolution.CONTINUE_PENDING,
    ConversationAction.LISTING_QUESTION,
    status=NEEDS,
    reason=ClarificationReason.AMBIGUOUS_REFERENCE,
    scope=ResultScope.SPECIFIC_RESULT,
)


async def test_ambiguous_listing_reference_asks_which_hotel(fakes):
    session = await _pending_session(fakes)
    fakes.capture_searches()
    reply = "Just check one of the hotels you showed me."
    fakes.route(reply, AMBIGUOUS_LISTING_REFERENCE)

    turn = await _turn(session, reply)

    assert turn.outcome_kind == "clarification"
    assert "Which hotel do you mean" in turn.assistant_message
    for item in session.shown_result_set.items:
        assert item.listing.name in turn.assistant_message
    assert ACTION_QUESTION not in turn.assistant_message
    assert turn.session.pending_clarification == session.pending_clarification
    assert turn.session.pending_clarification.kind == ClarificationReason.AMBIGUOUS_ACTION
    assert fakes.patch_calls == [] and fakes.factual_calls == [] and fakes.searched == []
    _assert_read_only(session, turn.session)


async def test_ambiguous_listing_reference_then_hotel_name_resumes_original_question(fakes):
    session = await _pending_session(fakes)
    reply = "Just check one of the hotels you showed me."
    fakes.route(reply, AMBIGUOUS_LISTING_REFERENCE)
    asked = await _turn(session, reply)

    # The live benchmark's "Fairmont": the user names one shown hotel.
    chosen = session.shown_result_set.items[-1]
    answer = chosen.listing.name
    fakes.route(answer, _continue_specific(chosen.result_id))
    turn = await _turn(asked.session, answer)

    assert fakes.router_inputs[-1].pending_clarification == session.pending_clarification
    assert fakes.factual_calls == [(answer, ORIGINAL)]
    assert [i.match.result_id for i in turn.inspector.listing_query] == [chosen.result_id]
    assert chosen.listing.name in turn.assistant_message
    assert turn.session.pending_clarification is None
    _assert_read_only(session, turn.session)


@pytest.mark.parametrize(
    "outcome",
    [
        UNRESOLVED,
        # Same "continue but unsure" shape, but about the action, not a listing.
        _pending_aware(
            PendingResolution.CONTINUE_PENDING,
            ConversationAction.LISTING_QUESTION,
            status=NEEDS,
            reason=ClarificationReason.AMBIGUOUS_ACTION,
            scope=ResultScope.CURRENT_RESULTS,
        ),
        # An ambiguous reference without a continuation keeps the original question.
        _pending_aware(
            PendingResolution.UNRESOLVED,
            ConversationAction.LISTING_QUESTION,
            status=NEEDS,
            reason=ClarificationReason.AMBIGUOUS_REFERENCE,
            scope=ResultScope.SPECIFIC_RESULT,
        ),
    ],
)
async def test_other_keep_turns_still_ask_update_or_inspect(fakes, outcome):
    session = await _pending_session(fakes)
    reply = "Hmm, I'm not really sure which option is better."
    fakes.route(reply, outcome)

    turn = await _turn(session, reply)

    assert ACTION_QUESTION in turn.assistant_message
    assert "Which hotel do you mean" not in turn.assistant_message
    assert turn.session.pending_clarification == session.pending_clarification
    _assert_read_only(session, turn.session)


# --- listing scope rules in the pending instruction -------------------------------------


def test_pending_instruction_scope_rules_cover_all_three_reference_cases():
    pending = router_agent_module.PENDING_CLARIFICATION_INSTRUCTION
    flat = " ".join(pending.split())

    # The shown set as a whole stays current_results.
    assert "the shown results as a whole (all of them, or which of them) -> result_scope=current_results" in flat
    # An identified listing resolves normally.
    assert "one shown listing that the reply identifies" in flat
    # An unidentified single listing is an unresolved reference - never guessed, never widened.
    assert "one single listing that the reply does NOT identify -> the reference is unresolved" in flat
    assert (
        "result_scope=specific_result, target_result_id=\"\", decision_status=needs_clarification, "
        "clarification_reason=ambiguous_reference" in flat
    )
    assert "Never choose a listing because it is first, most likely, or simply present in shown_results" in flat
    assert "never widen a question about one listing into a question about all shown results" in flat
    # The old default that made current_results the fallback scope is gone.
    assert "For listing_question use result_scope=current_results" not in flat


def test_scope_rules_are_only_sent_on_pending_turns():
    base = " ".join(router_agent_module.CONVERSATION_ROUTER_INSTRUCTION.split())

    assert "never widen a question about one listing" not in base
    assert router_agent_module.PENDING_AWARE_CONVERSATION_ROUTER_INSTRUCTION.startswith(
        router_agent_module.CONVERSATION_ROUTER_INSTRUCTION
    )
