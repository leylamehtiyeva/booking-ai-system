"""
Hermetic multi-turn integration tests over process_turn (P0-6).

Real: process_turn, conversation flow, date normalization, patch
application, required-context clarification, fixture retrieval,
structured matching, ranking/selection, reference validation, candidate
selection, deterministic verdicts and rendering, session commit.

Scripted (external LLM boundaries only):
- conversation router  - resolves "first"/"second" against the shown-results
  context it actually receives, the way the LLM would;
- intent extraction / intent update / factual extraction - fixed outputs;
- textual evidence fallback - a google-genai client returning "no evidence";
- response wording - the response model cannot be built, so the real
  deterministic fallback renders every answer.

Every scripted queue must be fully consumed and never over-consumed, so a
turn that calls an LLM it should not (e.g. factual extraction after an
invalid reference) fails the test.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import (
    constraint_evidence_resolution,
    conversation_flow,
    intent_router,
    intent_update,
)
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
    ResultScope,
    RouterInput,
)
from app.schemas.fields import Field
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.result_query import ResultQuery
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig, TurnResult
from app.services import conversation_service
from app.services.conversation_service import process_turn

REFERENCE_DATE = date(2026, 3, 1)
CONFIG = TurnConfig(persist_telemetry=False)  # top_n = 2, as in the app


# --- constraints --------------------------------------------------------------


def _amenity(field: Field, text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text.lower(),
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def WIFI() -> UserConstraint:
    return _amenity(Field.WIFI, "WiFi")


def BALCONY() -> UserConstraint:
    return _amenity(Field.BALCONY, "balcony")


def KITCHEN() -> UserConstraint:
    return _amenity(Field.KITCHEN, "kitchen")


def PETS() -> UserConstraint:
    return _amenity(Field.PET_FRIENDLY, "pet friendly")


# --- scripted LLM boundaries -------------------------------------------------------

RouterStep = Callable[[RouterInput], ConversationActionDecision]


def _decision(
    action: ConversationAction,
    *,
    scope: ResultScope = ResultScope.NOT_APPLICABLE,
    target: str = "",
    status: ConversationDecisionStatus = ConversationDecisionStatus.RESOLVED,
    reason: ClarificationReason = ClarificationReason.NONE,
) -> ConversationActionDecision:
    return ConversationActionDecision(
        action=action,
        reason="scripted",
        decision_status=status,
        clarification_reason=reason,
        result_scope=scope,
        target_result_id=target,
    )


def route(action: ConversationAction) -> RouterStep:
    return lambda router_input: _decision(action)


def ask_about(position: int) -> RouterStep:
    """'the first/second one' -> the result_id at that position in the context the router got."""

    def step(router_input: RouterInput) -> ConversationActionDecision:
        shown = router_input.latest_result_context["shown_results"]
        return _decision(
            ConversationAction.LISTING_QUESTION,
            scope=ResultScope.SPECIFIC_RESULT,
            target=shown[position - 1]["result_id"],
        )

    return step


def ask_about_unknown_id(router_input: RouterInput) -> ConversationActionDecision:
    return _decision(
        ConversationAction.LISTING_QUESTION,
        scope=ResultScope.SPECIFIC_RESULT,
        target="hallucinated-result-id",
    )


class ScriptedLlm:
    def __init__(self, monkeypatch) -> None:
        self.router_steps: list[RouterStep] = []
        self.intents: list[IntentRoute | BaseException] = []
        self.patches: list[SearchIntentPatch | BaseException] = []
        self.factual: list[ResultQuery] = []
        self.router_inputs: list[RouterInput] = []
        self.response_inputs: list[Any] = []
        self.fallback_calls = 0

        monkeypatch.setattr(conversation_flow, "route_conversation_async", self._route)
        monkeypatch.setattr(intent_router, "route_intent_adk_async", self._intent)
        monkeypatch.setattr(intent_update, "route_intent_update_patch_async", self._patch)
        monkeypatch.setattr(conversation_flow, "extract_factual_constraints_async", self._factual)
        monkeypatch.setattr(constraint_evidence_resolution, "_gemini_client", self._genai_client)

        def no_response_model(*args, **kwargs):
            raise LlmCallError("configuration_error", stage="conversation_response_generation")

        monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_response_model)

        real_generate = conversation_service.generate_conversation_response_with_llm

        async def spy_generate(response_input, *, trace=None):
            self.response_inputs.append(response_input)
            return await real_generate(response_input, trace=trace)

        monkeypatch.setattr(conversation_service, "generate_conversation_response_with_llm", spy_generate)

    @staticmethod
    def _pop(queue: list, name: str):
        assert queue, f"unexpected {name} LLM call - nothing scripted"
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def _route(self, *, router_input: RouterInput, trace=None, **kwargs):
        self.router_inputs.append(router_input)
        return self._pop(self.router_steps, "router")(router_input)

    async def _intent(self, user_text, **kwargs):
        return self._pop(self.intents, "intent extraction")

    async def _patch(self, previous_state, user_message, **kwargs):
        return self._pop(self.patches, "intent update")

    async def _factual(self, user_message, **kwargs):
        return self._pop(self.factual, "factual extraction")

    def _genai_client(self):
        llm = self

        class Models:
            def generate_content(self, **kwargs):
                llm.fallback_calls += 1
                return SimpleNamespace(
                    text=json.dumps({"signals": [], "evidence_missing": True, "reason": "No evidence."}),
                    usage_metadata=None,
                )

        return SimpleNamespace(models=Models())

    def assert_consumed(self) -> None:
        leftovers = {
            "router": self.router_steps,
            "intent": self.intents,
            "patch": self.patches,
            "factual": self.factual,
        }
        assert {name: len(q) for name, q in leftovers.items() if q} == {}


@pytest.fixture
def llm(monkeypatch):
    scripted = ScriptedLlm(monkeypatch)
    yield scripted
    scripted.assert_consumed()


class Conversation:
    """Carries the session between turns exactly like a caller of process_turn would."""

    def __init__(self) -> None:
        self.session = ConversationSession()
        self.turns: list[TurnResult] = []

    async def say(self, message: str) -> TurnResult:
        before = self.session.model_dump(mode="json")
        with clock.use_reference_date(REFERENCE_DATE):
            turn = await process_turn(self.session, message, config=CONFIG)
        # process_turn never mutates the session it was given.
        assert self.session.model_dump(mode="json") == before
        assert turn.session.session_id == self.session.session_id == turn.session_id
        assert turn.session.messages[-2:] == [
            turn.session.messages[-2].model_copy(update={"role": "user", "content": message}),
            turn.session.messages[-1].model_copy(update={"role": "assistant", "content": turn.assistant_message}),
        ]
        assert len(turn.session.messages) == len(self.session.messages) + 2
        self.session = turn.session
        self.turns.append(turn)
        return turn


def _shown_ids(session: ConversationSession) -> list[str]:
    return [item.result_id for item in session.shown_result_set.items]


def _listing_name(session: ConversationSession, result_id: str) -> str:
    return next(i.listing.name for i in session.shown_result_set.items if i.result_id == result_id)


def _verdicts(turn: TurnResult) -> dict[str, dict[str, str]]:
    return {
        item.match.result_id: {
            (c.field.value if c.field else c.raw_text): c.value.value
            for c in item.match.constraint_results
        }
        for item in turn.inspector.listing_query
    }


def _assert_search_turn(turn: TurnResult, action: ConversationAction) -> None:
    assert turn.conversation_action == action
    assert turn.outcome_kind == "search"
    assert turn.failure is None
    assert turn.inspector.search_status == "results"
    assert [card.result_id for card in turn.listing_cards] == _shown_ids(turn.session)
    assert [card.position for card in turn.listing_cards] == list(range(1, len(turn.listing_cards) + 1))
    assert len(turn.listing_cards) == CONFIG.top_n


def _initial_search(llm: ScriptedLlm) -> None:
    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(
        IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            property_types=["apartment"],
            constraints=[WIFI()],
        )
    )


START_MESSAGE = "I need an apartment in Baku for April 10-15 with WiFi"


# --- Scenario 1: search -> listing question -----------------------------------------


async def test_search_then_question_about_the_second_result(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)

    _assert_search_turn(search, ConversationAction.START_SEARCH)
    snapshot = search.session.shown_result_set
    assert snapshot is not None and len(snapshot.items) == 2
    second_id = snapshot.items[1].result_id

    llm.router_steps.append(ask_about(2))
    llm.factual.append(ResultQuery(constraints=[KITCHEN(), PETS()]))
    question = await chat.say("Does the second one have a kitchen, and are pets allowed?")

    # The router saw exactly the stored snapshot, in display order.
    assert [r["result_id"] for r in llm.router_inputs[-1].latest_result_context["shown_results"]] == _shown_ids(
        search.session
    )
    assert question.conversation_action == ConversationAction.LISTING_QUESTION
    assert question.outcome_kind == "listing_query"
    assert question.inspector.routing.target_result_id == second_id
    assert list(_verdicts(question)) == [second_id]
    assert _verdicts(question)[second_id]["kitchen"] == "YES"
    assert _listing_name(search.session, second_id) in question.assistant_message
    assert question.listing_cards == []
    # Read-only turn.
    assert question.session.search_request == search.session.search_request
    assert question.session.shown_result_set == snapshot


# --- Scenario 2: search -> update -> listing question -------------------------------


async def test_update_replaces_snapshot_and_references_resolve_against_the_new_one(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)
    old_snapshot = search.session.shown_result_set
    old_first = old_snapshot.items[0].result_id

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[BALCONY()]))
    update = await chat.say("It must also have a balcony")

    _assert_search_turn(update, ConversationAction.UPDATE_SEARCH)
    before, after = search.session.search_request, update.session.search_request
    assert [c.normalized_text for c in after.constraints] == ["wifi", "balcony"]
    assert after.constraints[0] == before.constraints[0]
    assert (after.city, after.check_in, after.check_out, after.property_types) == (
        before.city,
        before.check_in,
        before.check_out,
        before.property_types,
    )
    new_snapshot = update.session.shown_result_set
    assert new_snapshot.result_set_id != old_snapshot.result_set_id
    new_first = new_snapshot.items[0].result_id
    assert new_first != old_first
    assert old_first not in _shown_ids(update.session)

    llm.router_steps.append(ask_about(1))
    llm.factual.append(ResultQuery(constraints=[BALCONY()]))
    question = await chat.say("Does the first one have a balcony?")

    assert [r["result_id"] for r in llm.router_inputs[-1].latest_result_context["shown_results"]] == _shown_ids(
        update.session
    )
    assert question.inspector.routing.target_result_id == new_first
    assert _verdicts(question) == {new_first: {"balcony": "YES"}}
    assert _listing_name(update.session, new_first) in question.assistant_message
    assert question.session.shown_result_set == new_snapshot
    assert question.session.search_request == after


# --- Scenario 3: no-op update -----------------------------------------------------


async def test_restating_an_existing_requirement_is_a_no_op(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)
    fallback_calls_after_search = llm.fallback_calls

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[WIFI()]))  # a fresh constraint id
    no_op = await chat.say("Only places with WiFi, please")

    assert no_op.conversation_action == ConversationAction.UPDATE_SEARCH
    assert no_op.outcome_kind == "informational"
    assert no_op.failure is None
    assert no_op.assistant_message == "Your search hasn't changed, so I didn't run a new search."
    assert no_op.inspector.response_source == "deterministic"
    assert no_op.listing_cards == []
    assert no_op.inspector.search_status is None  # no search ran
    assert llm.fallback_calls == fallback_calls_after_search
    assert no_op.session.search_request == search.session.search_request
    assert no_op.session.shown_result_set == search.session.shown_result_set


# --- Scenario 4: invalid listing reference ------------------------------------------


async def test_hallucinated_reference_asks_instead_of_guessing(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)

    llm.router_steps.append(ask_about_unknown_id)
    # Nothing scripted for factual extraction: calling it would fail the test.
    question = await chat.say("Does that one have parking?")

    assert question.conversation_action == ConversationAction.LISTING_QUESTION
    assert question.outcome_kind == "clarification"
    assert question.failure is None
    assert question.inspector.listing_query == []
    for title in (item.listing.name for item in search.session.shown_result_set.items):
        assert title in question.assistant_message  # "Which hotel do you mean - A or B?"
    assert question.session.search_request == search.session.search_request
    assert question.session.shown_result_set == search.session.shown_result_set


# --- Scenario 5: missing information -> clarification -> search ---------------------


async def test_missing_dates_are_asked_for_then_completed_by_an_update(llm):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(IntentRoute(city="Baku", property_types=["apartment"], constraints=[WIFI()]))
    clarification = await chat.say("Find me an apartment in Baku with WiFi")

    assert clarification.conversation_action == ConversationAction.START_SEARCH
    assert clarification.outcome_kind == "clarification"
    assert "dates" in clarification.assistant_message.lower()
    assert clarification.listing_cards == []
    partial = clarification.session.search_request
    assert (partial.city, partial.check_in, partial.check_out) == ("Baku", None, None)
    assert [c.normalized_text for c in partial.constraints] == ["wifi"]
    assert clarification.session.shown_result_set is None

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(set_check_in="2026-04-10", set_check_out="2026-04-15"))
    search = await chat.say("April 10-15")

    # The router saw the partial search from the clarification turn.
    assert llm.router_inputs[-1].current_search == partial
    _assert_search_turn(search, ConversationAction.UPDATE_SEARCH)
    final = search.session.search_request
    assert (final.city, final.check_in, final.check_out) == ("Baku", date(2026, 4, 10), date(2026, 4, 15))
    assert [p.value for p in final.property_types] == ["apartment"]
    assert final.constraints == partial.constraints


# --- Scenario 6: failure -> recovery ----------------------------------------------


async def test_provider_failure_mid_conversation_then_recovery(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(LlmCallError("rate_limited", stage="intent_update", retryable=True))
    failed = await chat.say("Add a balcony")

    assert failed.conversation_action == ConversationAction.UPDATE_SEARCH
    assert failed.outcome_kind == "failure"
    assert failed.failure.model_dump() == {"stage": "search_state_update", "code": "rate_limited"}
    assert "has not been changed" in failed.assistant_message
    assert failed.listing_cards == []
    assert failed.session.search_request == search.session.search_request
    assert failed.session.shown_result_set == search.session.shown_result_set

    # The conversation continues on the last valid state: a reference into
    # the pre-failure snapshot, then the update that failed before.
    llm.router_steps.append(ask_about(2))
    llm.factual.append(ResultQuery(constraints=[KITCHEN()]))
    question = await chat.say("Does the second one have a kitchen?")

    second_id = search.session.shown_result_set.items[1].result_id
    assert llm.router_inputs[-1].current_search == search.session.search_request
    assert question.inspector.routing.target_result_id == second_id
    assert _verdicts(question) == {second_id: {"kitchen": "YES"}}

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[BALCONY()]))
    retried = await chat.say("Add a balcony")

    _assert_search_turn(retried, ConversationAction.UPDATE_SEARCH)
    assert [c.normalized_text for c in retried.session.search_request.constraints] == ["wifi", "balcony"]
    assert retried.session.shown_result_set.result_set_id != search.session.shown_result_set.result_set_id


# --- Scenario 7: general chat between search turns -----------------------------------


async def test_general_chat_between_search_turns_leaves_state_alone(llm):
    chat = Conversation()

    _initial_search(llm)
    search = await chat.say(START_MESSAGE)

    llm.router_steps.append(route(ConversationAction.GENERAL_CHAT))
    chat_turn = await chat.say("Thanks! By the way, what can you do?")

    assert chat_turn.conversation_action == ConversationAction.GENERAL_CHAT
    assert chat_turn.outcome_kind == "general_chat"
    assert chat_turn.listing_cards == []
    assert chat_turn.session.search_request == search.session.search_request
    assert chat_turn.session.shown_result_set == search.session.shown_result_set

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[BALCONY()]))
    update = await chat.say("Show me ones with a balcony too")

    assert llm.router_inputs[-1].current_search == search.session.search_request
    _assert_search_turn(update, ConversationAction.UPDATE_SEARCH)
    assert [c.normalized_text for c in update.session.search_request.constraints] == ["wifi", "balcony"]

    # History: every turn appended user + assistant, in order, and the
    # response layer for the update saw the earlier exchanges (not the
    # current message).
    assert [m.role for m in update.session.messages] == ["user", "assistant"] * 3
    assert [m.content for m in update.session.messages[::2]] == [
        START_MESSAGE,
        "Thanks! By the way, what can you do?",
        "Show me ones with a balcony too",
    ]
    history_for_update = [m.content for m in llm.response_inputs[-1].recent_messages]
    assert history_for_update == [m.content for m in chat_turn.session.messages]
