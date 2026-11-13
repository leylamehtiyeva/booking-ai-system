"""
Structured discourse context experiment (TurnConfig.discourse_context).

C2: a question about one shown listing that did not say which becomes an
open question (PendingClarification kind=ambiguous_reference) that the
user's answer resumes. C3: the last answered question about shown
listings (DiscourseContext) is given to the router and the factual
extractor so incomplete follow-ups can be interpreted.

Hermetic: the router, extractors, retrieval and matching are faked. The
router fake returns a scripted semantic decision per message; these tests
check what the application does with it (context lifecycle, invalidation,
identity validation, read-only invariants), not language understanding.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agents import conversation_router_agent as router_agent_module
from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, conversation_router, intent_router, intent_update
from app.logic.clarification_continuation import PendingTransition, resolve_reference_transition
from app.logic.conversation_router import ConversationRoutingError
from app.logic.factual_question_extraction import _build_factual_question_message
from app.logic.result_query_matching import ResultQueryMatchOutcome
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
    DiscourseContext,
    PendingAwareConversationActionDecision,
    PendingClarification,
    PendingResolution,
    ResultScope,
    RouterInput,
)
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.result_query import ResultQuery
from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
from app.schemas.search_response import NormalizedSearchResponse, SearchStatus
from app.schemas.session import ConversationSession
from app.schemas.shown_result_set import ShownResult, ShownResultSet
from app.schemas.turn import TurnConfig
from app.services.conversation_service import process_turn

TODAY = date(2026, 11, 2)
ON = TurnConfig(persist_telemetry=False, include_debug=True, discourse_context=True)
OFF = TurnConfig(persist_telemetry=False, include_debug=True, discourse_context=False)

_FIXTURES = {
    row["id"]: row
    for row in json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "listings_sample.json").read_text())
}
FIRST, SECOND = "baku-hotel-3", "baku-hotel-5"  # Grand City Hotel, City Lights Hotel
NAMES = {FIRST: "Grand City Hotel", SECOND: "City Lights Hotel"}


def _shown(set_id: str = "rs-baku", ids=(FIRST, SECOND)) -> ShownResultSet:
    return ShownResultSet(
        result_set_id=set_id,
        items=[ShownResult(result_id=i, listing=ListingRaw.model_validate(_FIXTURES[i])) for i in ids],
    )


def _session(**kwargs) -> ConversationSession:
    return ConversationSession(
        search_request=SearchRequest(
            city="Baku", check_in=date(2027, 4, 12), check_out=date(2027, 4, 15), property_types=["hotel"]
        ),
        shown_result_set=_shown(),
        **kwargs,
    )


def _fact(text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


def _decision(action, *, status=ConversationDecisionStatus.RESOLVED, reason=ClarificationReason.NONE,
              scope=ResultScope.NOT_APPLICABLE, target=""):
    return ConversationActionDecision(
        action=action, reason="test", decision_status=status, clarification_reason=reason,
        result_scope=scope, target_result_id=target,
    )


def _pending(resolution, action, **kwargs):
    return PendingAwareConversationActionDecision(
        **_decision(action, **kwargs).model_dump(), pending_resolution=resolution
    )


def _ask(target=FIRST):
    return _decision(ConversationAction.LISTING_QUESTION, scope=ResultScope.SPECIFIC_RESULT, target=target)


ALL = _decision(ConversationAction.LISTING_QUESTION, scope=ResultScope.CURRENT_RESULTS)
# A turn that opens a "which one?" question. Under safe scope widening an
# ambiguous_reference abstention is answered for all shown listings, so the
# question is opened by the other source: a target the router claims but that
# is not shown (conversation_flow never guesses, it asks).
WHICH = _ask("not-shown")
ABSTAIN_WHICH = _decision(
    ConversationAction.LISTING_QUESTION,
    status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
    reason=ClarificationReason.AMBIGUOUS_REFERENCE,
    scope=ResultScope.SPECIFIC_RESULT,
)
CONTINUE_FIRST = _pending(
    PendingResolution.CONTINUE_PENDING, ConversationAction.LISTING_QUESTION,
    scope=ResultScope.SPECIFIC_RESULT, target=FIRST,
)
UNRESOLVED = _pending(
    PendingResolution.UNRESOLVED, ConversationAction.LISTING_QUESTION,
    status=ConversationDecisionStatus.NEEDS_CLARIFICATION, reason=ClarificationReason.AMBIGUOUS_REFERENCE,
    scope=ResultScope.SPECIFIC_RESULT,
)


class Fakes:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.routes: dict[str, object] = {}
        self.facts: dict[str, list[str]] = {}
        self.patches: dict[str, SearchIntentPatch] = {}
        self.router_inputs: list[RouterInput] = []
        self.factual_calls: list[tuple[str, dict]] = []
        self.checked: list[list[str]] = []
        self.searched: list[SearchRequest] = []

        def no_model(*args, **kwargs):
            raise LlmCallError("configuration_error", stage="conversation_response_generation")

        monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_model)

        async def route(*, router_input, **kwargs):
            self.router_inputs.append(router_input)
            outcome = self.routes[router_input.user_message]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        async def factual(user_message, *, trace=None, **kwargs):
            self.factual_calls.append((user_message, kwargs))
            return ResultQuery(constraints=[_fact(f) for f in self.facts.get(user_message, ["breakfast"])])

        async def matcher(*, result_query, candidate_selection, **kwargs):
            self.checked.append([c.result_id for c in candidate_selection.candidates])
            return ResultQueryMatchOutcome(
                results=[
                    ResultQueryMatch(
                        result_id=c.result_id,
                        constraint_results=[
                            ConstraintMatchResult(constraint_id=q.id, raw_text=q.raw_text, value=Ternary.UNCERTAIN)
                            for q in result_query.constraints
                        ],
                    )
                    for c in candidate_selection.candidates
                ],
                status="ok",
            )

        async def patch(previous_state, user_message, **kwargs):
            return self.patches.get(user_message, SearchIntentPatch())

        async def orchestrate(state, **kwargs):
            self.searched.append(state)
            return (
                NormalizedSearchResponse(status=SearchStatus.NO_RESULTS, results=[]),
                _shown("rs-new", ("baku-hotel-2", "baku-hotel-6")),
            )

        monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
        monkeypatch.setattr(conversation_flow, "extract_factual_constraints_async", factual)
        monkeypatch.setattr(conversation_flow, "match_result_query_against_candidates", matcher)
        monkeypatch.setattr(conversation_flow, "orchestrate_search_request", orchestrate)
        monkeypatch.setattr(intent_update, "route_intent_update_patch_async", patch)


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    return Fakes(monkeypatch)


async def _turn(session, message, config=ON):
    with clock.use_reference_date(TODAY):
        return await process_turn(session, message, config=config)


def _read_only(before: ConversationSession, after: ConversationSession) -> None:
    assert after.search_request == before.search_request
    assert after.shown_result_set == before.shown_result_set


async def _answered(fakes, message="Does the second one have breakfast?", decision=None, facts=("breakfast",)):
    session = _session()
    fakes.routes[message] = decision or _ask(SECOND)
    fakes.facts[message] = list(facts)
    turn = await _turn(session, message)
    return turn.session


# --- flag off: baseline untouched --------------------------------------------------------


async def test_flag_off_keeps_previous_behaviour(fakes):
    session = _session()
    fakes.routes["Does the second one have breakfast?"] = _ask(SECOND)
    fakes.routes["Does it have a gym?"] = WHICH

    answered = await _turn(session, "Does the second one have breakfast?", OFF)
    asked = await _turn(answered.session, "Does it have a gym?", OFF)

    assert answered.session.discourse_context is None
    assert all(ri.discourse_context is None for ri in fakes.router_inputs)
    assert all(kwargs == {"original_question": None} for _, kwargs in fakes.factual_calls)
    assert asked.session.pending_clarification is None  # no open reference question


# --- C3: context creation and reuse -------------------------------------------------------


async def test_answered_question_becomes_context(fakes):
    session = await _answered(fakes)

    assert session.discourse_context == DiscourseContext(
        result_set_id="rs-baku",
        result_scope=ResultScope.SPECIFIC_RESULT,
        result_ids=[SECOND],
        asked_about=["breakfast"],
    )


async def test_question_about_all_results_becomes_context(fakes):
    session = await _answered(fakes, "Which of these have breakfast?", ALL)

    assert session.discourse_context.result_scope == ResultScope.CURRENT_RESULTS
    assert session.discourse_context.result_ids == [FIRST, SECOND]


async def test_follow_up_changing_only_the_target(fakes):
    session = await _answered(fakes)
    follow_up = "What about the first one?"
    fakes.routes[follow_up] = _ask(FIRST)
    fakes.facts[follow_up] = ["breakfast"]  # what the extractor makes of it, given the context

    turn = await _turn(session, follow_up)

    assert fakes.router_inputs[-1].discourse_context == session.discourse_context
    assert fakes.factual_calls[-1] == (follow_up, {"original_question": None, "previous_facts": ["breakfast"]})
    assert fakes.checked[-1] == [FIRST]
    assert turn.session.discourse_context.result_ids == [FIRST]
    _read_only(session, turn.session)


async def test_follow_up_changing_only_the_fact(fakes):
    session = await _answered(fakes)
    follow_up = "And a gym?"
    fakes.routes[follow_up] = _ask(SECOND)
    fakes.facts[follow_up] = ["gym"]

    turn = await _turn(session, follow_up)

    assert fakes.checked[-1] == [SECOND]
    assert turn.session.discourse_context.asked_about == ["gym"]
    assert turn.session.discourse_context.result_ids == [SECOND]


# --- C2: open reference questions ----------------------------------------------------------


async def test_unidentified_listing_opens_a_question_that_the_answer_resumes(fakes):
    session = _session()
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")

    assert asked.session.pending_clarification == PendingClarification(
        kind=ClarificationReason.AMBIGUOUS_REFERENCE,
        original_user_message="Does it have a gym?",
        possible_actions=[ConversationAction.LISTING_QUESTION],
        result_set_id="rs-baku",
    )
    assert all(name in asked.assistant_message for name in NAMES.values())
    assert fakes.factual_calls == []

    fakes.routes["The first one."] = CONTINUE_FIRST
    fakes.facts["The first one."] = ["gym"]
    turn = await _turn(asked.session, "The first one.")

    assert fakes.router_inputs[-1].pending_clarification == asked.session.pending_clarification
    assert fakes.factual_calls[-1][1]["original_question"] == "Does it have a gym?"
    assert fakes.checked[-1] == [FIRST]
    assert turn.session.pending_clarification is None
    assert turn.session.discourse_context.asked_about == ["gym"]
    _read_only(session, turn.session)


async def test_ambiguous_reference_abstention_is_answered_for_all_shown(fakes):
    session = _session()
    fakes.routes["Does it have a gym?"] = ABSTAIN_WHICH
    fakes.facts["Does it have a gym?"] = ["gym"]

    turn = await _turn(session, "Does it have a gym?")

    assert turn.outcome_kind == "listing_query"
    assert turn.session.pending_clarification is None
    assert fakes.checked[-1] == list(NAMES)
    assert turn.session.discourse_context.result_scope == ResultScope.CURRENT_RESULTS
    _read_only(session, turn.session)


async def test_router_target_outside_the_shown_set_also_opens_the_question(fakes):
    session = _session()
    fakes.routes["Does the third one have a gym?"] = _ask("not-shown")

    turn = await _turn(session, "Does the third one have a gym?")

    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification.kind == ClarificationReason.AMBIGUOUS_REFERENCE
    assert fakes.factual_calls == []


async def test_repeated_uncertainty_keeps_the_open_question(fakes):
    session = _session()
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")
    fakes.routes["Not sure which one I meant."] = UNRESOLVED

    again = await _turn(asked.session, "Not sure which one I meant.")
    third = await _turn(again.session, "Not sure which one I meant.")

    for turn in (again, third):
        assert turn.outcome_kind == "clarification"
        assert all(name in turn.assistant_message for name in NAMES.values())
        assert turn.session.pending_clarification == asked.session.pending_clarification
        _read_only(session, turn.session)
    assert fakes.factual_calls == []


async def test_continuation_without_a_valid_target_keeps_asking(fakes):
    session = _session()
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")
    fakes.routes["That one."] = _pending(
        PendingResolution.CONTINUE_PENDING, ConversationAction.LISTING_QUESTION,
        scope=ResultScope.SPECIFIC_RESULT, target="not-shown",
    )

    turn = await _turn(asked.session, "That one.")

    assert turn.outcome_kind == "clarification"
    assert turn.session.pending_clarification == asked.session.pending_clarification
    assert fakes.factual_calls == []


async def test_cancelled_question_cannot_be_resumed(fakes):
    session = _session()
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")
    fakes.routes["Never mind."] = _pending(PendingResolution.CANCEL_PENDING, ConversationAction.GENERAL_CHAT)
    fakes.routes["The first one."] = _ask(FIRST)

    cancelled = await _turn(asked.session, "Never mind.")
    after = await _turn(cancelled.session, "The first one.")

    assert cancelled.session.pending_clarification is None
    assert fakes.router_inputs[-1].pending_clarification is None
    assert fakes.factual_calls[-1][1]["original_question"] is None  # original question not resumed
    _read_only(session, after.session)


async def test_new_task_supersedes_the_open_question_and_clears_context(fakes):
    session = await _answered(fakes)
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")
    assert asked.session.discourse_context == session.discourse_context  # kept while asking

    move = "Move my dates to April 20-24."
    fakes.routes[move] = _pending(PendingResolution.NEW_REQUEST, ConversationAction.UPDATE_SEARCH)
    fakes.patches[move] = SearchIntentPatch(set_check_in="2027-04-20", set_check_out="2027-04-24")
    turn = await _turn(asked.session, move)

    assert len(fakes.searched) == 1
    assert turn.session.pending_clarification is None
    assert turn.session.discourse_context is None


# --- invalidation ---------------------------------------------------------------------------


async def test_new_result_set_invalidates_context(fakes):
    session = await _answered(fakes)
    change = "Only hotels with a gym"
    fakes.routes[change] = _decision(ConversationAction.UPDATE_SEARCH)
    fakes.patches[change] = SearchIntentPatch(add_constraints=[_fact("gym")])

    changed = await _turn(session, change)
    fakes.routes["What about the first one?"] = _ask("baku-hotel-2")
    await _turn(changed.session, "What about the first one?")

    assert changed.session.shown_result_set.result_set_id == "rs-new"
    assert changed.session.discourse_context is None
    assert fakes.router_inputs[-1].discourse_context is None
    assert "previous_facts" not in fakes.factual_calls[-1][1]


@pytest.mark.parametrize(
    "context",
    [
        DiscourseContext(result_set_id="rs-old", result_scope=ResultScope.SPECIFIC_RESULT,
                         result_ids=[FIRST], asked_about=["breakfast"]),
        DiscourseContext(result_set_id="rs-baku", result_scope=ResultScope.SPECIFIC_RESULT,
                         result_ids=["not-shown"], asked_about=["breakfast"]),
    ],
)
async def test_stale_context_is_never_used(fakes, context):
    session = _session(discourse_context=context)
    fakes.routes["What about the first one?"] = _ask(FIRST)

    turn = await _turn(session, "What about the first one?")

    assert fakes.router_inputs[-1].discourse_context is None
    assert "previous_facts" not in fakes.factual_calls[-1][1]
    assert turn.session.discourse_context.asked_about == ["breakfast"]  # rebuilt from this turn only


async def test_chat_clears_context(fakes):
    session = await _answered(fakes)
    fakes.routes["Thanks!"] = _decision(ConversationAction.GENERAL_CHAT)

    turn = await _turn(session, "Thanks!")

    assert turn.session.discourse_context is None


async def test_failed_turn_keeps_context_and_open_question(fakes):
    session = await _answered(fakes)
    fakes.routes["Does it have a gym?"] = WHICH
    asked = await _turn(session, "Does it have a gym?")
    fakes.routes["The first one."] = ConversationRoutingError("timeout")

    turn = await _turn(asked.session, "The first one.")

    assert turn.failure is not None
    assert turn.session.discourse_context == session.discourse_context
    assert turn.session.pending_clarification == asked.session.pending_clarification


async def test_partially_initialised_new_search_clears_context(fakes, monkeypatch):
    """New search waiting for dates keeps the old shown set - context must not survive it."""
    session = await _answered(fakes)
    start = "Start over: a hotel in Tbilisi"
    fakes.routes[start] = _decision(ConversationAction.START_SEARCH)

    async def intent(user_text, **kwargs):
        return IntentRoute(city="Tbilisi")

    monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)
    turn = await _turn(session, start)

    assert turn.outcome_kind == "clarification"
    assert turn.session.shown_result_set == session.shown_result_set  # existing behaviour
    assert turn.session.discourse_context is None


async def test_listing_turns_never_change_the_search(fakes):
    session = await _answered(fakes)
    for message, decision in [("And the first?", _ask(FIRST)), ("Which have breakfast?", ALL)]:
        fakes.routes[message] = decision
        turn = await _turn(session, message)
        _read_only(session, turn.session)
        session = turn.session


# --- router and extractor inputs --------------------------------------------------------------


def test_router_prompt_has_context_section_only_with_context():
    context = DiscourseContext(result_set_id="rs", result_scope=ResultScope.SPECIFIC_RESULT,
                               result_ids=[FIRST], asked_about=["breakfast"])
    base = RouterInput(user_message="and the first?")

    plain = conversation_router._build_router_prompt(router_input=base)
    with_context = conversation_router._build_router_prompt(
        router_input=base.model_copy(update={"discourse_context": context})
    )

    assert "Conversation context" not in plain
    section = with_context.split("Conversation context:\n", 1)[1].split("\n\nLatest user message:", 1)[0]
    assert json.loads(section) == context.router_view()
    assert "rs" not in json.loads(section)["last_answered_question"].values()


def test_context_instruction_is_only_in_the_context_aware_variants():
    module = router_agent_module
    assert "CONVERSATION CONTEXT" not in module.CONVERSATION_ROUTER_INSTRUCTION
    assert "CONVERSATION CONTEXT" not in module.PENDING_AWARE_CONVERSATION_ROUTER_INSTRUCTION
    assert module.CONTEXT_AWARE_CONVERSATION_ROUTER_INSTRUCTION.startswith(module.CONVERSATION_ROUTER_INSTRUCTION)
    assert module.CONTEXT_AWARE_PENDING_ROUTER_INSTRUCTION.startswith(
        module.PENDING_AWARE_CONVERSATION_ROUTER_INSTRUCTION
    )


class _Runner:
    response = ""

    def __init__(self, **kwargs):
        pass

    def run_async(self, **kwargs):
        text = self.response

        async def events():
            yield SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                                  is_final_response=lambda: True)

        return events()


class _Sessions:
    async def create_session(self, **kwargs):
        return None


def _patch_router(monkeypatch, response) -> dict[str, Mock]:
    builders = {
        name: Mock(return_value=name)
        for name in (
            "build_conversation_router_agent",
            "build_pending_aware_conversation_router_agent",
            "build_context_aware_conversation_router_agent",
        )
    }
    monkeypatch.setattr(conversation_router, "get_llm_profile", Mock(return_value=SimpleNamespace(model="m")))
    monkeypatch.setattr(conversation_router, "build_adk_model", Mock(return_value=object()))
    monkeypatch.setattr(conversation_router, "InMemorySessionService", _Sessions)
    monkeypatch.setattr(conversation_router, "Runner", type("R", (_Runner,), {"response": json.dumps(response)}))
    for name, builder in builders.items():
        monkeypatch.setattr(conversation_router, name, builder)
    return builders


async def test_router_uses_context_instruction_only_when_needed(monkeypatch):
    context = DiscourseContext(result_set_id="rs", result_scope=ResultScope.SPECIFIC_RESULT,
                               result_ids=[FIRST], asked_about=["breakfast"])
    plain = _ask(FIRST).model_dump(mode="json")

    builders = _patch_router(monkeypatch, plain)
    await conversation_router.route_conversation_async(router_input=RouterInput(user_message="x"))
    builders["build_conversation_router_agent"].assert_called_once()
    builders["build_context_aware_conversation_router_agent"].assert_not_called()

    builders = _patch_router(monkeypatch, plain)
    await conversation_router.route_conversation_async(
        router_input=RouterInput(user_message="x", discourse_context=context)
    )
    builders["build_context_aware_conversation_router_agent"].assert_called_once()
    assert builders["build_context_aware_conversation_router_agent"].call_args.kwargs["pending_aware"] is False
    builders["build_conversation_router_agent"].assert_not_called()

    reference = PendingClarification(kind=ClarificationReason.AMBIGUOUS_REFERENCE, original_user_message="x",
                                     possible_actions=[ConversationAction.LISTING_QUESTION], result_set_id="rs")
    builders = _patch_router(monkeypatch, CONTINUE_FIRST.model_dump(mode="json"))
    decision = await conversation_router.route_conversation_async(
        router_input=RouterInput(user_message="the first", pending_clarification=reference)
    )
    assert isinstance(decision, PendingAwareConversationActionDecision)
    assert builders["build_context_aware_conversation_router_agent"].call_args.kwargs["pending_aware"] is True


def test_extractor_input_with_previous_facts():
    assert _build_factual_question_message("What about the first one?", None) == "What about the first one?"
    text = _build_factual_question_message("What about the first one?", None, ["breakfast"])
    assert '["breakfast"]' in text and text.endswith('"What about the first one?"')
    both = _build_factual_question_message("The first one.", "Does it have a gym?", ["breakfast"])
    assert '"Does it have a gym?"' in both and '["breakfast"]' in both


# --- reference transition rules --------------------------------------------------------------

REFERENCE_PENDING = PendingClarification(
    kind=ClarificationReason.AMBIGUOUS_REFERENCE, original_user_message="Does it have a gym?",
    possible_actions=[ConversationAction.LISTING_QUESTION], result_set_id="rs-baku",
)


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (CONTINUE_FIRST, PendingTransition.CONTINUE),
        (_pending(PendingResolution.CONTINUE_PENDING, ConversationAction.LISTING_QUESTION,
                  scope=ResultScope.CURRENT_RESULTS), PendingTransition.CONTINUE),
        (_pending(PendingResolution.CONTINUE_PENDING, ConversationAction.LISTING_QUESTION),
         PendingTransition.KEEP),  # no scope: still not identified
        (_pending(PendingResolution.CONTINUE_PENDING, ConversationAction.UPDATE_SEARCH), PendingTransition.KEEP),
        (UNRESOLVED, PendingTransition.KEEP),
        (_pending(PendingResolution.CANCEL_PENDING, ConversationAction.GENERAL_CHAT), PendingTransition.CANCEL),
        (_pending(PendingResolution.CANCEL_PENDING, ConversationAction.UPDATE_SEARCH), PendingTransition.KEEP),
        (_pending(PendingResolution.NEW_REQUEST, ConversationAction.UPDATE_SEARCH), PendingTransition.SUPERSEDE),
    ],
)
def test_reference_transition_rules(decision, expected):
    assert resolve_reference_transition(REFERENCE_PENDING, decision) == expected


# --- default on (owner decision, router V1.3) ------------------------------------------------
#
# These turns use the DEFAULT TurnConfig (no discourse flag passed), so they check that
# production has the feature on. The router is still faked: they check that the
# Gemini router prompt receives the structured context, and what the application
# does with the decision - not language understanding (that is the router golden
# set's job).

import os
import subprocess
import sys

DEFAULT = TurnConfig(persist_telemetry=False, include_debug=True)
EXPLICIT_CHANGE_RULE = "never turns an explicit change of the search"


def _assert_router_prompt_carries_context(router_input: RouterInput) -> None:
    prompt = conversation_router._build_router_prompt(router_input=router_input)
    section = prompt.split("Conversation context:\n", 1)[1].split("\n\nLatest user message:", 1)[0]
    assert json.loads(section) == router_input.discourse_context.router_view()


def test_discourse_context_is_on_by_default_and_can_be_switched_off():
    assert TurnConfig().discourse_context is True
    code = "from app.schemas.turn import TurnConfig; print(TurnConfig().discourse_context)"
    for value, expected in (("0", "False"), ("off", "False"), ("1", "True"), ("", "True")):
        env = {**os.environ, "BOOKING_DISCOURSE_CONTEXT": value}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parents[1])
        assert out.stdout.strip() == expected, value
    assert TurnConfig(discourse_context=False).discourse_context is False


async def test_default_restaurant_then_gym_continues_the_question_for_all_shown(fakes):
    session = _session()
    first, follow_up = "Can we have dinner at either hotel, any restaurants?", "And a gym?"
    fakes.routes[first], fakes.facts[first] = ALL, ["restaurant"]
    fakes.routes[follow_up], fakes.facts[follow_up] = ALL, ["gym"]

    answered = await _turn(session, first, DEFAULT)
    turn = await _turn(answered.session, follow_up, DEFAULT)

    seen = fakes.router_inputs[-1]
    assert seen.discourse_context == DiscourseContext(
        result_set_id="rs-baku", result_scope=ResultScope.CURRENT_RESULTS, result_ids=[FIRST, SECOND],
        asked_about=["restaurant"])
    _assert_router_prompt_carries_context(seen)
    assert turn.conversation_action == ConversationAction.LISTING_QUESTION
    assert turn.outcome_kind == "listing_query"
    assert fakes.checked[-1] == [FIRST, SECOND]
    assert fakes.factual_calls[-1][1].get("previous_facts") == ["restaurant"]
    _read_only(session, turn.session)
    assert turn.session.discourse_context.asked_about == ["gym"]


async def test_default_follow_up_keeps_the_specific_result(fakes):
    session = _session()
    first, follow_up = "Does the second one have breakfast?", "And parking?"
    fakes.routes[first] = _ask(SECOND)
    fakes.routes[follow_up], fakes.facts[follow_up] = _ask(SECOND), ["parking"]

    answered = await _turn(session, first, DEFAULT)
    turn = await _turn(answered.session, follow_up, DEFAULT)

    seen = fakes.router_inputs[-1]
    assert seen.discourse_context.result_scope == ResultScope.SPECIFIC_RESULT
    assert seen.discourse_context.result_ids == [SECOND]
    _assert_router_prompt_carries_context(seen)
    assert fakes.checked[-1] == [SECOND]
    assert turn.session.discourse_context.result_scope == ResultScope.SPECIFIC_RESULT
    assert turn.session.discourse_context.result_ids == [SECOND]


@pytest.mark.parametrize("kind", ["update_search", "start_search"])
async def test_default_new_search_or_changed_criteria_clears_the_old_context(fakes, monkeypatch, kind):
    session = await _answered(fakes)  # answered under the default config
    assert session.discourse_context is not None
    if kind == "update_search":
        message = "Only hotels with a gym"
        fakes.routes[message] = _decision(ConversationAction.UPDATE_SEARCH)
        fakes.patches[message] = SearchIntentPatch(add_constraints=[_fact("gym")])
    else:
        message = "Start over: a hotel in Tbilisi"
        fakes.routes[message] = _decision(ConversationAction.START_SEARCH)

        async def intent(user_text, **kwargs):
            return IntentRoute(city="Tbilisi")

        monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)

    turn = await _turn(session, message, DEFAULT)
    assert turn.session.discourse_context is None

    fakes.routes["And parking?"] = ALL
    await _turn(turn.session, "And parking?", DEFAULT)
    assert fakes.router_inputs[-1].discourse_context is None


async def test_default_explicit_update_is_not_turned_into_a_listing_question(fakes):
    session = await _answered(fakes, message="Do any of them have a gym?", decision=ALL, facts=("gym",))
    message = "A gym is a must for us"
    fakes.routes[message] = _decision(ConversationAction.UPDATE_SEARCH)
    fakes.patches[message] = SearchIntentPatch(add_constraints=[_fact("gym")])
    checked_before = len(fakes.checked)

    turn = await _turn(session, message, DEFAULT)

    seen = fakes.router_inputs[-1]
    assert seen.discourse_context is not None  # the router saw the context...
    assert turn.conversation_action == ConversationAction.UPDATE_SEARCH  # ...and its update is executed as is
    assert fakes.searched and len(fakes.checked) == checked_before  # searched again; no listing check
    assert turn.session.shown_result_set.result_set_id == "rs-new"
    assert turn.session.discourse_context is None
    # The router is told the context never overrides an explicit change of the search.
    assert EXPLICIT_CHANGE_RULE in router_agent_module.CONTEXT_AWARE_CONVERSATION_ROUTER_INSTRUCTION
    assert EXPLICIT_CHANGE_RULE in router_agent_module.CONTEXT_AWARE_PENDING_ROUTER_INSTRUCTION
