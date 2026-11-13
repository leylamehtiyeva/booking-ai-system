"""
Routing policy V1.3 is stated in the Gemini router instructions.
Offline: string checks over the instructions, no model call.

Gemini-only port of the policy test from the router evaluation line
(eval/router-gold-v1.4): the Jev criteria and the golden-set overlap check
stay with that experiment and are not part of the application.
"""

from __future__ import annotations

from app.agents import conversation_router_agent as gemini

# The prompt examples of the V1.3 rules.
SHORT_FOLLOW_UPS = ("Room service?", "And a sauna?", "Airport shuttle included?", "Breakfast in the price?")
OTHER_OPTIONS = ("Can you find ones with a sauna instead?", "Do you have anything cheaper?")
STATED_REQUIREMENTS = ("A sauna is a must for us", "Add room service as a requirement", "Only hotels with a sauna")
EXPLICIT_CHANGE_RULE = "never turns an explicit change of the search"


def _flat(text: str) -> str:
    return " ".join(text.split())


GEMINI = _flat(gemini.CONVERSATION_ROUTER_INSTRUCTION)


def test_short_follow_up_questions_are_listing_questions():
    assert "SHORT FOLLOW-UP QUESTIONS ABOUT SHOWN RESULTS" in GEMINI
    for example in SHORT_FOLLOW_UPS:
        assert f'"{example}" -> listing_question / current_results' in GEMINI


def test_explicit_search_changes_stay_update_search():
    for example in OTHER_OPTIONS + STATED_REQUIREMENTS:
        assert example in GEMINI
    assert "These stay update_search even when results are shown" in GEMINI
    assert "An explicit change of the existing search is update_search" in GEMINI


def test_ambiguous_action_needs_both_readings():
    assert "Use ambiguous_action only when the message itself points to BOTH readings" in GEMINI


def test_no_shown_results_rule():
    assert "NO results are shown yet" in GEMINI
    assert "With a current search and no shown results, a mentioned requirement is update_search" in GEMINI


def test_explicit_change_rule_only_in_the_conversation_context_section():
    assert EXPLICIT_CHANGE_RULE in _flat(gemini.DISCOURSE_CONTEXT_INSTRUCTION)
    assert EXPLICIT_CHANGE_RULE not in GEMINI


def test_every_router_variant_carries_the_policy():
    # The pending-aware and context-aware instructions are composed from the
    # base instruction, so the V1.3 policy reaches every router call.
    for name in (
        "PENDING_AWARE_CONVERSATION_ROUTER_INSTRUCTION",
        "CONTEXT_AWARE_CONVERSATION_ROUTER_INSTRUCTION",
        "CONTEXT_AWARE_PENDING_ROUTER_INSTRUCTION",
    ):
        assert "SHORT FOLLOW-UP QUESTIONS ABOUT SHOWN RESULTS" in _flat(getattr(gemini, name)), name
