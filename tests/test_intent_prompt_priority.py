"""
D3: priority must reflect the meaning of the request - a direct request
is MUST, an explicitly optional wish is NICE, facts about the traveller
are not constraints. The extraction itself is an LLM call (validated by
the paid e2e run); offline we pin that the instruction states the rule
with near-identical contrasting wordings.
"""

from __future__ import annotations

import pytest

from app.agents.intent_router_agent import build_intent_router_agent


@pytest.fixture(scope="module")
def instruction() -> str:
    return build_intent_router_agent().instruction


@pytest.mark.parametrize(
    "line",
    [
        "-> nice atmosphere -> must",
        "\"Do you have anything with a really nice atmosphere?\"",
        "\"Could you find something that looks really trustworthy?\"",
        "\"I really want a sea view\" -> sea view -> must",
    ],
)
def test_direct_requests_are_must(instruction, line):
    assert line in instruction


@pytest.mark.parametrize(
    "line",
    [
        "\"A balcony would be great\" -> balcony -> nice",
        "\"It would be great if smoking was allowed\" -> smoking allowed -> nice",
        "\"I'd prefer somewhere quiet\" -> quiet -> nice",
    ],
)
def test_explicit_softness_is_nice(instruction, line):
    assert line in instruction


@pytest.mark.parametrize(
    "line",
    [
        "\"I won't be bringing any pets\" -> no constraint",
        "\"I'm travelling for work\" -> no constraint",
    ],
)
def test_traveller_facts_are_not_constraints(instruction, line):
    assert line in instruction


def test_existing_priority_rule_is_kept(instruction):
    assert "Only use priority=\"nice\" when the user explicitly signals softness." in instruction
    assert "bright rooms -> priority=\"must\"" in instruction


# --- the update prompt uses the same rules, plus how to change or remove ---


from app.agents.constraint_priority_rules import CONSTRAINT_PRIORITY_RULES  # noqa: E402
from app.agents.intent_update_agent import build_intent_update_agent  # noqa: E402


@pytest.fixture(scope="module")
def update_instruction() -> str:
    return build_intent_update_agent().instruction


def test_both_prompts_carry_the_same_priority_rules(instruction, update_instruction):
    assert CONSTRAINT_PRIORITY_RULES in instruction
    assert CONSTRAINT_PRIORITY_RULES in update_instruction


@pytest.mark.parametrize(
    "line",
    [
        "\"I'd prefer parking\" -> parking -> nice",
        "\"Ideally, parking\" -> parking -> nice",
        "\"Parking would be great\" -> parking -> nice",
        "\"Parking is essential\" -> parking -> must",
        "\"I absolutely need parking\" -> parking -> must",
        "\"no longer essential, but\nI'd still prefer it\" is a soft preference (nice)",
    ],
)
def test_parking_wordings_are_in_both_prompts(instruction, update_instruction, line):
    assert line in instruction
    assert line in update_instruction


@pytest.mark.parametrize(
    "line",
    [
        "If the user changes how strongly they want an existing constraint - stronger OR weaker - this IS a state change.",
        "The patch merge layer replaces the existing constraint's priority with the one you return.",
        "user says \"Parking is no longer essential, but I'd still prefer it\" -> priority=\"nice\"",
        "(\"forget about parking\", \"I don't care about parking anymore\", \"parking is not needed\"), use remove_constraint_texts.",
        "\"No parking\" / \"avoid parking\" is an exclusion (priority=\"forbidden\"), not a removal.",
        "User: \"Forget about parking\"\nReturn:\n{\"remove_constraint_texts\":[\"parking\"]}",
    ],
)
def test_update_prompt_says_how_to_change_or_remove_a_constraint(update_instruction, line):
    assert line in update_instruction


def test_update_prompt_no_longer_has_the_stale_promote_only_rule(update_instruction):
    assert "deduplicate and promote" not in update_instruction
    assert "\"priority\":\"nice\"" in update_instruction  # a nice example exists now
