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
