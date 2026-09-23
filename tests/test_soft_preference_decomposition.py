from __future__ import annotations

from app.logic.soft_evidence_routing import DETERMINISTIC_CLAIM_ALIASES
from app.logic.soft_preference_decomposition import (
    CANONICAL_CLAIM_ORDER,
    CLAIM_HYPOTHESES,
    CLAIM_RETRIEVAL_QUERIES,
    PreferenceDirection,
    decompose_constraints,
    get_retrieval_query,
)
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)


def _constraint(text: str, priority: ConstraintPriority = ConstraintPriority.NICE) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


# ---------------- exact claim inventory ----------------


def test_exactly_twelve_claims():
    assert set(CLAIM_HYPOTHESES) == {
        "Q1", "Q2", "Q3", "RW1", "RW2A", "RW2B", "FC1", "FAM1", "BQ1", "CL1", "NL1", "BC1",
    }


def test_hypothesis_text_matches_frozen_benchmark():
    assert CLAIM_HYPOTHESES["Q1"] == "The room/property has soundproofing."
    assert CLAIM_HYPOTHESES["Q2"] == "The immediate surroundings are explicitly described as quiet."
    assert CLAIM_HYPOTHESES["Q3"] == "Guests report little or no significant noise disturbance."
    assert CLAIM_HYPOTHESES["FC1"] == "The property's policies formally accommodate children."
    assert CLAIM_HYPOTHESES["FAM1"] == "The property is suitable for families."
    assert CLAIM_HYPOTHESES["RW1"] == "The room has a desk or dedicated workspace."
    assert CLAIM_HYPOTHESES["RW2A"] == "Wi-Fi / internet access is available."
    assert CLAIM_HYPOTHESES["RW2B"] == "Internet connection is reliable enough for remote work."
    assert CLAIM_HYPOTHESES["BQ1"] == "The breakfast is good in quality."
    assert CLAIM_HYPOTHESES["CL1"] == "The property is clean."
    assert CLAIM_HYPOTHESES["NL1"] == "Nightlife is available nearby."
    assert CLAIM_HYPOTHESES["BC1"] == "The bed is comfortable."


def test_fc2_is_not_produced_anywhere():
    assert "FC2" not in CLAIM_HYPOTHESES
    assert "FC3" not in CLAIM_HYPOTHESES
    assert "FC2" not in DETERMINISTIC_CLAIM_ALIASES
    assert "FC3" not in DETERMINISTIC_CLAIM_ALIASES
    assert "FC2" not in CANONICAL_CLAIM_ORDER


def test_canonical_order_matches_hypothesis_set():
    assert set(CANONICAL_CLAIM_ORDER) == set(CLAIM_HYPOTHESES)


# ---------------- retrieval query fallback ----------------


def test_validated_queries_used_where_they_exist():
    assert get_retrieval_query("Q1") == "soundproofing"
    assert get_retrieval_query("Q2") == "quiet surroundings"
    assert get_retrieval_query("Q3") == "no noise disturbance"
    assert get_retrieval_query("RW1") == "desk or workspace in the room"
    assert get_retrieval_query("RW2A") == "wifi available"
    assert get_retrieval_query("RW2B") == "reliable fast internet connection"
    assert get_retrieval_query("FC1") == "children welcome policy"


def test_unvalidated_queries_fall_back_to_hypothesis_text():
    for claim_id in ("FAM1", "BQ1", "CL1", "NL1", "BC1"):
        assert claim_id not in CLAIM_RETRIEVAL_QUERIES
        assert get_retrieval_query(claim_id) == CLAIM_HYPOTHESES[claim_id]


# ---------------- family matching ----------------


def test_each_family_maps_to_its_claims():
    cases = {
        "quiet neighborhood please": {"Q1", "Q2", "Q3"},
        "good for remote work": {"RW1", "RW2A", "RW2B"},
        "family friendly property": {"FC1", "FAM1"},
        "good breakfast": {"BQ1"},
        "very clean rooms": {"CL1"},
        "close to nightlife": {"NL1"},
        "comfortable bed": {"BC1"},
    }
    for text, expected_claims in cases.items():
        assignments = decompose_constraints([_constraint(text)])
        assert {a.claim_id for a in assignments} == expected_claims


def test_unsupported_constraint_produces_no_claims():
    assignments = decompose_constraints([_constraint("has a kitchen")])
    assert assignments == []


def test_compound_constraint_maps_to_multiple_families():
    assignments = decompose_constraints([_constraint("quiet and good for remote work")])
    claim_ids = {a.claim_id for a in assignments}
    assert claim_ids == {"Q1", "Q2", "Q3", "RW1", "RW2A", "RW2B"}


def test_claim_dedup_across_two_constraints_mapping_to_same_family():
    constraints = [_constraint("quiet please"), _constraint("silent neighborhood")]
    assignments = decompose_constraints(constraints)
    claim_ids = [a.claim_id for a in assignments]
    assert len(claim_ids) == len(set(claim_ids))
    assert set(claim_ids) == {"Q1", "Q2", "Q3"}
    # first-seen constraint wins the assignment
    assert all(a.source_constraint_text == "quiet please" for a in assignments)


# ---------------- preference direction ----------------


def test_forbidden_priority_maps_to_forbidden_direction():
    assignments = decompose_constraints([_constraint("nightlife nearby", priority=ConstraintPriority.FORBIDDEN)])
    assert len(assignments) == 1
    assert assignments[0].claim_id == "NL1"
    assert assignments[0].preference_direction == PreferenceDirection.FORBIDDEN


def test_must_and_nice_priority_map_to_desired_direction():
    for priority in (ConstraintPriority.MUST, ConstraintPriority.NICE):
        assignments = decompose_constraints([_constraint("quiet please", priority=priority)])
        assert all(a.preference_direction == PreferenceDirection.DESIRED for a in assignments)


def test_no_lexical_negation_handling_forbidden_still_produces_factual_claim():
    """
    A FORBIDDEN "nightlife nearby" constraint still produces the ordinary
    factual NL1 claim (not an inverted hypothesis) - the evidence pipeline
    stays factual; only preference_direction carries the polarity.
    """
    assignments = decompose_constraints([_constraint("nightlife nearby", priority=ConstraintPriority.FORBIDDEN)])
    assert assignments[0].hypothesis == "Nightlife is available nearby."


def test_empty_constraints_list():
    assert decompose_constraints([]) == []


def test_assignment_carries_source_constraint_id():
    c = _constraint("quiet please")
    assignments = decompose_constraints([c])
    assert all(a.source_constraint_id == c.id for a in assignments)
