from __future__ import annotations

import pytest

from app.logic.soft_constraint_decision_adapter import soft_constraint_decision_to_resolution_dict
from app.logic.soft_constraint_decision_policy import (
    apply_preference_direction,
    combine_family_states,
    decide_constraint,
    families_for_constraint,
)
from app.logic.soft_preference_decomposition import CLAIM_HYPOTHESES, PreferenceDirection
from app.schemas.constraints import ConstraintPriority, UserConstraint
from app.schemas.soft_constraint_decision import ConstraintDecisionValue, FactualState
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimRelation,
    EvidenceItem,
    EvidenceRelation,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SoftPreferenceEvidence,
)


def claim(
    claim_id: str,
    relation: ClaimRelation,
    *,
    retrieval_status: RetrievalStatus = RetrievalStatus.SUCCESS,
    retrieval_errors: list[str] | None = None,
    evidence_items: list[EvidenceItem] | None = None,
) -> AtomicClaimResult:
    return AtomicClaimResult(
        claim_id=claim_id,
        hypothesis=CLAIM_HYPOTHESES[claim_id],
        relation=relation,
        evidence_items=evidence_items or [],
        retrieval_status=retrieval_status,
        retrieval_errors=retrieval_errors or ([] if retrieval_status == RetrievalStatus.SUCCESS else ["boom"]),
    )


def evidence(*claims: AtomicClaimResult) -> SoftPreferenceEvidence:
    return SoftPreferenceEvidence(claims=list(claims), semantic_verifier=None)


def resolved_item(text: str, relation: EvidenceRelation) -> EvidenceItem:
    return EvidenceItem(
        evidence_text=text,
        relation=relation,
        resolution_method="gemini",
        resolution_status=EvidenceResolutionStatus.RESOLVED,
    )


def constraint(text: str, priority: ConstraintPriority, *, constraint_id: str = "c1") -> UserConstraint:
    return UserConstraint(
        id=constraint_id, raw_text=text, normalized_text=text, priority=priority,
    )


SUP, CON, NEE, MIX = (
    ClaimRelation.SUPPORT,
    ClaimRelation.CONTRADICT,
    ClaimRelation.NOT_ENOUGH_EVIDENCE,
    ClaimRelation.MIXED,
)


# ==================================================================
# Q3 attribution bug fix - end-to-end through clean_and_route_candidates
# (routing-unit tests live in tests/test_soft_evidence_routing.py)
# ==================================================================


def test_q3_non_review_evidence_never_reaches_free_text_routing():
    from app.logic.soft_evidence_cleanup import clean_and_route_candidates
    from app.logic.soft_evidence_retrieval import RetrievedEvidence

    candidates = [
        RetrievedEvidence(
            text="Quiet street view", source_type="facilities",
            source_path="listing.facilities[4].name", retrieval_score=0.9,
        ),
    ]
    routed = clean_and_route_candidates("Q3", candidates)

    assert routed.free_text_candidates == []
    assert len(routed.deterministic_items) == 1
    assert routed.deterministic_items[0].relation == EvidenceRelation.NOT_ENOUGH_EVIDENCE
    assert routed.deterministic_items[0].resolution_method.value == "deterministic"


def test_q3_review_summary_evidence_reaches_free_text_routing():
    from app.logic.soft_evidence_cleanup import clean_and_route_candidates
    from app.logic.soft_evidence_retrieval import RetrievedEvidence

    candidates = [
        RetrievedEvidence(
            text="The room was very quiet and we heard no traffic at night.",
            source_type="review_summary",
            source_path="raw.reviewSummary.pros[0].description", retrieval_score=0.9,
        ),
    ]
    routed = clean_and_route_candidates("Q3", candidates)

    assert routed.deterministic_items == []
    assert len(routed.free_text_candidates) == 1
    assert routed.free_text_candidates[0].text == "The room was very quiet and we heard no traffic at night."


def test_q2_same_evidence_still_reaches_free_text_routing():
    """The Q3 fix must not affect Q2 - a property-description claim."""
    from app.logic.soft_evidence_cleanup import clean_and_route_candidates
    from app.logic.soft_evidence_retrieval import RetrievedEvidence

    candidates = [
        RetrievedEvidence(
            text="Quiet street view", source_type="description",
            source_path="listing.description", retrieval_score=0.9,
        ),
    ]
    routed = clean_and_route_candidates("Q2", candidates)

    assert routed.deterministic_items == []
    assert len(routed.free_text_candidates) == 1


def test_q3_all_non_review_pool_produces_not_enough_evidence_claim_not_false_support():
    """
    Full regression proof: a hotel whose only Q3-retrieved evidence is
    non-review text ("Quiet street view") must resolve Q3 to
    NOT_ENOUGH_EVIDENCE (and therefore never drive a false quiet YES via
    Q3 alone) - the routing fix, not the aggregation policy, is what
    prevents the false positive.
    """
    from app.logic.soft_evidence_cleanup import clean_and_route_candidates
    from app.logic.soft_evidence_retrieval import RetrievedEvidence
    from app.schemas.soft_evidence import AtomicClaimResult

    candidates = [
        RetrievedEvidence(
            text="Quiet street view", source_type="facilities",
            source_path="listing.facilities[4].name", retrieval_score=0.9,
        ),
    ]
    routed = clean_and_route_candidates("Q3", candidates)
    assert routed.free_text_candidates == []  # nothing left for Gemini to (mis)score

    result = AtomicClaimResult.from_evidence_items(
        claim_id="Q3", hypothesis=CLAIM_HYPOTHESES["Q3"], evidence_items=routed.deterministic_items,
    )
    assert result.relation.value == "NOT_ENOUGH_EVIDENCE"


# ==================================================================
# QUIET - candidate-matching semantics (Q1=soundproofing,
# Q2=quiet surroundings, Q3=guest-reported noise experience)
# ==================================================================


def test_quiet_q1_support_alone_is_satisfied():
    """Soundproofing alone is a useful positive search signal -> YES."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", SUP), "Q2": claim("Q2", NEE), "Q3": claim("Q3", NEE)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.SATISFIED
    assert outcome.decisive_claim_ids == ("Q1",)


def test_quiet_q2_support_alone_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", NEE), "Q2": claim("Q2", SUP), "Q3": claim("Q3", NEE)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.SATISFIED
    assert outcome.decisive_claim_ids == ("Q2",)


def test_quiet_q3_support_alone_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", NEE), "Q2": claim("Q2", NEE), "Q3": claim("Q3", SUP)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.SATISFIED
    assert outcome.decisive_claim_ids == ("Q3",)


def test_quiet_all_not_enough_evidence_is_unresolved():
    """Nizami Hotel scenario: Q1/Q2/Q3 all NEE -> UNCERTAIN."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", NEE), "Q2": claim("Q2", NEE), "Q3": claim("Q3", NEE)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_quiet_support_plus_mixed_is_unresolved():
    """Q1 SUPPORT, Q2 MIXED, Q3 SUPPORT -> UNCERTAIN: material conflict, not a confident YES."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", SUP), "Q2": claim("Q2", MIX), "Q3": claim("Q3", SUP)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.UNRESOLVED
    assert set(outcome.decisive_claim_ids) >= {"Q2"}  # the conflicting claim is preserved


def test_quiet_support_plus_explicit_contradiction_is_unresolved_not_no():
    """
    Q1 SUPPORT, Q2 NEE, Q3 CONTRADICT -> UNCERTAIN. Must NOT discard the
    candidate as if we know the hotel is objectively noisy - the
    conflict is preserved for the user, not silently resolved to NO.
    """
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", SUP), "Q2": claim("Q2", NEE), "Q3": claim("Q3", CON)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.UNRESOLVED
    assert set(outcome.decisive_claim_ids) == {"Q1", "Q3"}


def test_quiet_contradiction_with_no_positive_signal_is_violated():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", NEE), "Q2": claim("Q2", NEE), "Q3": claim("Q3", CON)}
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.VIOLATED
    assert outcome.decisive_claim_ids == ("Q3",)


def test_quiet_partial_retrieval_contradiction_still_violates():
    """A contradiction found in a successfully-retrieved partial batch is real evidence."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {
        "Q1": claim("Q1", NEE),
        "Q2": claim("Q2", CON, retrieval_status=RetrievalStatus.PARTIAL),
        "Q3": claim("Q3", NEE),
    }
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.VIOLATED


def test_quiet_partial_retrieval_support_does_not_satisfy():
    """PARTIAL retrieval must not drive a positive SATISFIED conclusion."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {
        "Q1": claim("Q1", NEE),
        "Q2": claim("Q2", NEE),
        "Q3": claim("Q3", SUP, retrieval_status=RetrievalStatus.PARTIAL),
    }
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_quiet_failed_retrieval_support_does_not_satisfy():
    """Defensive: even if relation were somehow SUPPORT, FAILED retrieval can't SATISFY."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {
        "Q1": claim("Q1", NEE),
        "Q2": claim("Q2", NEE),
        "Q3": claim("Q3", SUP, retrieval_status=RetrievalStatus.FAILED),
    }
    outcome = compute_family_factual_state("quiet", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_quiet_reason_never_asserts_a_guarantee():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"Q1": claim("Q1", SUP), "Q2": claim("Q2", NEE), "Q3": claim("Q3", NEE)}
    outcome = compute_family_factual_state("quiet", claims)
    assert "this hotel is quiet" not in outcome.reason_fragment.lower()
    assert "soundproofing" in outcome.reason_fragment.lower()


# ==================================================================
# REMOTE WORK - candidate-matching semantics (RW1=desk, RW2A=wifi,
# RW2B=reliability, never required for YES)
# ==================================================================


def test_remote_work_desk_wifi_reliability_all_support_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", SUP), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", SUP)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.SATISFIED


def test_remote_work_desk_and_wifi_with_unknown_reliability_is_satisfied():
    """Nizami / Metro City / Maestro scenario: RW1+RW2A SUPPORT, RW2B NEE -> YES."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", SUP), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.SATISFIED
    assert outcome.decisive_claim_ids == ("RW1", "RW2A")
    # reason must explicitly say reliability was not established, never claim it IS reliable
    reason = outcome.reason_fragment.lower()
    assert "reliability" in reason and "not established" in reason
    assert "reliable connectivity" not in reason


def test_remote_work_wifi_only_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", NEE), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_remote_work_desk_only_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", SUP), "RW2A": claim("RW2A", NEE), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_remote_work_essential_contradiction_no_offsetting_support_is_violated():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", CON), "RW2A": claim("RW2A", NEE), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.VIOLATED
    assert outcome.decisive_claim_ids == ("RW1",)


def test_remote_work_essential_contradiction_with_offsetting_support_is_unresolved():
    """RW1 CONTRADICT (no desk) but RW2A SUPPORT (good wifi) - conflicting, not a clean NO."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", CON), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_remote_work_reliability_contradiction_despite_desk_and_wifi_is_unresolved():
    """RW1+RW2A SUPPORT but RW2B explicitly CONTRADICT - material conflict, not YES."""
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", SUP), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", CON)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_remote_work_mixed_reliability_despite_desk_and_wifi_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", SUP), "RW2A": claim("RW2A", SUP), "RW2B": claim("RW2B", MIX)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_remote_work_no_evidence_at_all_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"RW1": claim("RW1", NEE), "RW2A": claim("RW2A", NEE), "RW2B": claim("RW2B", NEE)}
    outcome = compute_family_factual_state("remote_work", claims)
    assert outcome.state == FactualState.UNRESOLVED


# ==================================================================
# FAMILY FRIENDLY - FC1 (formal policy) carries stronger weight than
# FAM1 (descriptive suitability)
# ==================================================================


def test_family_friendly_both_support_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", SUP), "FAM1": claim("FAM1", SUP)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.SATISFIED


def test_family_friendly_fc1_support_alone_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", SUP), "FAM1": claim("FAM1", NEE)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.SATISFIED


def test_family_friendly_fam1_support_alone_is_satisfied():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", NEE), "FAM1": claim("FAM1", SUP)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.SATISFIED


def test_family_friendly_fc1_contradict_is_violated_regardless_of_fam1():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", CON), "FAM1": claim("FAM1", SUP)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.VIOLATED
    assert outcome.decisive_claim_ids == ("FC1",)


def test_family_friendly_positive_plus_conflicting_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", SUP), "FAM1": claim("FAM1", CON)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.UNRESOLVED


def test_family_friendly_no_evidence_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {"FC1": claim("FC1", NEE), "FAM1": claim("FAM1", NEE)}
    outcome = compute_family_factual_state("family_friendly", claims)
    assert outcome.state == FactualState.UNRESOLVED


# ==================================================================
# SINGLE-CLAIM FAMILIES
# ==================================================================


@pytest.mark.parametrize("family,claim_id", [
    ("breakfast_quality", "BQ1"),
    ("cleanliness", "CL1"),
    ("nightlife", "NL1"),
    ("bed_comfort", "BC1"),
])
@pytest.mark.parametrize("relation,expected", [
    (SUP, FactualState.SATISFIED),
    (CON, FactualState.VIOLATED),
    (MIX, FactualState.UNRESOLVED),
    (NEE, FactualState.UNRESOLVED),
])
def test_single_claim_families(family, claim_id, relation, expected):
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    claims = {claim_id: claim(claim_id, relation)}
    outcome = compute_family_factual_state(family, claims)
    assert outcome.state == expected


def test_single_claim_missing_claim_is_unresolved():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    outcome = compute_family_factual_state("nightlife", {})
    assert outcome.state == FactualState.UNRESOLVED
    assert "not evaluated" in outcome.technical_notes[0]


def test_unknown_family_raises():
    from app.logic.soft_constraint_decision_policy import compute_family_factual_state

    with pytest.raises(ValueError):
        compute_family_factual_state("not_a_real_family", {})


# ==================================================================
# combine_family_states
# ==================================================================


def test_combine_empty_is_unresolved():
    assert combine_family_states([]) == FactualState.UNRESOLVED


def test_combine_any_violated_wins_over_satisfied():
    assert combine_family_states([FactualState.SATISFIED, FactualState.VIOLATED]) == FactualState.VIOLATED


def test_combine_all_satisfied_is_satisfied():
    assert combine_family_states([FactualState.SATISFIED, FactualState.SATISFIED]) == FactualState.SATISFIED


def test_combine_mixed_satisfied_and_unresolved_is_unresolved():
    assert combine_family_states([FactualState.SATISFIED, FactualState.UNRESOLVED]) == FactualState.UNRESOLVED


# ==================================================================
# apply_preference_direction (full truth table)
# ==================================================================


@pytest.mark.parametrize("state,expected", [
    (FactualState.SATISFIED, ConstraintDecisionValue.YES),
    (FactualState.VIOLATED, ConstraintDecisionValue.NO),
    (FactualState.UNRESOLVED, ConstraintDecisionValue.UNCERTAIN),
])
def test_apply_preference_direction_desired(state, expected):
    assert apply_preference_direction(state, PreferenceDirection.DESIRED) == expected


@pytest.mark.parametrize("state,expected", [
    (FactualState.SATISFIED, ConstraintDecisionValue.NO),
    (FactualState.VIOLATED, ConstraintDecisionValue.YES),
    (FactualState.UNRESOLVED, ConstraintDecisionValue.UNCERTAIN),
])
def test_apply_preference_direction_forbidden(state, expected):
    assert apply_preference_direction(state, PreferenceDirection.FORBIDDEN) == expected


# ==================================================================
# families_for_constraint
# ==================================================================


def test_families_for_constraint_matches_quiet():
    c = constraint("quiet please", ConstraintPriority.NICE)
    assert families_for_constraint(c) == ["quiet"]


def test_families_for_constraint_unsupported_returns_empty():
    c = constraint("has a kitchen", ConstraintPriority.NICE)
    assert families_for_constraint(c) == []


def test_families_for_constraint_can_match_multiple_families():
    c = constraint("quiet and good for remote work", ConstraintPriority.MUST)
    families = families_for_constraint(c)
    assert set(families) == {"quiet", "remote_work"}


# ==================================================================
# decide_constraint - end-to-end deterministic aggregation
# ==================================================================


def test_decide_constraint_unsupported_returns_none():
    c = constraint("has a kitchen", ConstraintPriority.NICE)
    ev = evidence()
    assert decide_constraint(c, ev) is None


def test_decide_constraint_desired_quiet_satisfied_is_yes():
    c = constraint("quiet please", ConstraintPriority.MUST)
    ev = evidence(claim("Q1", SUP), claim("Q2", SUP), claim("Q3", NEE))
    decision = decide_constraint(c, ev)
    assert decision is not None
    assert decision.factual_state == FactualState.SATISFIED
    assert decision.decision == ConstraintDecisionValue.YES
    assert decision.families == ["quiet"]
    assert decision.preference_direction == PreferenceDirection.DESIRED
    assert decision.confidence is None
    assert decision.reason.startswith("YES:")


# --- FORBIDDEN nightlife examples from the Phase C spec (section 9) ---


def test_decide_constraint_forbidden_nightlife_support_is_no():
    c = constraint("no nightlife nearby please", ConstraintPriority.FORBIDDEN)
    ev = evidence(claim("NL1", SUP))
    decision = decide_constraint(c, ev)
    assert decision.factual_state == FactualState.SATISFIED  # nightlife IS present, factually
    assert decision.decision == ConstraintDecisionValue.NO
    assert decision.reason.startswith("NO:")


def test_decide_constraint_forbidden_nightlife_contradict_is_yes():
    c = constraint("no nightlife nearby please", ConstraintPriority.FORBIDDEN)
    ev = evidence(claim("NL1", CON))
    decision = decide_constraint(c, ev)
    assert decision.factual_state == FactualState.VIOLATED
    assert decision.decision == ConstraintDecisionValue.YES


@pytest.mark.parametrize("relation", [NEE, MIX])
def test_decide_constraint_forbidden_nightlife_nee_or_mixed_is_uncertain(relation):
    c = constraint("no nightlife nearby please", ConstraintPriority.FORBIDDEN)
    ev = evidence(claim("NL1", relation))
    decision = decide_constraint(c, ev)
    assert decision.factual_state == FactualState.UNRESOLVED
    assert decision.decision == ConstraintDecisionValue.UNCERTAIN


def test_decide_constraint_multi_family_combination_any_violated_wins():
    c = constraint("quiet and good for remote work", ConstraintPriority.MUST)
    ev = evidence(
        claim("Q1", SUP), claim("Q2", SUP), claim("Q3", NEE),  # quiet -> SATISFIED
        claim("RW1", CON), claim("RW2A", NEE), claim("RW2B", NEE),  # remote_work -> VIOLATED (no offset)
    )
    decision = decide_constraint(c, ev)
    assert set(decision.families) == {"quiet", "remote_work"}
    assert decision.factual_state == FactualState.VIOLATED
    assert decision.decision == ConstraintDecisionValue.NO


def test_decide_constraint_multi_family_combination_unresolved_when_not_all_satisfied():
    """
    quiet SATISFIED + remote_work UNRESOLVED (no essential contradiction,
    just unconfirmed reliability alongside desk+wifi being SUPPORT would
    actually be SATISFIED - use a genuinely unresolved remote_work case).
    """
    c = constraint("quiet and good for remote work", ConstraintPriority.NICE)
    ev = evidence(
        claim("Q1", SUP), claim("Q2", NEE), claim("Q3", NEE),  # quiet -> SATISFIED
        claim("RW1", SUP), claim("RW2A", NEE), claim("RW2B", NEE),  # remote_work -> UNRESOLVED
    )
    decision = decide_constraint(c, ev)
    assert decision.factual_state == FactualState.UNRESOLVED
    assert decision.decision == ConstraintDecisionValue.UNCERTAIN


def test_decide_constraint_evidence_refs_populated_for_decisive_claims():
    c = constraint("no nightlife nearby please", ConstraintPriority.FORBIDDEN)
    ev = evidence(
        claim("NL1", CON, evidence_items=[resolved_item("Loud bars right outside", EvidenceRelation.CONTRADICT)]),
    )
    decision = decide_constraint(c, ev)
    assert decision.decisive_claim_ids == ["NL1"]
    assert len(decision.evidence) == 1
    assert decision.evidence[0].evidence_text == "Loud bars right outside"
    assert decision.evidence[0].relation == EvidenceRelation.CONTRADICT


def test_decide_constraint_technical_notes_preserved():
    c = constraint("quiet please", ConstraintPriority.MUST)
    ev = evidence(
        claim("Q1", NEE),
        claim("Q2", NEE, retrieval_status=RetrievalStatus.FAILED, retrieval_errors=["timeout"]),
        claim("Q3", NEE),
    )
    decision = decide_constraint(c, ev)
    assert any("Q2" in note and "retrieval failed" in note for note in decision.technical_notes)
    assert decision.factual_state == FactualState.UNRESOLVED


# ==================================================================
# adapter
# ==================================================================


def test_adapter_produces_valid_constraint_resolution_item():
    from app.schemas.search_response import ConstraintResolutionItem

    c = constraint("quiet please", ConstraintPriority.MUST, constraint_id="abc")
    ev = evidence(claim("Q1", SUP), claim("Q2", SUP), claim("Q3", NEE))
    decision = decide_constraint(c, ev)

    raw = soft_constraint_decision_to_resolution_dict(decision, listing_id="l1", listing_title="Hotel A")
    item = ConstraintResolutionItem.model_validate(raw)

    assert item.constraint_id == "abc"
    assert item.decision == "YES"
    assert item.resolution_status == "matched"
    assert item.source_stage == "soft_evidence"
    assert item.listing_id == "l1"
    assert item.explicit_negative is False
    assert raw["priority"] == "must"


@pytest.mark.parametrize("decision_value,expected_status", [
    (ConstraintDecisionValue.YES, "matched"),
    (ConstraintDecisionValue.NO, "failed"),
    (ConstraintDecisionValue.UNCERTAIN, "uncertain"),
])
def test_adapter_resolution_status_mapping(decision_value, expected_status):
    from app.schemas.soft_constraint_decision import SoftConstraintDecision

    decision = SoftConstraintDecision(
        constraint_id="c1", raw_text="x", normalized_text="x", priority=ConstraintPriority.NICE,
        families=["nightlife"], preference_direction=PreferenceDirection.DESIRED,
        factual_state=FactualState.UNRESOLVED, decision=decision_value, reason="r",
    )
    raw = soft_constraint_decision_to_resolution_dict(decision, listing_id=None, listing_title=None)
    assert raw["resolution_status"] == expected_status


def test_adapter_explicit_negative_true_only_when_violated():
    from app.schemas.soft_constraint_decision import SoftConstraintDecision

    violated = SoftConstraintDecision(
        constraint_id="c1", raw_text="x", normalized_text="x", priority=ConstraintPriority.MUST,
        families=["nightlife"], preference_direction=PreferenceDirection.FORBIDDEN,
        factual_state=FactualState.VIOLATED, decision=ConstraintDecisionValue.YES, reason="r",
    )
    satisfied = SoftConstraintDecision(
        constraint_id="c2", raw_text="x", normalized_text="x", priority=ConstraintPriority.MUST,
        families=["nightlife"], preference_direction=PreferenceDirection.FORBIDDEN,
        factual_state=FactualState.SATISFIED, decision=ConstraintDecisionValue.NO, reason="r",
    )
    assert soft_constraint_decision_to_resolution_dict(violated, listing_id=None, listing_title=None)["explicit_negative"] is True
    assert soft_constraint_decision_to_resolution_dict(satisfied, listing_id=None, listing_title=None)["explicit_negative"] is False
