"""
Audit: does app.logic.soft_constraint_decision_policy.apply_preference_direction
emit a `decision` value that means the same thing the EXISTING downstream
contract (app.logic.listing_evaluation._fails_constraint_resolution /
_apply_constraint_resolution_scoring) already expects for FORBIDDEN
constraints?

Ground truth for what downstream expects, established from the EXISTING,
already-passing legacy-fallback test
tests/test_listing_evaluation.py::test_forbidden_resolved_yes_via_llm_fallback_excludes_listing
and ::test_forbidden_resolved_no_via_llm_fallback_keeps_listing: for a
non-inverted-polarity FORBIDDEN constraint, decision="YES" means "the
underlying (positively-phrased) proposition IS confirmed" (a violation -
reject), and decision="NO" means it is confirmed absent (safe).

This file tests the FULL real path - decide_constraint -> adapter ->
the actual (unmocked) _fails_constraint_resolution and
_apply_constraint_resolution_scoring functions imported directly from
app.logic.listing_evaluation - not just apply_preference_direction in
isolation.
"""

from __future__ import annotations

import pytest

from app.logic import listing_evaluation
from app.logic.soft_constraint_decision_adapter import soft_constraint_decision_to_resolution_dict
from app.logic.soft_constraint_decision_policy import decide_constraint
from app.schemas.constraints import ConstraintPriority, UserConstraint
from app.schemas.soft_evidence import AtomicClaimResult, ClaimRelation, SoftPreferenceEvidence
from app.logic.soft_preference_decomposition import CLAIM_HYPOTHESES

SUP, CON, NEE, MIX = (
    ClaimRelation.SUPPORT,
    ClaimRelation.CONTRADICT,
    ClaimRelation.NOT_ENOUGH_EVIDENCE,
    ClaimRelation.MIXED,
)


def _nl1_evidence(relation: ClaimRelation) -> SoftPreferenceEvidence:
    return SoftPreferenceEvidence(
        claims=[AtomicClaimResult(claim_id="NL1", hypothesis=CLAIM_HYPOTHESES["NL1"], relation=relation)],
        semantic_verifier=None,
    )


def _no_nightlife_constraint() -> UserConstraint:
    return UserConstraint(
        id="c1", raw_text="no nightlife nearby", normalized_text="nightlife nearby",
        priority=ConstraintPriority.FORBIDDEN,
    )


def _resolution_dict(relation: ClaimRelation) -> dict:
    decision = decide_constraint(_no_nightlife_constraint(), _nl1_evidence(relation))
    assert decision is not None
    return soft_constraint_decision_to_resolution_dict(decision, listing_id="l1", listing_title="Hotel")


# ==================================================================
# Ground truth: what YES/NO mean downstream for the legacy fallback,
# already established/passing in tests/test_listing_evaluation.py -
# reproduced here narrowly, at the _fails_constraint_resolution /
# _apply_constraint_resolution_scoring boundary directly (no LLM, no
# evaluate_listings), to make the contract explicit.
# ==================================================================


def test_ground_truth_forbidden_yes_is_a_violation_decision():
    """Non-inverted FORBIDDEN + decision=YES -> _fails_constraint_resolution returns True."""
    results = [{"priority": "forbidden", "decision": "YES", "mapped_fields": []}]
    assert listing_evaluation._fails_constraint_resolution(results) is True


def test_ground_truth_forbidden_no_is_the_safe_decision():
    results = [{"priority": "forbidden", "decision": "NO", "mapped_fields": []}]
    assert listing_evaluation._fails_constraint_resolution(results) is False


def test_ground_truth_forbidden_scoring_rewards_no_penalizes_yes():
    item_yes = {"constraint_resolution_results": [
        {"priority": "forbidden", "decision": "YES", "normalized_text": "nightlife nearby"}
    ], "score": 0.0, "why": []}
    item_no = {"constraint_resolution_results": [
        {"priority": "forbidden", "decision": "NO", "normalized_text": "nightlife nearby"}
    ], "score": 0.0, "why": []}
    scored = listing_evaluation._apply_constraint_resolution_scoring([item_yes, item_no])
    by_score = {round(i["score"], 1) for i in scored}
    assert by_score == {-100.0, 3.0}


# ==================================================================
# REQUIRED TEST MATRIX (A-D): SoftConstraintDecision -> adapter ->
# the REAL _fails_constraint_resolution / _apply_constraint_resolution_scoring
# ==================================================================


def test_A_nl1_support_forbidden_nightlife_must_be_rejected():
    """
    NL1 SUPPORT = nightlife exists. User said "no nightlife nearby"
    (FORBIDDEN). The listing MUST be excluded.
    """
    raw = _resolution_dict(SUP)
    assert listing_evaluation._fails_constraint_resolution([raw]) is True

    item = {"constraint_resolution_results": [raw], "score": 0.0, "why": []}
    scored = listing_evaluation._apply_constraint_resolution_scoring([item])[0]
    assert scored["score"] == pytest.approx(-100.0)


def test_B_nl1_contradict_forbidden_nightlife_listing_remains_eligible():
    """NL1 CONTRADICT = evidence supports absence of nightlife. Listing stays eligible and is rewarded as safe."""
    raw = _resolution_dict(CON)
    assert listing_evaluation._fails_constraint_resolution([raw]) is False

    item = {"constraint_resolution_results": [raw], "score": 0.0, "why": []}
    scored = listing_evaluation._apply_constraint_resolution_scoring([item])[0]
    assert scored["score"] == pytest.approx(3.0)


def test_C_nl1_not_enough_evidence_is_neutral_not_falsely_safe():
    """NL1 NEE -> listing remains uncertain/neutral: not excluded, but also not rewarded as known-safe."""
    raw = _resolution_dict(NEE)
    assert listing_evaluation._fails_constraint_resolution([raw]) is False

    item = {"constraint_resolution_results": [raw], "score": 0.0, "why": []}
    scored = listing_evaluation._apply_constraint_resolution_scoring([item])[0]
    assert scored["score"] == pytest.approx(0.0)  # neither -100 nor +3


def test_D_nl1_mixed_is_neutral_not_falsely_safe():
    """NL1 MIXED -> same neutral treatment as NEE."""
    raw = _resolution_dict(MIX)
    assert listing_evaluation._fails_constraint_resolution([raw]) is False

    item = {"constraint_resolution_results": [raw], "score": 0.0, "why": []}
    scored = listing_evaluation._apply_constraint_resolution_scoring([item])[0]
    assert scored["score"] == pytest.approx(0.0)


# ==================================================================
# Full end-to-end path through evaluate_listings (authoritative mode)
# ==================================================================


async def test_full_path_forbidden_nightlife_support_excludes_listing_end_to_end(monkeypatch):
    from datetime import date
    from app.schemas.listing import ListingRaw
    from app.schemas.query import SearchRequest
    from app.schemas.soft_evidence_pipeline_policy import SoftEvidenceIntegrationMode, SoftEvidencePipelinePolicy

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_nl1_evidence(SUP) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = SearchRequest.model_validate({
        "city": "Baku", "check_in": date(2026, 4, 8), "check_out": date(2026, 4, 15),
        "constraints": [{
            "raw_text": "no nightlife nearby", "normalized_text": "nightlife nearby",
            "priority": "forbidden", "category": "other", "mapping_status": "unresolved",
            "mapped_fields": [], "evidence_strategy": "textual",
        }],
    })
    listing = ListingRaw(id="l1", name="Party Hotel")

    result = await listing_evaluation.evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(
            enabled=True, integration_mode=SoftEvidenceIntegrationMode.AUTHORITATIVE_FOR_SUPPORTED,
        ),
    )

    assert result.ranked_items == []  # MUST be rejected


async def test_full_path_forbidden_nightlife_contradict_keeps_listing_end_to_end(monkeypatch):
    from datetime import date
    from app.schemas.listing import ListingRaw
    from app.schemas.query import SearchRequest
    from app.schemas.soft_evidence_pipeline_policy import SoftEvidenceIntegrationMode, SoftEvidencePipelinePolicy

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_nl1_evidence(CON) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = SearchRequest.model_validate({
        "city": "Baku", "check_in": date(2026, 4, 8), "check_out": date(2026, 4, 15),
        "constraints": [{
            "raw_text": "no nightlife nearby", "normalized_text": "nightlife nearby",
            "priority": "forbidden", "category": "other", "mapping_status": "unresolved",
            "mapped_fields": [], "evidence_strategy": "textual",
        }],
    })
    listing = ListingRaw(id="l1", name="Quiet Hotel")

    result = await listing_evaluation.evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(
            enabled=True, integration_mode=SoftEvidenceIntegrationMode.AUTHORITATIVE_FOR_SUPPORTED,
        ),
    )

    assert len(result.ranked_items) == 1
    assert result.ranked_items[0]["score"] == pytest.approx(3.0)
