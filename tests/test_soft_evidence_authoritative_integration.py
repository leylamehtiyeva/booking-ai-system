from __future__ import annotations

from datetime import date

import pytest

from app.logic import listing_evaluation
from app.logic.listing_evaluation import evaluate_listings
from app.logic.soft_constraint_decision_policy import decide_constraint
from app.observability.trace import RequestTrace
from app.schemas.constraints import ConstraintPriority, UserConstraint
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_constraint_decision import ConstraintDecisionValue, FactualState
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimRelation,
    SoftPreferenceEvidence,
)
from app.schemas.soft_evidence_pipeline_policy import SoftEvidenceIntegrationMode, SoftEvidencePipelinePolicy
from app.logic.soft_preference_decomposition import CLAIM_HYPOTHESES

SUP, CON, NEE, MIX = (
    ClaimRelation.SUPPORT,
    ClaimRelation.CONTRADICT,
    ClaimRelation.NOT_ENOUGH_EVIDENCE,
    ClaimRelation.MIXED,
)


def make_request(**overrides) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(2026, 4, 8),
        "check_out": date(2026, 4, 15),
        "constraints": [],
    }
    data.update(overrides)
    return SearchRequest.model_validate(data)


def quiet_must_constraint(constraint_id: str = "quiet-1") -> dict:
    return {
        "id": constraint_id,
        "raw_text": "must be quiet",
        "normalized_text": "must be quiet",
        "priority": "must",
        "category": "other",
        "mapping_status": "unresolved",
        "mapped_fields": [],
        "evidence_strategy": "textual",
    }


def kitchen_constraint(constraint_id: str = "kitchen-1") -> dict:
    """Maps to no Phase B family - always routed to legacy fallback."""
    return {
        "id": constraint_id,
        "raw_text": "has a kitchen",
        "normalized_text": "has a kitchen",
        "priority": "must",
        "category": "amenity",
        "mapping_status": "unresolved",
        "mapped_fields": [],
        "evidence_strategy": "textual",
    }


def _claim(claim_id: str, relation: ClaimRelation) -> AtomicClaimResult:
    return AtomicClaimResult(claim_id=claim_id, hypothesis=CLAIM_HYPOTHESES[claim_id], relation=relation)


def _fake_evidence(*relations: tuple[str, ClaimRelation]) -> SoftPreferenceEvidence:
    return SoftPreferenceEvidence(
        claims=[_claim(cid, rel) for cid, rel in relations],
        semantic_verifier=None,
    )


def authoritative_policy(**overrides) -> SoftEvidencePipelinePolicy:
    data = {"enabled": True, "integration_mode": SoftEvidenceIntegrationMode.AUTHORITATIVE_FOR_SUPPORTED}
    data.update(overrides)
    return SoftEvidencePipelinePolicy(**data)


# ==================================================================
# A. OFF / default
# ==================================================================


async def test_off_default_behavior_unchanged(monkeypatch):
    fallback_calls: list[list[str]] = []

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        fallback_calls.append([c.normalized_text for c in constraints])
        return []

    async def _never_call_orchestration(*args, **kwargs):
        raise AssertionError("soft-evidence pipeline must not run OFF by default")

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Hotel A")

    result = await evaluate_listings(req, [listing])

    assert result.ranked_items[0]["soft_preference_evidence"] is None
    assert fallback_calls == [["must be quiet"]]


# ==================================================================
# B. SHADOW
# ==================================================================


async def test_shadow_mode_legacy_stays_authoritative(monkeypatch):
    fallback_calls: list[list[str]] = []

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        fallback_calls.append([c.normalized_text for c in constraints])
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence(("Q1", SUP), ("Q2", SUP), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Hotel A")

    trace = RequestTrace()
    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),  # default mode: SHADOW
        trace=trace,
    )

    item = result.ranked_items[0]
    # legacy fallback was still called with the SAME constraint - never skipped in SHADOW
    assert fallback_calls == [["must be quiet"]]
    # soft evidence populated for telemetry only
    assert item["soft_preference_evidence"] is not None
    # constraint_resolution_results has no soft_evidence-sourced entry
    assert all(r.get("source_stage") != "soft_evidence" for r in item["constraint_resolution_results"])
    assert trace.soft_evidence_summary is not None
    assert trace.soft_evidence_summary["integration_mode"] == "shadow"


# ==================================================================
# C. AUTHORITATIVE_FOR_SUPPORTED
# ==================================================================


async def test_authoritative_supported_constraint_uses_new_decision_and_skips_legacy(monkeypatch):
    fallback_calls: list[list[str]] = []

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        fallback_calls.append([c.normalized_text for c in constraints])
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence(("Q1", SUP), ("Q2", SUP), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Hotel A")

    trace = RequestTrace()
    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
        trace=trace,
    )

    item = result.ranked_items[0]

    # legacy fallback received NO call for the supported constraint at all
    assert fallback_calls == [[]]

    soft_entries = [r for r in item["constraint_resolution_results"] if r.get("source_stage") == "soft_evidence"]
    assert len(soft_entries) == 1
    assert soft_entries[0]["decision"] == "YES"
    assert soft_entries[0]["normalized_text"] == "must be quiet"

    assert item["soft_preference_evidence"] is not None
    assert trace.soft_evidence_summary["integration_mode"] == "authoritative_for_supported"


async def test_authoritative_unsupported_constraint_still_uses_legacy_fallback(monkeypatch):
    fallback_calls: list[list[str]] = []

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        fallback_calls.append([c.normalized_text for c in constraints])
        return []

    async def _never_call_orchestration(*args, **kwargs):
        raise AssertionError("no supported family -> pipeline must not run at all")

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)

    req = make_request(constraints=[kitchen_constraint()])
    listing = ListingRaw(id="l1", name="Hotel A")

    # families_for_constraint("has a kitchen") == [] -> decompose_constraints
    # produces no assignments -> the pipeline never actually runs, so it's
    # safe to leave build_shadow_soft_preference_evidence un-mocked here
    # too, but assert-via-raise makes the "never called" guarantee explicit.
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
    )

    item = result.ranked_items[0]
    assert fallback_calls == [["has a kitchen"]]
    assert item["soft_preference_evidence"] is None
    # The mocked fallback returned no resolution for the unresolved MUST
    # constraint, so coverage normalization adds a synthetic UNCERTAIN
    # placeholder - it never gets a soft_evidence-sourced entry.
    assert all(r.get("source_stage") != "soft_evidence" for r in item["constraint_resolution_results"])


# ==================================================================
# D. Mixed request: one supported + one unsupported
# ==================================================================


async def test_mixed_request_routes_each_constraint_to_the_correct_path(monkeypatch):
    fallback_calls: list[list[str]] = []

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        fallback_calls.append([c.normalized_text for c in constraints])
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence(("Q1", SUP), ("Q2", SUP), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint(), kitchen_constraint()])
    listing = ListingRaw(id="l1", name="Hotel A")

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
    )

    item = result.ranked_items[0]

    # legacy fallback only ever saw the unsupported constraint
    assert fallback_calls == [["has a kitchen"]]

    soft_entries = [r for r in item["constraint_resolution_results"] if r.get("source_stage") == "soft_evidence"]
    assert [e["normalized_text"] for e in soft_entries] == ["must be quiet"]

    # exactly one resolution per constraint - no duplicate resolution
    all_texts = [r["normalized_text"] for r in item["constraint_resolution_results"]]
    assert sorted(all_texts) == ["has a kitchen", "must be quiet"]
    assert len(all_texts) == len(set(all_texts))


# ==================================================================
# Downstream effect: SoftPreferenceEvidence -> SoftConstraintDecision ->
# constraint_resolution_results -> existing deterministic eligibility
# ==================================================================


async def test_authoritative_no_decision_makes_must_constraint_reject_listing(monkeypatch):
    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        # Q2 CONTRADICT -> quiet VIOLATED -> MUST constraint -> NO -> reject
        return [_fake_evidence(("Q1", SUP), ("Q2", CON), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Loud Hotel")

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
    )

    assert result.ranked_items == []


async def test_authoritative_yes_decision_keeps_listing_and_scores_it(monkeypatch):
    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence(("Q1", SUP), ("Q2", SUP), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Hotel")

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
    )

    assert len(result.ranked_items) == 1
    # a MUST constraint resolved YES adds +3.0 via _apply_constraint_resolution_scoring,
    # exactly like a legacy fallback YES would have.
    assert result.ranked_items[0]["score"] == pytest.approx(3.0)
    assert any("CONSTRAINT_MATCH" in w for w in result.ranked_items[0]["why"])


async def test_authoritative_uncertain_decision_does_not_reject_or_score(monkeypatch):
    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        # Nizami-style: everything NEE -> UNRESOLVED -> MUST -> UNCERTAIN
        return [_fake_evidence(("Q1", NEE), ("Q2", NEE), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Unknown Hotel")

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=authoritative_policy(),
    )

    assert len(result.ranked_items) == 1  # UNCERTAIN never rejects
    assert result.ranked_items[0]["score"] == pytest.approx(0.0)  # UNCERTAIN never scores


async def test_final_response_schema_unchanged_in_authoritative_mode(monkeypatch):
    """Public NormalizedSearchResponse schema/shape must remain unchanged."""
    from app.logic.normalize_search_response import normalize_search_response
    from app.logic.result_selection import select_ranked_items

    async def _fake_fallback(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence(("Q1", SUP), ("Q2", SUP), ("Q3", NEE)) for _ in listings]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", _fake_fallback)
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_must_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Hotel", url="https://example.test/l1")

    result = await evaluate_listings(
        req, [listing], soft_evidence_pipeline_policy=authoritative_policy(),
    )
    selected = select_ranked_items(result.ranked_items, top_n=5)
    response = normalize_search_response(req, selected, top_n=5, dropped_requests=[])

    dumped = response.model_dump(mode="json")
    assert dumped["status"] == "results"
    assert len(dumped["results"]) == 1
    matched_names = [c["name"] for c in dumped["results"][0]["matched_requested_constraints"]]
    assert "must be quiet" in matched_names


# ==================================================================
# Live-derived regression scenarios (section 9)
# ==================================================================


def _combined_decision(constraint: UserConstraint, *relations: tuple[str, ClaimRelation]) -> ConstraintDecisionValue:
    ev = SoftPreferenceEvidence(claims=[_claim(cid, rel) for cid, rel in relations], semantic_verifier=None)
    decision = decide_constraint(constraint, ev)
    assert decision is not None
    return decision.decision


def _quiet_constraint() -> UserConstraint:
    return UserConstraint(id="q", raw_text="quiet", normalized_text="quiet", priority=ConstraintPriority.NICE)


def _remote_work_constraint() -> UserConstraint:
    return UserConstraint(
        id="rw", raw_text="good for remote work", normalized_text="good for remote work",
        priority=ConstraintPriority.NICE,
    )


def test_nizami_hotel_quiet_is_uncertain():
    assert _combined_decision(_quiet_constraint(), ("Q1", NEE), ("Q2", NEE), ("Q3", NEE)) == ConstraintDecisionValue.UNCERTAIN


def test_nizami_hotel_remote_work_is_uncertain():
    """The old fallback previously returned YES largely because Wi-Fi existed - the new policy must not."""
    assert (
        _combined_decision(_remote_work_constraint(), ("RW1", SUP), ("RW2A", SUP), ("RW2B", NEE))
        == ConstraintDecisionValue.UNCERTAIN
    )


def test_metro_city_quiet_is_uncertain():
    """Soundproofing alone is not sufficient proof that the hotel is quiet."""
    assert _combined_decision(_quiet_constraint(), ("Q1", SUP), ("Q2", NEE), ("Q3", NEE)) == ConstraintDecisionValue.UNCERTAIN


def test_metro_city_remote_work_is_uncertain():
    assert (
        _combined_decision(_remote_work_constraint(), ("RW1", SUP), ("RW2A", SUP), ("RW2B", NEE))
        == ConstraintDecisionValue.UNCERTAIN
    )


def test_maestro_address_quiet_is_uncertain():
    """MIXED surroundings evidence must prevent an overconfident YES."""
    assert _combined_decision(_quiet_constraint(), ("Q1", SUP), ("Q2", MIX), ("Q3", SUP)) == ConstraintDecisionValue.UNCERTAIN


def test_maestro_address_remote_work_is_uncertain():
    assert (
        _combined_decision(_remote_work_constraint(), ("RW1", SUP), ("RW2A", SUP), ("RW2B", NEE))
        == ConstraintDecisionValue.UNCERTAIN
    )
