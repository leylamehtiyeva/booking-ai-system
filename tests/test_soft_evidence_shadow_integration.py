from __future__ import annotations

from datetime import date

import pytest

from app.logic import listing_evaluation
from app.logic.listing_evaluation import evaluate_listings
from app.observability.trace import RequestTrace
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimRelation,
    SoftPreferenceEvidence,
)
from app.schemas.soft_evidence_pipeline_policy import SoftEvidencePipelinePolicy


def make_request(**overrides) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(2026, 4, 8),
        "check_out": date(2026, 4, 15),
        "constraints": [],
    }
    data.update(overrides)
    return SearchRequest.model_validate(data)


def quiet_nice_constraint() -> dict:
    return {
        "raw_text": "quiet please",
        "normalized_text": "quiet please",
        "priority": "nice",
        "category": "other",
        "mapping_status": "unresolved",
        "mapped_fields": [],
        "evidence_strategy": "textual",
    }


def kitchen_constraint() -> dict:
    """Maps to no Phase B family at all."""
    return {
        "raw_text": "has a kitchen",
        "normalized_text": "has a kitchen",
        "priority": "nice",
        "category": "amenity",
        "mapping_status": "unresolved",
        "mapped_fields": [],
        "evidence_strategy": "textual",
    }


async def _never_call_orchestration(*args, **kwargs):
    raise AssertionError("build_shadow_soft_preference_evidence must not be called")


def _fake_evidence(claim_id: str) -> SoftPreferenceEvidence:
    return SoftPreferenceEvidence(
        claims=[
            AtomicClaimResult.from_evidence_items(
                claim_id=claim_id, hypothesis="h", evidence_items=[],
            )
        ],
        semantic_verifier=None,
    )


# ---------------- default off ----------------


async def test_disabled_by_default_no_calls_and_none_everywhere(monkeypatch):
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)

    req = make_request(constraints=[quiet_nice_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Apartment")

    result = await evaluate_listings(req, [listing])

    assert len(result.ranked_items) == 1
    assert result.ranked_items[0]["soft_preference_evidence"] is None


async def test_unsupported_constraints_produce_none_without_calling_orchestration(monkeypatch):
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)

    req = make_request(constraints=[kitchen_constraint()])
    listing = ListingRaw(id="l1", name="Apartment")

    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),
    )

    assert result.ranked_items[0]["soft_preference_evidence"] is None


# ---------------- enabled, mocked orchestration ----------------


async def test_enabled_attaches_soft_preference_evidence_for_shadow_scope(monkeypatch):
    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence("Q1") for _ in listings]

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_nice_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Apartment")

    trace = RequestTrace()
    result = await evaluate_listings(
        req, [listing],
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),
        trace=trace,
    )

    evidence = result.ranked_items[0]["soft_preference_evidence"]
    assert evidence is not None
    assert evidence.claims[0].claim_id == "Q1"

    assert trace.soft_evidence_shadow_detail is not None
    assert len(trace.soft_evidence_shadow_detail) == 1
    assert trace.soft_evidence_shadow_detail[0]["listing_id"] == "l1"
    assert "old_constraint_resolution_results" in trace.soft_evidence_shadow_detail[0]
    assert trace.soft_evidence_claim_assignments
    assert trace.soft_evidence_claim_assignments[0]["claim_id"] == "Q1"


async def test_shadow_scope_bounded_to_top_k(monkeypatch):
    seen_listing_counts = []

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        seen_listing_counts.append(len(listings))
        return [_fake_evidence("Q1") for _ in listings]

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)

    req = make_request(constraints=[quiet_nice_constraint()])
    listings = [ListingRaw(id=f"l{i}", name=f"Apartment {i}") for i in range(7)]

    result = await evaluate_listings(
        req, listings,
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True, shadow_hotel_top_k=3),
    )

    assert seen_listing_counts == [3]
    populated = [item for item in result.ranked_items if item["soft_preference_evidence"] is not None]
    none_items = [item for item in result.ranked_items if item["soft_preference_evidence"] is None]
    assert len(populated) == 3
    assert len(none_items) == 4


# ---------------- regression: old path unaffected ----------------


async def _run_and_strip_soft_evidence(req, listings, **kwargs):
    result = await evaluate_listings(req, listings, **kwargs)
    stripped = []
    for item in result.ranked_items:
        item_copy = dict(item)
        item_copy.pop("soft_preference_evidence", None)
        stripped.append(item_copy)
    return stripped, result.debug_notes


async def test_old_scoring_and_constraint_resolution_unchanged_with_shadow_enabled(monkeypatch):
    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence("Q1") for _ in listings]

    req = make_request(constraints=[quiet_nice_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Apartment", description="A quiet street.")

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)
    disabled_items, disabled_notes = await _run_and_strip_soft_evidence(req, [listing])

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)
    enabled_items, enabled_notes = await _run_and_strip_soft_evidence(
        req, [listing], soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),
    )

    assert len(disabled_items) == len(enabled_items) == 1
    disabled_item, enabled_item = disabled_items[0], enabled_items[0]

    assert disabled_item["score"] == enabled_item["score"]
    assert disabled_item["constraint_resolution_results"] == enabled_item["constraint_resolution_results"]
    assert disabled_item["why"] == enabled_item["why"]
    assert disabled_item["matched_must_count"] == enabled_item["matched_must_count"]
    assert disabled_notes == enabled_notes


async def test_final_selection_and_normalization_unchanged_with_shadow_enabled(monkeypatch):
    """
    End-to-end through select_ranked_items + normalize_search_response -
    the actual public response shape - is identical whether the shadow
    layer is enabled or not.
    """
    from app.logic.normalize_search_response import normalize_search_response
    from app.logic.result_selection import select_ranked_items

    async def _fake_build(*, listings, claim_assignments, pipeline_policy, verifier_policy, trace):
        return [_fake_evidence("Q1") for _ in listings]

    req = make_request(constraints=[quiet_nice_constraint()])
    listing = ListingRaw(id="l1", name="Quiet Apartment", url="https://example.test/l1")

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)
    disabled_result = await evaluate_listings(req, [listing])
    disabled_selected = select_ranked_items(disabled_result.ranked_items, top_n=5)
    disabled_response = normalize_search_response(req, disabled_selected, top_n=5, dropped_requests=[])

    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _fake_build)
    enabled_result = await evaluate_listings(
        req, [listing], soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),
    )
    enabled_selected = select_ranked_items(enabled_result.ranked_items, top_n=5)
    enabled_response = normalize_search_response(req, enabled_selected, top_n=5, dropped_requests=[])

    assert disabled_response.model_dump(mode="json") == enabled_response.model_dump(mode="json")


async def test_no_listings_short_circuits_before_shadow_layer(monkeypatch):
    monkeypatch.setattr(listing_evaluation, "build_shadow_soft_preference_evidence", _never_call_orchestration)
    req = make_request(constraints=[quiet_nice_constraint()])
    result = await evaluate_listings(
        req, [], soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(enabled=True),
    )
    assert result.ranked_items == []
