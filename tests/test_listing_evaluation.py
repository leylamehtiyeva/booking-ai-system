from datetime import date

import pytest

from app.logic import listing_evaluation
from app.logic.constraint_evidence_resolution import ConstraintResolutionResult
from app.logic.listing_evaluation import evaluate_listings
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


def make_request(**overrides) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(2026, 4, 8),
        "check_out": date(2026, 4, 15),
        "constraints": [],
    }
    data.update(overrides)
    return SearchRequest.model_validate(data)


def forbidden_smoking_constraint(mapping_status: str = "known") -> dict:
    return {
        "raw_text": "no smoking",
        "normalized_text": "no smoking",
        "priority": "forbidden",
        "category": "amenity",
        "mapping_status": mapping_status,
        "mapped_fields": ["smoking_allowed"] if mapping_status == "known" else [],
        "evidence_strategy": "structured" if mapping_status == "known" else "textual",
    }


def forbidden_non_smoking_constraint() -> dict:
    return {
        "raw_text": "no smoking",
        "normalized_text": "non-smoking",
        "priority": "forbidden",
        "category": "policy",
        "mapping_status": "known",
        "mapped_fields": ["non_smoking"],
        "evidence_strategy": "structured",
    }


def forbidden_free_cancellation_constraint() -> dict:
    return {
        "raw_text": "no non-refundable rate",
        "normalized_text": "free cancellation",
        "priority": "forbidden",
        "category": "policy",
        "mapping_status": "known",
        "mapped_fields": ["free_cancellation"],
        "evidence_strategy": "structured",
    }


def unresolved_must_constraint(raw_text: str = "quiet street") -> dict:
    return {
        "raw_text": raw_text,
        "normalized_text": raw_text,
        "priority": "must",
        "category": "other",
        "mapping_status": "unresolved",
        "mapped_fields": [],
        "evidence_strategy": "textual",
    }


@pytest.mark.asyncio
async def test_structurally_confirmed_forbidden_violation_excludes_listing():
    req = make_request(constraints=[forbidden_smoking_constraint()])

    listing = ListingRaw(
        id="l1",
        name="Smokers Loft",
        policies=[
            {"title": "Smoking", "content": "Smoking allowed on the balcony."},
        ],
    )

    result = await evaluate_listings(req, [listing])

    assert result.ranked_items == []


@pytest.mark.asyncio
async def test_forbidden_resolved_yes_via_llm_fallback_excludes_listing(monkeypatch):
    req = make_request(
        constraints=[forbidden_smoking_constraint(mapping_status="unresolved")]
    )

    listing = ListingRaw(id="l2", name="Unclear Policy Apartment")

    async def fake_resolve(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return [
            ConstraintResolutionResult(
                listing_id=listing.id,
                listing_title=listing.name,
                constraint_id=constraints[0].id,
                raw_text=constraints[0].raw_text,
                normalized_text=constraints[0].normalized_text,
                resolver_type="textual",
                priority="forbidden",
                decision="YES",
                resolution_status="matched",
                confidence=0.9,
                reason="Smoking is explicitly allowed per the listing text.",
                evidence=[],
            )
        ]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", fake_resolve)

    result = await evaluate_listings(req, [listing])

    assert result.ranked_items == []


@pytest.mark.asyncio
async def test_forbidden_resolved_no_via_llm_fallback_keeps_listing(monkeypatch):
    req = make_request(
        constraints=[forbidden_smoking_constraint(mapping_status="unresolved")]
    )

    listing = ListingRaw(id="l3", name="Non-smoking Apartment")

    async def fake_resolve(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return [
            ConstraintResolutionResult(
                listing_id=listing.id,
                listing_title=listing.name,
                constraint_id=constraints[0].id,
                raw_text=constraints[0].raw_text,
                normalized_text=constraints[0].normalized_text,
                resolver_type="textual",
                priority="forbidden",
                decision="NO",
                resolution_status="failed",
                confidence=0.9,
                reason="Smoking is explicitly not allowed per the listing text.",
                evidence=[],
            )
        ]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", fake_resolve)

    result = await evaluate_listings(req, [listing])

    assert len(result.ranked_items) == 1
    assert result.ranked_items[0]["listing_id"] == "l3"


@pytest.mark.asyncio
async def test_non_smoking_field_excludes_listing_that_allows_smoking():
    """Inverted-polarity regression case: non_smoking is phrased as the SAFE
    state (YES = safe), opposite of smoking_allowed (YES = violation)."""
    req = make_request(constraints=[forbidden_non_smoking_constraint()])

    listing = ListingRaw(
        id="ns-bad",
        name="Smokers Loft",
        policies=[{"title": "Smoking", "content": "Smoking allowed on the balcony."}],
    )

    result = await evaluate_listings(req, [listing])

    assert result.ranked_items == []


@pytest.mark.asyncio
async def test_non_smoking_field_keeps_listing_that_confirms_non_smoking():
    req = make_request(constraints=[forbidden_non_smoking_constraint()])

    listing = ListingRaw(
        id="ns-good",
        name="Non-smoking Apartment",
        policies=[{"title": "Smoking", "content": "Smoking is not allowed anywhere on the property."}],
    )

    result = await evaluate_listings(req, [listing])

    assert len(result.ranked_items) == 1
    assert result.ranked_items[0]["listing_id"] == "ns-good"


@pytest.mark.asyncio
async def test_non_smoking_llm_fallback_is_actually_invoked_and_polarity_aware(monkeypatch):
    """Wiring regression: structurally-uncertain forbidden fields must reach
    the LLM fallback (previously never invoked because structured_matches_by_field
    excluded forbidden_matches), and the result must respect inverted polarity."""
    req = make_request(constraints=[forbidden_non_smoking_constraint()])

    # No smoking-related text at all -> structurally UNCERTAIN.
    listing = ListingRaw(id="ns-ambiguous", name="Unclear Policy Apartment")

    called = {"count": 0}

    async def fake_resolve(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        called["count"] += 1
        # The eligibility wiring fix must have surfaced the structured
        # UNCERTAIN value for non_smoking to the caller.
        from app.schemas.fields import Field as F
        assert structured_matches_by_field.get(F.NON_SMOKING) is not None

        return [
            ConstraintResolutionResult(
                listing_id=listing.id,
                listing_title=listing.name,
                constraint_id=constraints[0].id,
                raw_text=constraints[0].raw_text,
                normalized_text=constraints[0].normalized_text,
                resolver_type="textual",
                priority="forbidden",
                mapped_fields=["non_smoking"],
                decision="NO",  # per inverted polarity: NO = violation (smoking confirmed)
                resolution_status="failed",
                confidence=0.9,
                reason="Smoking is explicitly allowed per the listing text.",
                evidence=[],
            )
        ]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", fake_resolve)

    result = await evaluate_listings(req, [listing])

    assert called["count"] == 1
    assert result.ranked_items == []


@pytest.mark.asyncio
async def test_free_cancellation_field_excludes_non_refundable_listing():
    """Second inverted-polarity field: proves the registry generalizes."""
    req = make_request(constraints=[forbidden_free_cancellation_constraint()])

    listing = ListingRaw(
        id="fc-bad",
        name="Budget Apartment",
        policies=[{"title": "Cancellation", "content": "Non-refundable rate."}],
    )

    result = await evaluate_listings(req, [listing])

    assert result.ranked_items == []


@pytest.mark.asyncio
async def test_must_constraint_confirmed_via_llm_fallback_increases_score(monkeypatch):
    req = make_request(constraints=[unresolved_must_constraint()])

    listing = ListingRaw(id="l4", name="Quiet Street Apartment")

    async def fake_resolve(*, listing, constraints, structured_matches_by_field, policy, trace=None):
        return [
            ConstraintResolutionResult(
                listing_id=listing.id,
                listing_title=listing.name,
                constraint_id=constraints[0].id,
                raw_text=constraints[0].raw_text,
                normalized_text=constraints[0].normalized_text,
                resolver_type="textual",
                priority="must",
                decision="YES",
                resolution_status="matched",
                confidence=0.9,
                reason="Quiet street is explicitly confirmed in the listing text.",
                evidence=[],
            )
        ]

    monkeypatch.setattr(listing_evaluation, "resolve_listing_constraints_with_fallback", fake_resolve)

    result = await evaluate_listings(req, [listing])

    assert len(result.ranked_items) == 1
    assert result.ranked_items[0]["score"] == 3.0
