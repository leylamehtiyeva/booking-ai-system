from datetime import date

import pytest
from app.logic import listing_evaluation
from app.logic.constraint_evidence_resolution import ConstraintResolutionResult
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.listing import ListingRaw, Room
from app.schemas.query import SearchRequest
from app.tools import orchestrate_search_tool
from app.tools.orchestrate_search_tool import orchestrate_search_request
from app.schemas.search_response import SearchStatus


def make_request(**overrides) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(2026, 4, 8),
        "check_out": date(2026, 4, 15),
        "constraints": [],
    }
    data.update(overrides)
    return SearchRequest.model_validate(data)


def kitchen_constraint(priority: str = "must") -> dict:
    return {
        "raw_text": "kitchen",
        "normalized_text": "kitchen",
        "priority": priority,
        "category": "amenity",
        "mapping_status": "known",
        "mapped_fields": ["kitchen"],
        "evidence_strategy": "structured",
    }


def private_bathroom_constraint(priority: str = "must") -> dict:
    return {
        "raw_text": "private bathroom",
        "normalized_text": "private bathroom",
        "priority": priority,
        "category": "amenity",
        "mapping_status": "known",
        "mapped_fields": ["private_bathroom"],
        "evidence_strategy": "structured",
    }


@pytest.mark.asyncio
async def test_core_orchestrator_requires_complete_search_request():
    req = SearchRequest(
        city="Baku",
        check_in=None,
        check_out=None,
        constraints=[],
    )

    with pytest.raises(
        ValueError,
        match="city, check_in and check_out",
    ):
        await orchestrate_search_request(
            req,
            source="fixtures",
            candidate_pool_size=10,
            fallback_policy=FallbackPolicy(enabled=False),
        )


@pytest.mark.asyncio
async def test_baku_kitchen_returns_apartment():
    req = make_request(
        constraints=[kitchen_constraint()],
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert out.status == SearchStatus.RESULTS
    assert out.results[0].title == "Compact Apartment"


@pytest.mark.asyncio
async def test_tokyo_returns_no_results_on_fixtures():
    req = make_request(
        city="Tokyo",
        check_in=date(2026, 2, 12),
        check_out=date(2026, 2, 14),
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert out.status == SearchStatus.NO_RESULTS
    assert out.results == []


@pytest.mark.asyncio
async def test_candidate_pool_size_and_result_limit_have_different_responsibilities(
    monkeypatch,
):
    received = {}

    async def fake_get_candidates(
        req,
        max_items,
        source,
        trace=None,
    ):
        received["max_items"] = max_items
        return [
            ListingRaw(
                id=f"listing-{i}",
                name=f"Apartment {i}",
                city="Baku",
                url=f"https://example.com/{i}",
                description="Apartment in Baku.",
                available_dates={
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
                rooms=[],
            )
            for i in range(6)
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request()

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=6,
        result_limit=2,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert received["max_items"] == 6
    assert len(out.results) == 2


@pytest.mark.asyncio
async def test_numeric_filters_are_applied_in_orchestrator(monkeypatch):
    async def fake_get_candidates(req, max_items, source, trace=None):
        return [
            ListingRaw(
                id="small-1",
                name="One-Bedroom Apartment",
                city="Baku",
                url="https://example.com/small-1",
                description=(
                    "Apartment in Baku, 45 sqm, "
                    "private bathroom, kitchen."
                ),
                available_dates={
                    "check_in": "2026-02-01",
                    "check_out": "2026-03-31",
                },
                facilities=[
                    {"name": "Kitchen"},
                    {"name": "Private bathroom"},
                ],
                rooms=[
                    Room(
                        name="One-Bedroom Apartment",
                        facilities=[
                            "Kitchen",
                            "Private bathroom",
                        ],
                    )
                ],
            ),
            ListingRaw(
                id="big-1",
                city="Baku",
                name="Three-Bedroom Apartment",
                url="https://example.com/big-1",
                description=(
                    "Apartment in Baku, 2196 feet², "
                    "private bathroom, kitchen."
                ),
                available_dates={
                    "check_in": "2026-02-01",
                    "check_out": "2026-03-31",
                },
                facilities=[
                    {"name": "Kitchen"},
                    {"name": "Private bathroom"},
                ],
                rooms=[
                    Room(
                        name="Three-Bedroom Apartment with Balcony",
                        facilities=[
                            "Kitchen",
                            "Private bathroom",
                        ],
                    )
                ],
            ),
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request(
        check_in=date(2026, 2, 12),
        check_out=date(2026, 2, 14),
        constraints=[
            kitchen_constraint(),
            private_bathroom_constraint(),
        ],
        filters={
            "bedrooms_min": 2,
            "area_sqm_min": 80,
        },
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
    )

    assert out.status == SearchStatus.RESULTS
    assert len(out.results) == 1
    assert out.results[0].result_id == "big-1"
    assert any(
        "BEDROOMS:" in reason
        for reason in out.results[0].why
    )
    assert any(
        "AREA:" in reason
        for reason in out.results[0].why
    )


@pytest.mark.asyncio
async def test_price_filter_per_night_is_applied(monkeypatch):
    async def fake_get_candidates(req, max_items, source, trace=None):
        return [
            ListingRaw(
                id="too-expensive",
                name="Apartment STEL",
                city="Baku",
                url="https://example.com/baku-expensive",
                description="Apartment in Baku city center.",
                price=700.0,
                currency="US$",
                available_dates={
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
                rooms=[],
            ),
            ListingRaw(
                id="good-price",
                name="Budget Apartment",
                city="Baku",
                url="https://example.com/baku-budget",
                description="Budget apartment in Baku.",
                price=300.0,
                currency="US$",
                available_dates={
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
                rooms=[],
            ),
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request(
        filters={
            "price": {
                "max_amount": 50,
                "currency": "USD",
                "scope": "per_night",
            }
        }
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
    )

    assert out.status == SearchStatus.RESULTS
    assert len(out.results) == 1
    assert out.results[0].result_id == "good-price"
    assert any(
        "PRICE:" in reason
        for reason in out.results[0].why
    )


@pytest.mark.asyncio
async def test_bathroom_filter_is_applied(monkeypatch):
    async def fake_get_candidates(req, max_items, source, trace=None):
        return [
            ListingRaw(
                id="one-bathroom",
                name="Apartment in Baku",
                city="Baku",
                url="https://example.com/baku-one-bathroom",
                description="Nice apartment in Baku with 1 bathroom.",
                rooms=[],
            ),
            ListingRaw(
                id="two-bathrooms",
                name="Family apartment in Baku",
                city="Baku",
                url="https://example.com/baku-two-bathrooms",
                description="Family apartment in Baku with 2 bathrooms.",
                rooms=[],
            ),
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request(
        filters={
            "bathrooms_min": 2,
        }
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
    )

    assert out.status == SearchStatus.RESULTS
    assert len(out.results) == 1
    assert out.results[0].result_id == "two-bathrooms"
    assert any(
        "BATHROOMS:" in reason
        for reason in out.results[0].why
    )


@pytest.mark.asyncio
async def test_property_type_filter_is_applied(monkeypatch):
    async def fake_get_candidates(req, max_items, source, trace=None):
        return [
            ListingRaw(
                id="hotel-one",
                name="Hotel in Baku",
                city="Baku",
                url="https://example.com/baku-hotel",
                description="Nice hotel in Baku.",
                rooms=[],
            ),
            ListingRaw(
                id="apartment-one",
                name="Apartment in Baku",
                city="Baku",
                url="https://example.com/baku-apartment",
                description="Entire apartment in Baku with kitchen.",
                rooms=[],
            ),
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request(
        property_types=["apartment"],
        occupancy_types=[],
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
    )

    assert out.status == SearchStatus.RESULTS
    assert len(out.results) == 1
    assert out.results[0].result_id == "apartment-one"
    assert any(
        "PROPERTY_TYPE:" in reason
        for reason in out.results[0].why
    )


@pytest.mark.asyncio
async def test_orchestrator_returns_normalized_response(monkeypatch):
    async def fake_get_candidates(req, max_items, source, trace=None):
        return [
            ListingRaw(
                id=None,
                name="Apartment in Baku",
                city="Baku",
                url="https://example.com/baku-apartment",
                description=(
                    "Apartment in Baku with private kitchen "
                    "and private bathroom."
                ),
                price=300.0,
                currency="US$",
                rooms=[],
                available_dates={
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            )
        ]

    monkeypatch.setattr(
        orchestrate_search_tool,
        "get_candidates",
        fake_get_candidates,
    )

    req = make_request(
        constraints=[kitchen_constraint()],
        property_types=["apartment"],
        occupancy_types=[],
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(
            enabled=True,
            top_k=0,
        ),
    )

    assert out.status == SearchStatus.RESULTS
    assert out.request_summary is not None
    assert out.results

    assert out.request_summary.city == "Baku"
    assert (
        out.request_summary.constraints[0][
            "normalized_text"
        ]
        == "kitchen"
    )
    assert out.request_summary.property_types == [
        "apartment"
    ]

    assert len(out.results) == 1

    first = out.results[0]

    assert first.result_id
    assert first.title == "Apartment in Baku"
    assert first.url == (
        "https://example.com/baku-apartment"
    )
    assert isinstance(first.matched_constraints, list)
    assert isinstance(first.uncertain_constraints, list)
    assert isinstance(first.facts, list)


@pytest.mark.asyncio
async def test_constraint_resolution_results_are_attached(monkeypatch):
    req = make_request(
        constraints=[
            {
                "raw_text": "ironing facilities",
                "normalized_text": "iron",
                "priority": "must",
                "category": "amenity",
                "mapping_status": "known",
                "mapped_fields": ["iron"],
                "evidence_strategy": "structured",
            },
            {
                "raw_text": "satellite TV",
                "normalized_text": "satellite TV",
                "priority": "must",
                "category": "amenity",
                "mapping_status": "unresolved",
                "mapped_fields": [],
                "evidence_strategy": "textual",
            },
        ],
        property_types=["apartment"],
        occupancy_types=[],
    )

    async def fake_resolve_listing_constraints_with_fallback(
        *args,
        **kwargs,
    ):
        return [
            ConstraintResolutionResult(
                listing_id="compact-apartment",
                listing_title="Compact Apartment",
                constraint_id=None,
                raw_text="satellite TV",
                normalized_text="satellite TV",
                resolver_type="textual",
                decision="YES",
                resolution_status="matched",
                confidence=0.95,
                reason="Satellite TV is explicitly mentioned",
                evidence=[],
            )
        ]

    monkeypatch.setattr(
        listing_evaluation,
        "resolve_listing_constraints_with_fallback",
        fake_resolve_listing_constraints_with_fallback,
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=5,
        fallback_policy=FallbackPolicy(enabled=True),
    )

    assert out.status == SearchStatus.RESULTS
    assert out.results

    compact = next(
        result
        for result in out.results
        if result.title == "Compact Apartment"
    )

    assert compact.constraint_resolution_results

    first = compact.constraint_resolution_results[0]

    assert first.normalized_text == "satellite TV"
    assert first.resolution_status == "matched"
    assert (
        first.reason
        == "Satellite TV is explicitly mentioned"
    )


@pytest.mark.asyncio
async def test_occupancy_filter_is_applied():
    req = make_request(
        adults=7,
        children=0,
        rooms=1,
        constraints=[kitchen_constraint()],
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert out.status == SearchStatus.RESULTS
    assert out.results





@pytest.mark.asyncio
async def test_unresolved_must_textual_constraint_cannot_disappear_when_fallback_disabled():
    req = make_request(
        constraints=[
            kitchen_constraint(),
            {
                "raw_text": "WiFi",
                "normalized_text": "wifi",
                "priority": "must",
                "category": "amenity",
                "mapping_status": "known",
                "mapped_fields": ["wifi"],
                "evidence_strategy": "structured",
            },
            {
                "raw_text": "seaview",
                "normalized_text": "seaview",
                "priority": "must",
                "category": "location",
                "mapping_status": "unresolved",
                "mapped_fields": [],
                "evidence_strategy": "textual",
            },
        ],
        property_types=["apartment"],
        occupancy_types=[],
    )

    out = await orchestrate_search_request(
        req,
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert out.status == SearchStatus.RESULTS
    assert out.results

    first = out.results[0]

    assert first.match_tier == "partial"
    assert first.eligibility_status == "eligible"
    assert (
        "all required constraints are confirmed"
        not in first.selection_reasons
    )
    assert (
        "some requested constraints are not fully confirmed"
        in first.selection_reasons
    )

    resolution_results = first.constraint_resolution_results

    seaview = next(
        result
        for result in resolution_results
        if result.normalized_text == "seaview"
    )

    assert seaview.decision == "UNCERTAIN"
    assert seaview.resolution_status == "uncertain"
    assert (
        seaview.source_stage
        == "coverage_normalization"
    )

    uncertain_names = [
        constraint.name
        for constraint in first.uncertain_requested_constraints
    ]

    assert "seaview" in uncertain_names