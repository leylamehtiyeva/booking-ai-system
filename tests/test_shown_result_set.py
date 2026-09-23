from __future__ import annotations

from datetime import date

import pytest

from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
from app.schemas.shown_result_set import ShownResultSet
from app.tools import orchestrate_search_tool
from app.tools.orchestrate_search_tool import orchestrate_search_request


def make_request(**overrides) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(2026, 4, 8),
        "check_out": date(2026, 4, 15),
        "constraints": [],
    }
    data.update(overrides)
    return SearchRequest.model_validate(data)


def make_listing(listing_id: str, name: str, **overrides) -> ListingRaw:
    data = {
        "id": listing_id,
        "name": name,
        "city": "Baku",
        "url": f"https://example.com/{listing_id}",
        "description": f"Description for {name}.",
        "facilities": [{"name": "Kitchen"}],
        "policies": [
            {"title": "Smoking", "content": "Smoking is not allowed."}
        ],
        "rooms": [
            {
                "name": f"{name} Room",
                "facilities": ["Private bathroom"],
            }
        ],
    }
    data.update(overrides)
    return ListingRaw.model_validate(data)


@pytest.mark.asyncio
async def test_shown_result_set_preserves_display_order(monkeypatch):
    listings = [
        make_listing("A", "Hotel A"),
        make_listing("B", "Hotel B"),
        make_listing("C", "Hotel C"),
    ]

    async def fake_get_candidates(req, max_items, source, trace=None):
        return listings

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates
    )

    _, snapshot = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        result_limit=3,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert [item.result_id for item in snapshot.items] == ["A", "B", "C"]
    assert snapshot.items[1].result_id == "B"
    assert snapshot.items[1].listing.name == "Hotel B"


@pytest.mark.asyncio
async def test_shown_result_set_carries_full_listing_raw_through_serialization(
    monkeypatch,
):
    listings = [make_listing("A", "Hotel A")]

    async def fake_get_candidates(req, max_items, source, trace=None):
        return listings

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates
    )

    _, snapshot = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    listing = snapshot.items[0].listing
    assert listing.facilities
    assert listing.rooms
    assert listing.rooms[0].facilities
    assert listing.description
    assert listing.policies

    # Round-trip through the same JSON contract used to store it in
    # session state (model_dump(mode="json") -> ... -> model_validate).
    restored = ShownResultSet.model_validate(
        snapshot.model_dump(mode="json", exclude_none=True)
    )
    restored_listing = restored.items[0].listing
    assert restored_listing.facilities
    assert restored_listing.rooms[0].facilities
    assert restored_listing.description
    assert restored_listing.policies


@pytest.mark.asyncio
async def test_shown_result_set_matches_normalized_response_result_ids(
    monkeypatch,
):
    listings = [make_listing("A", "Hotel A"), make_listing("B", "Hotel B")]

    async def fake_get_candidates(req, max_items, source, trace=None):
        return listings

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates
    )

    normalized, snapshot = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        result_limit=2,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert [r.result_id for r in normalized.results] == [
        item.result_id for item in snapshot.items
    ]


@pytest.mark.asyncio
async def test_new_results_search_replaces_snapshot_not_merges(monkeypatch):
    async def fake_get_candidates_1(req, max_items, source, trace=None):
        return [make_listing("A", "Hotel A"), make_listing("B", "Hotel B")]

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates_1
    )
    _, snapshot_1 = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    async def fake_get_candidates_2(req, max_items, source, trace=None):
        return [make_listing("C", "Hotel C"), make_listing("D", "Hotel D")]

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates_2
    )
    _, snapshot_2 = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert snapshot_1.result_set_id != snapshot_2.result_set_id
    assert [item.result_id for item in snapshot_2.items] == ["C", "D"]
    assert "A" not in [item.result_id for item in snapshot_2.items]
    assert "B" not in [item.result_id for item in snapshot_2.items]


@pytest.mark.asyncio
async def test_no_results_returns_valid_empty_snapshot():
    normalized, snapshot = await orchestrate_search_request(
        make_request(city="NoSuchCity"),
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert normalized.status.value == "no_results"
    assert isinstance(snapshot, ShownResultSet)
    assert snapshot.result_set_id
    assert snapshot.items == []


@pytest.mark.asyncio
async def test_normalization_drops_raw_data_but_snapshot_keeps_it(
    monkeypatch,
):
    listings = [make_listing("A", "Hotel A")]

    async def fake_get_candidates(req, max_items, source, trace=None):
        return listings

    monkeypatch.setattr(
        orchestrate_search_tool, "get_candidates", fake_get_candidates
    )

    normalized, snapshot = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=10,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    normalized_result = normalized.results[0]
    assert not hasattr(normalized_result, "facilities")
    assert not hasattr(normalized_result, "rooms")
    assert not hasattr(normalized_result, "description")

    shown_listing = snapshot.items[0].listing
    assert shown_listing.facilities
    assert shown_listing.rooms
    assert shown_listing.description
