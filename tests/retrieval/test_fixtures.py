import json
from datetime import date
from pathlib import Path

import pytest

from app.retrieval.fixtures import (
    FixturesRetriever,
)
from app.schemas.query import SearchRequest


def _write_fixture(
    path: Path,
    listings: list[dict],
) -> None:
    path.write_text(
        json.dumps(listings),
        encoding="utf-8",
    )


def _request(
    *,
    city: str = "Baku",
    check_in: date = date(
        2026,
        4,
        10,
    ),
    check_out: date = date(
        2026,
        4,
        15,
    ),
) -> SearchRequest:
    return SearchRequest(
        city=city,
        check_in=check_in,
        check_out=check_out,
        constraints=[],
    )


@pytest.mark.asyncio
async def test_fixtures_retriever_filters_by_city(
    tmp_path: Path,
):
    fixture_path = (
        tmp_path / "listings.json"
    )

    _write_fixture(
        fixture_path,
        [
            {
                "id": "baku",
                "name": "Baku Apartment",
                "city": "Baku",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
            {
                "id": "tokyo",
                "name": "Tokyo Apartment",
                "city": "Tokyo",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
        ],
    )

    retriever = FixturesRetriever(
        path=fixture_path
    )

    results = await retriever.get_candidates(
        _request(city="Baku"),
        max_items=10,
    )

    assert len(results) == 1
    assert results[0].id == "baku"


@pytest.mark.asyncio
async def test_fixture_city_filter_uses_city_field_only(
    tmp_path: Path,
):
    fixture_path = (
        tmp_path / "listings.json"
    )

    _write_fixture(
        fixture_path,
        [
            {
                "id": "wrong-city",
                "name": "Apartment",
                "city": "Baku",
                "description": (
                    "Beautiful apartment "
                    "in Hong Kong."
                ),
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
            {
                "id": "correct-city",
                "name": "Hong Kong Studio",
                "city": "Hong Kong",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
        ],
    )

    retriever = FixturesRetriever(
        path=fixture_path
    )

    results = await retriever.get_candidates(
        _request(city="Hong Kong"),
        max_items=10,
    )

    assert len(results) == 1
    assert (
        results[0].id
        == "correct-city"
    )


@pytest.mark.asyncio
async def test_listing_without_city_is_excluded(
    tmp_path: Path,
):
    fixture_path = (
        tmp_path / "listings.json"
    )

    _write_fixture(
        fixture_path,
        [
            {
                "id": "missing-city",
                "name": "Apartment",
                "city": None,
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
        ],
    )

    retriever = FixturesRetriever(
        path=fixture_path
    )

    results = await retriever.get_candidates(
        _request(),
        max_items=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_fixtures_retriever_filters_by_dates(
    tmp_path: Path,
):
    fixture_path = (
        tmp_path / "listings.json"
    )

    _write_fixture(
        fixture_path,
        [
            {
                "id": "available",
                "name": "Available Apartment",
                "city": "Baku",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
            {
                "id": "not-available",
                "name": (
                    "Unavailable Apartment"
                ),
                "city": "Baku",
                "available_dates": {
                    "check_in": "2026-05-01",
                    "check_out": "2026-05-30",
                },
            },
        ],
    )

    retriever = FixturesRetriever(
        path=fixture_path
    )

    results = await retriever.get_candidates(
        _request(),
        max_items=10,
    )

    assert len(results) == 1
    assert (
        results[0].id
        == "available"
    )


@pytest.mark.asyncio
async def test_max_items_is_applied_after_filtering(
    tmp_path: Path,
):
    fixture_path = (
        tmp_path / "listings.json"
    )

    _write_fixture(
        fixture_path,
        [
            {
                "id": "tokyo-first",
                "name": "Tokyo",
                "city": "Tokyo",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
            {
                "id": "baku-1",
                "name": "Baku 1",
                "city": "Baku",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
            {
                "id": "baku-2",
                "name": "Baku 2",
                "city": "Baku",
                "available_dates": {
                    "check_in": "2026-04-01",
                    "check_out": "2026-04-30",
                },
            },
        ],
    )

    retriever = FixturesRetriever(
        path=fixture_path
    )

    results = await retriever.get_candidates(
        _request(),
        max_items=2,
    )

    assert len(results) == 2
    assert [
        result.id
        for result in results
    ] == [
        "baku-1",
        "baku-2",
    ]