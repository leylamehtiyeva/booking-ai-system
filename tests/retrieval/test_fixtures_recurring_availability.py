"""
Fixture listings carry availability windows dated 2026, and their `price`
is the TOTAL for exactly that window (rate x window nights - see
evaluation/tasks/end_to_end/README.md). Rewriting the dates would either
change window lengths (breaking price semantics) or the explicit-2026
golden datasets, so the fixture retriever instead treats each window as a
recurring yearly calendar: the offline demo must not expire with time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.retrieval.fixtures import FixturesRetriever
from app.schemas.query import SearchRequest


def _request(check_in: date, check_out: date, city: str = "Baku") -> SearchRequest:
    return SearchRequest(city=city, check_in=check_in, check_out=check_out, constraints=[])


def _write(path: Path, windows: dict[str, tuple[str, str]]) -> FixturesRetriever:
    path.write_text(
        json.dumps(
            [
                {
                    "id": listing_id,
                    "name": listing_id,
                    "city": "Baku",
                    "available_dates": {"check_in": start, "check_out": end},
                }
                for listing_id, (start, end) in windows.items()
            ]
        ),
        encoding="utf-8",
    )
    return FixturesRetriever(path=path)


async def _ids(retriever: FixturesRetriever, request: SearchRequest) -> list[str]:
    return [listing.id for listing in await retriever.get_candidates(request, max_items=10)]


@pytest.mark.parametrize("year", [2026, 2027, 2030])
async def test_window_matches_same_month_and_day_in_any_year(tmp_path, year):
    retriever = _write(
        tmp_path / "listings.json",
        {"april": ("2026-04-01", "2026-04-30"), "may": ("2026-05-01", "2026-05-30")},
    )

    ids = await _ids(retriever, _request(date(year, 4, 10), date(year, 4, 15)))

    assert ids == ["april"]


async def test_stay_outside_window_is_still_excluded(tmp_path):
    retriever = _write(tmp_path / "listings.json", {"april": ("2026-04-05", "2026-04-25")})

    assert await _ids(retriever, _request(date(2027, 4, 1), date(2027, 4, 10))) == []
    assert await _ids(retriever, _request(date(2027, 4, 20), date(2027, 4, 28))) == []


async def test_cross_year_window_covers_stay_after_new_year(tmp_path):
    retriever = _write(tmp_path / "listings.json", {"winter": ("2026-12-20", "2027-01-10")})

    ids = await _ids(retriever, _request(date(2028, 1, 2), date(2028, 1, 5)))

    assert ids == ["winter"]


async def test_cross_year_stay_against_cross_year_window(tmp_path):
    retriever = _write(tmp_path / "listings.json", {"winter": ("2026-12-20", "2027-01-10")})

    ids = await _ids(retriever, _request(date(2027, 12, 28), date(2028, 1, 3)))

    assert ids == ["winter"]


async def test_readme_demo_scenario_is_stable_across_years():
    """README example: apartment in Baku, April 10-15."""
    retriever = FixturesRetriever()

    results_by_year = {
        year: await _ids(retriever, _request(date(year, 4, 10), date(year, 4, 15)))
        for year in (2026, 2027, 2028)
    }

    assert results_by_year[2026]
    assert results_by_year[2027] == results_by_year[2026]
    assert results_by_year[2028] == results_by_year[2026]


@pytest.mark.parametrize(
    ("window", "stay", "expected"),
    [
        # A Feb 29 window edge has no counterpart in a non-leap year; the
        # window narrows to the nearest real day instead of never matching.
        (("2028-02-01", "2028-02-29"), (date(2027, 2, 10), date(2027, 2, 15)), True),
        (("2028-02-01", "2028-02-29"), (date(2027, 2, 25), date(2027, 2, 28)), True),
        (("2028-02-29", "2028-03-20"), (date(2027, 3, 5), date(2027, 3, 10)), True),
        # ...but never widens: Feb 28 is not inside a window starting Feb 29.
        (("2028-02-29", "2028-03-20"), (date(2027, 2, 28), date(2027, 3, 3)), False),
        # The window is unchanged in leap years.
        (("2028-02-01", "2028-02-29"), (date(2032, 2, 25), date(2032, 2, 29)), True),
    ],
)
async def test_leap_day_window_edges(tmp_path, window, stay, expected):
    retriever = _write(tmp_path / "listings.json", {"feb": window})

    assert (await _ids(retriever, _request(*stay)) == ["feb"]) is expected


@pytest.mark.parametrize(
    ("stay", "expected"),
    [
        ((date(2027, 4, 1), date(2027, 4, 30)), True),
        ((date(2027, 3, 31), date(2027, 4, 5)), False),
        ((date(2027, 4, 25), date(2027, 5, 1)), False),
    ],
    ids=["exact-window", "starts-day-before", "ends-day-after"],
)
async def test_window_boundaries_are_inclusive_and_strict(tmp_path, stay, expected):
    retriever = _write(tmp_path / "listings.json", {"april": ("2026-04-01", "2026-04-30")})

    assert (await _ids(retriever, _request(*stay)) == ["april"]) is expected
