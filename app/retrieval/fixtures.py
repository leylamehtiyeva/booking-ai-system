from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


FIXTURES_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "listings_sample.json"
)


def _normalize_city(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip().casefold()
    return text or None


def _listing_city_matches(
    listing: ListingRaw,
    requested_city: str | None,
) -> bool:
    requested = _normalize_city(
        requested_city
    )

    if not requested:
        return False

    listing_city = _normalize_city(
        listing.city
    )

    if not listing_city:
        return False

    return listing_city == requested


def _parse_iso_date(
    value: Any,
) -> date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(
                value.strip()
            )
        except ValueError:
            return None

    return None


def _covers_dates(
    listing: ListingRaw,
    check_in: date,
    check_out: date,
) -> bool:
    available_dates = (
        listing.available_dates
    )

    if available_dates is None:
        return True

    if isinstance(
        available_dates,
        dict,
    ):
        listing_check_in = _parse_iso_date(
            available_dates.get(
                "check_in"
            )
        )
        listing_check_out = _parse_iso_date(
            available_dates.get(
                "check_out"
            )
        )
    else:
        listing_check_in = _parse_iso_date(
            available_dates.check_in
        )
        listing_check_out = _parse_iso_date(
            available_dates.check_out
        )

    if (
        listing_check_in is None
        or listing_check_out is None
    ):
        return True

    return (
        listing_check_in <= check_in
        and check_out <= listing_check_out
    )


class FixturesRetriever:
    def __init__(
        self,
        path: Path = FIXTURES_PATH,
    ):
        self.path = path

    async def get_candidates(
        self,
        req: SearchRequest,
        max_items: int,
    ) -> list[ListingRaw]:
        data = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        listings = [
            ListingRaw.model_validate(item)
            for item in data
        ]

        listings = [
            listing
            for listing in listings
            if _listing_city_matches(
                listing,
                req.city,
            )
        ]

        if (
            req.check_in is not None
            and req.check_out is not None
        ):
            listings = [
                listing
                for listing in listings
                if _covers_dates(
                    listing,
                    req.check_in,
                    req.check_out,
                )
            ]

        return listings[:max_items]