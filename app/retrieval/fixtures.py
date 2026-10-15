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


def _availability_window(
    listing: ListingRaw,
) -> tuple[date | None, date | None] | None:
    """
    (start, end) of the listing's availability window as stored in the
    fixture file, or None when the listing has no window at all.
    """
    available_dates = getattr(
        listing,
        "available_dates",
        None,
    )

    if available_dates is None:
        return None

    if isinstance(
        available_dates,
        dict,
    ):
        return (
            _parse_iso_date(available_dates.get("check_in")),
            _parse_iso_date(available_dates.get("check_out")),
        )

    return (
        _parse_iso_date(getattr(available_dates, "check_in", None)),
        _parse_iso_date(getattr(available_dates, "check_out", None)),
    )


def _covers_dates(
    listing: ListingRaw,
    check_in: date,
    check_out: date,
) -> bool:
    window = _availability_window(listing)

    if window is None:
        return True

    listing_check_in, listing_check_out = window

    if (
        listing_check_in is None
        or listing_check_out is None
    ):
        return True

    # Fixture availability is a recurring yearly calendar: the window is
    # matched by month/day in the requested year (or the year before, for
    # windows spanning New Year), so the offline demo does not expire.
    # Shifting by whole years keeps the window length - and therefore the
    # fixture `price`, which is the total for the whole window - unchanged.
    year_shift = check_in.year - listing_check_in.year

    for shift in (year_shift, year_shift - 1):
        window_start = _shift_years(listing_check_in, shift, is_window_end=False)
        window_end = _shift_years(listing_check_out, shift, is_window_end=True)

        if window_start <= check_in and check_out <= window_end:
            return True

    return False


def _shift_years(
    value: date,
    years: int,
    *,
    is_window_end: bool,
) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # A Feb 29 edge in a non-leap target year: narrow the window to the
        # nearest real day (start -> Mar 1, end -> Feb 28), never widen it.
        target_year = value.year + years
        if is_window_end:
            return date(target_year, 2, 28)
        return date(target_year, 3, 1)


def _price_for_requested_stay(
    listing: ListingRaw,
    check_in: date | None,
    check_out: date | None,
) -> float | None:
    """
    Fixture files store `price` as the total for the whole availability
    window (rate x window nights). Downstream, ListingRaw.price means the
    total for the requested stay - what Apify returns - so it is prorated
    by night count:

        requested_total = window_total * requested_nights / window_nights

    The ORIGINAL window length is used, which keeps the nightly rate the
    same for recurring (other-year), cross-year and leap-narrowed windows.
    Without a known window length or stay length there is no defensible
    stay price, so None is returned rather than an invented one.
    """
    if listing.price is None or check_in is None or check_out is None:
        return None

    window = _availability_window(listing)
    if window is None or window[0] is None or window[1] is None:
        return None

    window_nights = (window[1] - window[0]).days
    requested_nights = (check_out - check_in).days

    if window_nights <= 0 or requested_nights <= 0:
        return None

    return round(
        float(listing.price) * requested_nights / window_nights,
        2,
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

        # Return copies priced for the requested stay; the fixture data
        # itself is never modified.
        listings = [
            listing.model_copy(
                update={
                    "price": _price_for_requested_stay(
                        listing,
                        req.check_in,
                        req.check_out,
                    )
                }
            )
            for listing in listings[:max_items]
        ]

        return listings