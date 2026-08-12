from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date
from typing import Any, Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from app.observability.trace import (
    ExternalCallTrace,
    RequestTrace,
)
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


FAST_ACTOR_DEFAULT = "voyager~fast-booking-scraper"


APIFY_PROPERTY_TYPES = {
    "hotel": "Hotels",
    "apartment": "Apartments",
    "hostel": "Hostels",
    "guest_house": "Guest houses",
    "homestay": "Homestays",
    "bed_and_breakfast": "Bed and breakfasts",
    "holiday_home": "Holiday homes",
    "villa": "Villas",
    "resort": "Resorts",
    "campsite": "Campsites",
}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")

        if not cleaned:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

        return None

    if isinstance(value, str):
        cleaned = value.strip()

        if not cleaned:
            return None

        try:
            number = float(cleaned)
        except ValueError:
            return None

        if number.is_integer():
            return int(number)

    return None


def _iso_date(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, str):
        cleaned = value.strip()

        if cleaned:
            return cleaned

    raise ValueError(
        f"Invalid date value: {value!r}"
    )


def _single_apify_property_type(
    req: SearchRequest,
) -> str | None:
    """
    Push property type to the provider only when the request
    contains exactly one supported property type.

    The Fast actor accepts only one propertyType value.
    If the user allows several property types, pushing only the
    first one would incorrectly reduce recall.
    """

    property_types = req.property_types or []

    if len(property_types) != 1:
        return None

    property_type = property_types[0]

    value = (
        property_type.value
        if hasattr(property_type, "value")
        else str(property_type)
    )

    return APIFY_PROPERTY_TYPES.get(value)


def normalize_fast_listing(
    item: Mapping[str, Any],
) -> ListingRaw:
    """
    Convert one Voyager Fast result into ListingRaw.

    Fast provider fields are translated into our internal names.
    Rich evidence that Fast does not provide is left unknown.
    """

    return ListingRaw(
        # Do NOT copy SearchRequest.city here.
        # Search scope is not proof of listing location.
        city=None,
        address=_clean_text(item.get("address")),
        name=_clean_text(item.get("name")),
        url=_clean_text(item.get("url")),
        price=_safe_float(item.get("price")),
        currency=_clean_text(item.get("currency")),
        rating=_safe_float(item.get("rating")),
        stars=_safe_int(item.get("stars")),
        room_type=_clean_text(
            item.get("roomType")
        ),
        max_occupancy=_safe_int(
            item.get("persons")
        ),
        raw=dict(item),
    )
    
def _format_provider_number(
    value: float | int,
) -> str:
    """
    Format a numeric filter value for the Voyager actor.

    Examples:
        50.0 -> "50"
        50.5 -> "50.5"
    """

    number = float(value)

    if number.is_integer():
        return str(int(number))

    return str(number)


def _build_per_night_price_filter(
    req: SearchRequest,
) -> str | None:
    """
    Build Voyager minMaxPrice only when the canonical price
    constraint is explicitly per-night.

    Voyager minMaxPrice is a per-night provider filter.

    total_stay and unspecified scope are intentionally not
    pushed down because they have different semantics and
    must be checked locally.
    """

    if req.filters is None:
        return None

    price = req.filters.price

    if price is None:
        return None

    if price.scope != "per_night":
        return None

    min_amount = price.min_amount
    max_amount = price.max_amount

    if min_amount is None and max_amount is None:
        return None

    # Invalid negative values are not pushed to the provider.
    # The canonical/local validation layer can deal with them
    # separately without risking provider-side false negatives.
    if (
        min_amount is not None
        and min_amount < 0
    ):
        return None

    if (
        max_amount is not None
        and max_amount < 0
    ):
        return None

    if (
        min_amount is not None
        and max_amount is not None
    ):
        return (
            f"{_format_provider_number(min_amount)}"
            "-"
            f"{_format_provider_number(max_amount)}"
        )

    if min_amount is not None:
        return (
            f"{_format_provider_number(min_amount)}+"
        )

    return (
        "0-"
        f"{_format_provider_number(max_amount)}"
    )


def build_fast_search_input(
    req: SearchRequest,
    *,
    max_items: int,
    language: str = "en-gb",
) -> dict[str, Any]:
    """
    Build Voyager Fast actor input from the canonical SearchRequest.

    Safe provider push-down currently includes:
    - search context
    - dates
    - occupancy
    - one supported property type
    - minimum guest rating
    - explicitly per-night price range

    Constraints that cannot be represented safely by the provider
    remain for downstream local evaluation.
    """

    if max_items <= 0:
        raise ValueError(
            "max_items must be > 0"
        )

    if not req.city:
        raise ValueError(
            "SearchRequest.city is required "
            "for Fast retrieval"
        )

    if (
        req.check_in is None
        or req.check_out is None
    ):
        raise ValueError(
            "SearchRequest.check_in/check_out "
            "are required for Fast retrieval"
        )

    if req.adults < 1:
        raise ValueError(
            "SearchRequest.adults must be >= 1"
        )

    if req.children < 0:
        raise ValueError(
            "SearchRequest.children must be >= 0"
        )

    if req.rooms < 1:
        raise ValueError(
            "SearchRequest.rooms must be >= 1"
        )

    price_filter = (
        req.filters.price
        if req.filters is not None
        else None
    )

    # If the user explicitly gave a currency for a per-night
    # budget, the provider must apply the price range in that
    # same currency.
    provider_currency = (
        price_filter.currency
        if (
            price_filter is not None
            and price_filter.scope == "per_night"
            and price_filter.currency
        )
        else req.currency
    ) or "USD"

    actor_input: dict[str, Any] = {
        "search": req.city.strip(),
        "maxItems": int(max_items),
        "currency": provider_currency,
        "language": language,
        "checkIn": _iso_date(req.check_in),
        "checkOut": _iso_date(req.check_out),
        "adults": int(req.adults),
        "children": int(req.children),
        "rooms": int(req.rooms),
    }

    property_type = (
        _single_apify_property_type(req)
    )

    if property_type is not None:
        actor_input["propertyType"] = (
            property_type
        )

    if (
        req.min_guest_rating is not None
    ):
        actor_input["minScore"] = (
            _format_provider_number(
                req.min_guest_rating
            )
        )

    per_night_price = (
        _build_per_night_price_filter(req)
    )

    if per_night_price is not None:
        actor_input["minMaxPrice"] = (
            per_night_price
        )

    return actor_input


def _post_json_sync(
    url: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> Any:
    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urlrequest.Request(
        url,
        data=data,
        headers={
            "Content-Type": (
                "application/json; charset=utf-8"
            ),
            "Accept": "application/json",
        },
        method="POST",
    )

    with urlrequest.urlopen(
        request,
        timeout=timeout,
    ) as response:
        body = response.read().decode(
            "utf-8"
        )

    return json.loads(body)


def _estimate_fast_cost_usd(
    result_count: int,
) -> float:
    """
    Estimate Fast actor cost from returned result count.

    The default can be overridden without changing code.
    """

    price_per_result = float(
        os.getenv(
            "APIFY_FAST_COST_PER_RESULT_USD",
            "0.002",
        )
    )

    return result_count * price_per_result


def _record_external_call(
    *,
    trace: RequestTrace | None,
    latency_ms: float,
    actor: str,
    max_items: int,
    success: bool,
    returned_items: int | None = None,
    error: str | None = None,
) -> None:
    if trace is None:
        return

    estimated_cost = None

    if (
        success
        and returned_items is not None
    ):
        estimated_cost = (
            _estimate_fast_cost_usd(
                returned_items
            )
        )

    trace.add_external_call(
        ExternalCallTrace(
            step="apify_booking_fast_search",
            provider="apify",
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost,
            success=success,
            error=error,
            metadata={
                "actor": actor,
                "max_items": max_items,
                "returned_items": (
                    returned_items
                ),
                "retrieval_mode": "fast",
            },
        )
    )


class ApifyFastRetriever:
    async def get_candidates(
        self,
        req: SearchRequest,
        max_items: int,
        trace: RequestTrace | None = None,
    ) -> list[ListingRaw]:
        token = os.getenv("APIFY_TOKEN")

        if not token:
            raise ValueError(
                "Missing APIFY_TOKEN "
                "in environment"
            )

        actor = os.getenv(
            "APIFY_FAST_BOOKING_ACTOR",
            FAST_ACTOR_DEFAULT,
        )

        language = os.getenv(
            "APIFY_LANGUAGE",
            "en-gb",
        )

        actor_input = build_fast_search_input(
            req,
            max_items=max_items,
            language=language,
        )

        api_base = os.getenv(
            "APIFY_BASE_URL",
            "https://api.apify.com",
        ).rstrip("/")

        query = urlencode(
            {
                "token": token,
            }
        )

        url = (
            f"{api_base}/v2/acts/{actor}"
            "/run-sync-get-dataset-items"
            f"?{query}"
        )

        started = time.perf_counter()

        try:
            items = await asyncio.to_thread(
                _post_json_sync,
                url,
                actor_input,
                180,
            )

        except HTTPError as exc:
            latency_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            body = ""

            try:
                body = exc.read().decode(
                    "utf-8"
                )
            except Exception:
                pass

            error = (
                f"Apify HTTPError "
                f"{exc.code}: {body}"
            )

            _record_external_call(
                trace=trace,
                latency_ms=latency_ms,
                actor=actor,
                max_items=max_items,
                success=False,
                error=error,
            )

            raise RuntimeError(
                error
            ) from exc

        except URLError as exc:
            latency_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            error = (
                f"Apify URLError: {exc}"
            )

            _record_external_call(
                trace=trace,
                latency_ms=latency_ms,
                actor=actor,
                max_items=max_items,
                success=False,
                error=error,
            )

            raise RuntimeError(
                error
            ) from exc

        latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        if not isinstance(items, list):
            error = (
                "Unexpected Apify Fast "
                "response type: "
                f"{type(items)}"
            )

            _record_external_call(
                trace=trace,
                latency_ms=latency_ms,
                actor=actor,
                max_items=max_items,
                success=False,
                error=error,
            )

            raise RuntimeError(error)

        items = items[:max_items]

        listings = [
            normalize_fast_listing(item)
            for item in items
            if isinstance(item, Mapping)
        ]

        _record_external_call(
            trace=trace,
            latency_ms=latency_ms,
            actor=actor,
            max_items=max_items,
            success=True,
            returned_items=len(listings),
        )

        return listings