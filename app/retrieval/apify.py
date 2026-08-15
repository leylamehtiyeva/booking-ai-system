from __future__ import annotations

import asyncio
import json
import os
from datetime import date

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
import logging
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
import time
from app.observability.trace import RequestTrace, ExternalCallTrace
from app.observability.pricing import estimate_apify_cost_usd
logger = logging.getLogger(__name__)
from app.schemas.listing import (
    BedGroup,
    Facility,
    Highlight,
    ListingRaw,
    Policy,
    Room,
    RoomOption,
)



def _normalize_policies(
    value: Any,
) -> list[Policy]:
    if not isinstance(value, list):
        return []

    out: list[Policy] = []

    for item in value:
        if not isinstance(item, Mapping):
            continue

        out.append(
            Policy(
                title=_clean_text(
                    item.get("title")
                ),
                content=_clean_text(
                    item.get("content")
                ),
            )
        )

    return out


def _normalize_highlights(
    value: Any,
) -> list[Highlight]:
    if not isinstance(value, list):
        return []

    out: list[Highlight] = []

    for item in value:
        if not isinstance(item, Mapping):
            continue

        out.append(
            Highlight(
                header=_clean_text(
                    item.get("header")
                ),
                contents=_clean_string_list(
                    item.get("contents")
                ),
            )
        )

    return out

def _save_apify_debug_payload(actor_input: Dict[str, Any], items: Any) -> None:
    debug_dir = Path("logs/apify_raw")
    debug_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": ts,
        "actor_input": actor_input,
        "items_count": len(items) if isinstance(items, list) else None,
        "items": items,
    }

    path = debug_dir / f"apify_raw_{ts}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    logger.debug("Apify raw payload saved to %s", path)


def _iso(d: Any) -> str:
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        return d.strip()
    raise ValueError(f"Invalid date value: {d!r}")


def _clean_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _safe_float(
    value: Any,
) -> float | None:
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


def _safe_int(
    value: Any,
) -> int | None:
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


def _safe_bool(
    value: Any,
) -> bool | None:
    if isinstance(value, bool):
        return value

    return None


def _clean_string_list(
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        return []

    out: list[str] = []

    for item in value:
        text = _clean_text(item)

        if text is not None:
            out.append(text)

    return out


def _normalize_facility(
    value: Any,
    *,
    category: str | None = None,
    overview: str | None = None,
) -> Facility | None:
    if isinstance(value, str):
        name = _clean_text(value)

        if name is None:
            return None

        return Facility(
            name=name,
            category=category,
            overview=overview,
        )

    if not isinstance(value, Mapping):
        return None

    name = _clean_text(
        value.get("name")
    )

    if name is None:
        return None

    additional_info = value.get(
        "additionalInfo"
    )

    if not isinstance(
        additional_info,
        Mapping,
    ):
        additional_info = {}

    return Facility(
        name=name,
        category=category,
        overview=overview,
        requires_additional_charge=_safe_bool(
            additional_info.get(
                "requiresAdditionalCharge"
            )
        ),
        is_off_site=_safe_bool(
            additional_info.get(
                "isOffSite"
            )
        ),
    )
    
    
def _normalize_listing_facilities(
    value: Any,
) -> list[Facility]:
    if not isinstance(value, list):
        return []

    out: list[Facility] = []

    for group in value:
        if isinstance(group, str):
            facility = _normalize_facility(
                group
            )

            if facility is not None:
                out.append(facility)

            continue

        if not isinstance(group, Mapping):
            continue

        category = _clean_text(
            group.get("name")
        )

        overview = _clean_text(
            group.get("overview")
        )

        nested = group.get("facilities")

        if (
            isinstance(nested, list)
            and nested
        ):
            for item in nested:
                facility = _normalize_facility(
                    item,
                    category=category,
                    overview=overview,
                )

                if facility is not None:
                    out.append(facility)

            continue

        # Important:
        # e.g. {
        #   "name": "Parking",
        #   "overview": "No parking available.",
        #   "facilities": []
        # }
        #
        # We must not lose this evidence.
        if category is not None:
            out.append(
                Facility(
                    name=category,
                    category=category,
                    overview=overview,
                )
            )

    return out




def _normalize_bed_groups(
    value: Any,
) -> list[BedGroup]:
    if not isinstance(value, list):
        return []

    out: list[BedGroup] = []

    for item in value:
        if not isinstance(item, Mapping):
            continue

        out.append(
            BedGroup(
                room=_clean_text(
                    item.get("room")
                ),
                beds=_clean_string_list(
                    item.get("beds")
                ),
            )
        )

    return out


def _normalize_room_options(
    value: Any,
) -> list[RoomOption]:
    if not isinstance(value, list):
        return []

    out: list[RoomOption] = []

    for item in value:
        if not isinstance(item, Mapping):
            continue

        out.append(
            RoomOption(
                name=_clean_text(
                    item.get("name")
                ),
                price=_safe_float(
                    item.get("price")
                ),
                currency=_clean_text(
                    item.get("currency")
                ),
                persons=_safe_int(
                    item.get("persons")
                ),
                cancellation_type=_clean_text(
                    item.get(
                        "cancellationType"
                    )
                    or item.get(
                        "cancellation_type"
                    )
                ),
                free_cancellation=_safe_bool(
                    item.get(
                        "freeCancellation"
                    )
                ),
                choices=_clean_string_list(
                    item.get(
                        "yourChoices"
                    )
                    or item.get(
                        "choices"
                    )
                ),
            )
        )

    return out


def _normalize_room_facilities(
    value: Any,
) -> list[Facility]:
    if not isinstance(value, list):
        return []

    out: list[Facility] = []

    for item in value:
        facility = _normalize_facility(
            item
        )

        if facility is not None:
            out.append(facility)

    return out


def _normalize_rooms(
    value: Any,
) -> list[Room]:
    if not isinstance(value, list):
        return []

    out: list[Room] = []

    for item in value:
        if not isinstance(item, Mapping):
            continue

        out.append(
            Room(
                name=_clean_text(
                    item.get("roomType")
                    or item.get("name")
                ),
                available=_safe_bool(
                    item.get("available")
                ),
                persons=_safe_int(
                    item.get("persons")
                ),
                bed_types=_normalize_bed_groups(
                    item.get("bedTypes")
                    or item.get("bed_types")
                ),
                facilities=(
                    _normalize_room_facilities(
                        item.get("facilities")
                    )
                ),
                options=(
                    _normalize_room_options(
                        item.get("options")
                    )
                ),
            )
        )

    return out



def normalize_apify_listing(
    item: Mapping[str, Any],
) -> ListingRaw:
    """
    Convert one Voyager Booking Scraper result
    into our provider-independent ListingRaw.

    This function is the boundary between
    Voyager's external schema and our internal
    search domain.
    """

    raw_address = item.get("address")

    city: str | None = None
    address: str | None = None

    if isinstance(raw_address, Mapping):
        city = _clean_text(
            raw_address.get("city")
        )

        address = _clean_text(
            raw_address.get("full")
        )

    elif isinstance(raw_address, str):
        address = _clean_text(
            raw_address
        )

    if city is None:
        city = _clean_text(
            item.get("city")
        )

    raw_id = (
        item.get("hotelId")
        if item.get("hotelId") is not None
        else item.get("id")
    )

    listing_id = (
        str(raw_id)
        if raw_id is not None
        else None
    )

    return ListingRaw(
        id=listing_id,
        city=city,
        address=address,

        name=_clean_text(
            item.get("name")
        ),
        url=_clean_text(
            item.get("url")
        ),

        price=_safe_float(
            item.get("price")
        ),
        currency=_clean_text(
            item.get("currency")
        ),

        rating=_safe_float(
            item.get("rating")
        ),
        stars=_safe_int(
            item.get("stars")
        ),

        property_type=_clean_text(
            item.get("type")
            or item.get("property_type")
        ),

        # Full Voyager currently has no reliable
        # top-level equivalent in the payload we
        # inspected. Do not infer it from rooms.
        room_type=_clean_text(
            item.get("roomType")
            or item.get("room_type")
        ),

        # Do not derive this from max(room persons).
        # That was the occupancy bug we just removed.
        max_occupancy=_safe_int(
            item.get("max_occupancy")
        ),

        description=_clean_text(
            item.get("description")
        ),
        
        fine_print=_clean_text(
            item.get("finePrint")
            or item.get("fine_print")
        ),

        policies=_normalize_policies(
            item.get("policies")
        ),

        highlights=_normalize_highlights(
            item.get("highlights")
        ),

        facilities=(
            _normalize_listing_facilities(
                item.get("facilities")
            )
        ),

        rooms=_normalize_rooms(
            item.get("rooms")
        ),

        raw=dict(item),
    )

def _apify_property_type(req: SearchRequest) -> str | None:
    """
    Convert internal canonical property type into Apify Booking actor format.

    Internal:
        apartment, hotel

    Apify input expects:
        Apartments, Hotels

    Apify output may return:
        apartment, hotel
    """
    if not req.property_types:
        return None

    pt = req.property_types[0]
    value = pt.value if hasattr(pt, "value") else str(pt)

    apify_property_types = {
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
        "motel": "Motels",
        "boat": "Boats",
        "holiday_park": "Holiday parks",
        "luxury_tent": "Luxury tents",
    }

    return apify_property_types.get(value)


def _post_json_sync(url: str, payload: Dict[str, Any], timeout: int = 180) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


class ApifyRetriever:
    async def get_candidates(
        self,
        req: SearchRequest,
        max_items: int,
        trace: RequestTrace | None = None,
        ) -> List[ListingRaw]:        
        token = os.getenv("APIFY_TOKEN")
        if not token:
            raise ValueError("Missing APIFY_TOKEN in environment")

        actor = os.getenv("APIFY_BOOKING_ACTOR", "voyager~booking-scraper")

        if not req.city:
            raise ValueError("SearchRequest.city is required for Apify search")

        if req.check_in is None or req.check_out is None:
            raise ValueError("SearchRequest.check_in/check_out are required for Apify search")

        currency = getattr(req, "currency", None) or os.getenv("APIFY_CURRENCY", "USD")
        language = os.getenv("APIFY_LANGUAGE", "en-gb")
        adults = int(getattr(req, "adults", 2) or 2)
        children = int(getattr(req, "children", 0) or 0)
        rooms = int(getattr(req, "rooms", 1) or 1)


        search_query = str(req.city).strip()
        property_type = _apify_property_type(req)
        

        actor_input = {
            "search": search_query,
            "currency": currency,
            "language": language,
            "maxItems": int(max_items),
            "checkIn": _iso(req.check_in),
            "checkOut": _iso(req.check_out),
            "adults": adults,
            "children": children,
            "rooms": rooms,
        }
        
        if property_type:
            actor_input["propertyType"] = property_type

        api_base = os.getenv("APIFY_BASE_URL", "https://api.apify.com")
        omit_fields = "images,roomImages,breadcrumbs,categoryReviews"

        url = (
            f"{api_base}/v2/acts/{actor}/run-sync-get-dataset-items"
            f"?token={token}"
            f"&format=json"
            f"&clean=true"
            f"&timeout=180"
            f"&maxItems={int(max_items)}"
            f"&omit={omit_fields}"
        )

        try:
            started = time.perf_counter()
            items = await asyncio.to_thread(_post_json_sync, url, actor_input, 180)

            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            if trace is not None:
                trace.add_external_call(
                    ExternalCallTrace(
                        step="apify_booking_search",
                        provider="apify",
                        latency_ms=latency_ms,
                        estimated_cost_usd=estimate_apify_cost_usd(run_count=1),
                        success=True,
                        metadata={
                            "actor": actor,
                            "max_items": max_items,
                            "city": req.city,
                            "check_in": _iso(req.check_in),
                            "check_out": _iso(req.check_out),
                        },
                    )
                )
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass

            raise RuntimeError(f"Apify HTTPError {e.code}: {body}") from e
        except URLError as e:
            raise RuntimeError(f"Apify URLError: {e}") from e

        _save_apify_debug_payload(actor_input, items)
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected Apify response type: {type(items)}")

        out: List[ListingRaw] = []

        for index, item in enumerate(
            items[:max_items]
        ):
            if not isinstance(item, Mapping):
                logger.warning(
                    "Skipping Apify item at index %s: "
                    "expected mapping, got %s",
                    index,
                    type(item).__name__,
                )
                continue

            try:
                listing = normalize_apify_listing(
                    item
                )
            except Exception:
                logger.exception(
                    "Failed to normalize Apify listing "
                    "at index %s, name=%r",
                    index,
                    item.get("name"),
                )
                continue

            out.append(listing)

        return out

