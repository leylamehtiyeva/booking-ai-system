from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from typing import Any, Dict, List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


def _iso(d: Any) -> str:
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        return d.strip()
    raise ValueError(f"Invalid date value: {d!r}")

def _apify_property_type(req: SearchRequest) -> str | None:
    """
    Convert internal canonical property type into Apify Booking actor format.

    Internal example:
        apartment

    Apify expects:
        Apartments
    """
    if not req.property_types:
        return None

    if len(req.property_types) > 1:
        # MVP: Apify supports one propertyType value.
        # We use only the first one for retrieval.
        # The rest can still be handled later by internal filtering/matching.
        pass

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
    async def get_candidates(self, req: SearchRequest, max_items: int) -> List[ListingRaw]:
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
        url = (
            f"{api_base}/v2/acts/{actor}/run-sync-get-dataset-items"
            f"?token={token}&format=json&clean=true&timeout=180&maxItems={int(max_items)}"
        )

        try:
            print("APIFY ACTOR INPUT:", json.dumps(actor_input, ensure_ascii=False, indent=2))
            items = await asyncio.to_thread(_post_json_sync, url, actor_input, 180)
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass

            # ✅ Debug what we sent (so 1 paid run gives full diagnosis)
            print("APIFY ACTOR INPUT:", json.dumps(actor_input, ensure_ascii=False, indent=2))

            raise RuntimeError(f"Apify HTTPError {e.code}: {body}") from e
        except URLError as e:
            print("APIFY ACTOR INPUT:", json.dumps(actor_input, ensure_ascii=False, indent=2))
            raise RuntimeError(f"Apify URLError: {e}") from e

        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected Apify response type: {type(items)}")

        out: List[ListingRaw] = []
        for x in items[:max_items]:
            try:
                out.append(ListingRaw.model_validate(x))
            except Exception:
                continue

        return out

