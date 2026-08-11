from __future__ import annotations

import asyncio
import json
import os
from datetime import date

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
import logging
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
import time
from app.observability.trace import RequestTrace, ExternalCallTrace
from app.observability.pricing import estimate_apify_cost_usd
logger = logging.getLogger(__name__)




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
        for x in items[:max_items]:
            try:
                out.append(ListingRaw.model_validate(x))
            except Exception:
                continue

        return out

