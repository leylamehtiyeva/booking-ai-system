from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from app.retrieval.apify import (
    _post_json_sync,
    _save_apify_debug_payload,
)


load_dotenv()


async def main() -> None:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_TOKEN in environment")

    api_base = os.getenv(
        "APIFY_BASE_URL",
        "https://api.apify.com",
    )

    # ВАЖНО:
    # специально не используем APIFY_BOOKING_ACTOR,
    # потому что там сейчас настроен Full Booking Scraper.
    actor = "voyager~fast-booking-scraper"

    actor_input = {
        "search": "Baku",
        "currency": "USD",
        "language": "en-gb",
        "maxItems": 10,
        "checkIn": "2026-09-15",
        "checkOut": "2026-09-18",
        "adults": 2,
        "children": 0,
        "rooms": 1,
    }

    url = (
        f"{api_base}/v2/acts/{actor}/run-sync-get-dataset-items"
        f"?token={token}"
        f"&format=json"
        f"&clean=true"
        f"&timeout=180"
    )

    started = time.perf_counter()

    items = await asyncio.to_thread(
        _post_json_sync,
        url,
        actor_input,
        180,
    )

    latency_seconds = time.perf_counter() - started

    _save_apify_debug_payload(
        actor_input,
        items,
    )

    print("\n=== FAST RETRIEVAL BENCHMARK ===")
    print("City: Baku")
    print(f"Requested listings: {actor_input['maxItems']}")

    if not isinstance(items, list):
        print(f"Unexpected response type: {type(items)}")
        return

    print(f"Returned listings: {len(items)}")
    print(f"Total Apify latency: {latency_seconds:.2f} sec")

    print("\n=== RETURNED HOTELS ===")

    for index, item in enumerate(items, start=1):
        print(
            f"{index}. "
            f"{item.get('name', '<missing name>')}"
        )

    print("\n=== FIELD COVERAGE ===")

    fields = [
        "url",
        "name",
        "address",
        "price",
        "currency",
        "rating",
        "reviews",
        "stars",
        "roomType",
        "persons",
        "image",
        "type",
        "description",
        "facilities",
        "rooms",
        "policies",
        "highlights",
    ]

    for field in fields:
        count = sum(
            item.get(field) is not None
            and item.get(field) != []
            and item.get(field) != ""
            for item in items
        )

        print(
            f"{field}: "
            f"{count}/{len(items)}"
        )

    print("\n=== RAW KEYS ===")

    if items:
        print(sorted(items[0].keys()))


if __name__ == "__main__":
    asyncio.run(main())