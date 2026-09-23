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


HOTEL_URLS = [
    "https://www.booking.com/hotel/az/star-baki.en-gb.html?checkout=2026-09-18&lang=en-gb&group_children=0&soz=1&explicit_lang_change=1&checkin=2026-09-15&selected_currency=USD&group_adults=2&no_rooms=1&lang_changed=1&explicit_curr_change=1",

    "https://www.booking.com/hotel/az/urban-apartment.en-gb.html?checkout=2026-09-18&lang=en-gb&group_children=0&soz=1&explicit_lang_change=1&checkin=2026-09-15&selected_currency=USD&group_adults=2&no_rooms=1&lang_changed=1&explicit_curr_change=1",

    "https://www.booking.com/hotel/az/kvartira-v-tsentre-baku-baku3.en-gb.html?checkout=2026-09-18&lang=en-gb&group_children=0&soz=1&explicit_lang_change=1&checkin=2026-09-15&selected_currency=USD&group_adults=2&no_rooms=1&lang_changed=1&explicit_curr_change=1",

    "https://www.booking.com/hotel/az/nb-baku.en-gb.html?checkout=2026-09-18&lang=en-gb&group_children=0&soz=1&explicit_lang_change=1&checkin=2026-09-15&selected_currency=USD&group_adults=2&no_rooms=1&lang_changed=1&explicit_curr_change=1",

    "https://www.booking.com/hotel/az/royal-caravan-baku.en-gb.html?checkout=2026-09-18&lang=en-gb&group_children=0&soz=1&explicit_lang_change=1&checkin=2026-09-15&selected_currency=USD&group_adults=2&no_rooms=1&lang_changed=1&explicit_curr_change=1",
]


async def main() -> None:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_TOKEN in environment")

    actor = os.getenv(
        "APIFY_BOOKING_ACTOR",
        "voyager~booking-scraper",
    )

    api_base = os.getenv(
        "APIFY_BASE_URL",
        "https://api.apify.com",
    )

    actor_input = {
        "startUrls": [
            {"url": url}
            for url in HOTEL_URLS
        ],
        "currency": "USD",
        "language": "en-gb",
        "maxItems": 5,
        "checkIn": "2026-09-15",
        "checkOut": "2026-09-18",
        "adults": 2,
        "children": 0,
        "rooms": 1,
    }

    omit_fields = (
        "images,"
        "roomImages,"
        "breadcrumbs,"
        "categoryReviews"
    )

    url = (
        f"{api_base}/v2/acts/{actor}/run-sync-get-dataset-items"
        f"?token={token}"
        f"&format=json"
        f"&clean=true"
        f"&timeout=180"
        f"&maxItems=5"
        f"&omit={omit_fields}"
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

    print("\n=== DIRECT URL RETRIEVAL BENCHMARK ===")
    print(f"Requested URLs: {len(HOTEL_URLS)}")

    if isinstance(items, list):
        print(f"Returned listings: {len(items)}")
    else:
        print(f"Unexpected response type: {type(items)}")

    print(
        f"Total Apify latency: "
        f"{latency_seconds:.2f} sec"
    )

    if isinstance(items, list):
        print("\n=== RETURNED HOTELS ===")

        for index, item in enumerate(items, start=1):
            print(
                f"{index}. "
                f"{item.get('name', '<missing name>')}"
            )

        print("\n=== FIELD COVERAGE ===")

        for index, item in enumerate(items, start=1):
            print(
                f"\nListing #{index}: "
                f"{item.get('name')}"
            )
            print(
                f"  description: "
                f"{bool(item.get('description'))}"
            )
            print(
                f"  facilities: "
                f"{bool(item.get('facilities'))}"
            )
            print(
                f"  rooms: "
                f"{bool(item.get('rooms'))}"
            )
            print(
                f"  policies: "
                f"{bool(item.get('policies'))}"
            )
            print(
                f"  highlights: "
                f"{bool(item.get('highlights'))}"
            )


if __name__ == "__main__":
    asyncio.run(main())