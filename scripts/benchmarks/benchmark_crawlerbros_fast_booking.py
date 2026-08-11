from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from app.retrieval.apify import _post_json_sync


load_dotenv()


DEBUG_DIR = Path("logs/apify_raw")


def save_debug_payload(
    actor: str,
    actor_input: dict,
    items,
) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": timestamp,
        "actor": actor,
        "actor_input": actor_input,
        "items_count": len(items) if isinstance(items, list) else None,
        "items": items,
    }

    path = (
        DEBUG_DIR
        / f"crawlerbros_fast_booking_{timestamp}.json"
    )

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Raw payload saved to: {path}")


async def main() -> None:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_TOKEN in environment")

    api_base = os.getenv(
        "APIFY_BASE_URL",
        "https://api.apify.com",
    )

    actor = "crawlerbros~fast-booking-scraper"

    actor_input = {
        "location": "Baku",
        "checkinDate": "2026-09-15",
        "checkoutDate": "2026-09-18",
        "adults": 2,
        "rooms": 1,
        "currency": "USD",
        "maxResults": 5,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }

    url = (
        f"{api_base}/v2/acts/{actor}"
        f"/run-sync-get-dataset-items"
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

    save_debug_payload(
        actor=actor,
        actor_input=actor_input,
        items=items,
    )

    print("\n=== CRAWLERBROS FAST BOOKING BENCHMARK ===")
    print("City: Baku")
    print("Requested listings: 5")

    if not isinstance(items, list):
        print(f"Unexpected response type: {type(items)}")
        return

    print(f"Returned listings: {len(items)}")
    print(
        f"Total Apify latency: "
        f"{latency_seconds:.2f} sec"
    )

    print("\n=== RETURNED HOTELS ===")

    for index, item in enumerate(items, start=1):
        print(
            f"{index}. "
            f"{item.get('name', '<missing name>')}"
        )

    fields = [
        "url",
        "name",
        "price",
        "currency",
        "rating",
        "reviewCount",
        "stars",
        "propertyType",
        "address",
        "city",
        "distance",
        "photoUrl",
    ]

    print("\n=== FIELD COVERAGE ===")

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