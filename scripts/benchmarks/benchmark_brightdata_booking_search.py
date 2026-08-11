from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from dotenv import load_dotenv


load_dotenv()


DATASET_ID = "gd_m4bf7a917zfezv9d5"

API_BASE = "https://api.brightdata.com"

DEBUG_DIR = Path("logs/brightdata_raw")


def _request_json(
    request: Request,
    timeout: int = 180,
):
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)

    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")

        print("\n=== BRIGHT DATA HTTP ERROR ===")
        print(f"Status: {exc.code}")
        print(f"Reason: {exc.reason}")
        print(f"Body: {raw}")

        raise


def _get_json(
    url: str,
    api_key: str,
):
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )

    return _request_json(request)


def _save_debug_payload(
    actor_input: dict,
    items,
) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": timestamp,
        "dataset_id": DATASET_ID,
        "input": actor_input,
        "items_count": (
            len(items)
            if isinstance(items, list)
            else None
        ),
        "items": items,
    }

    path = (
        DEBUG_DIR
        / f"brightdata_booking_search_{timestamp}.json"
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Raw payload saved to: {path}")


def _wait_for_snapshot(
    snapshot_id: str,
    api_key: str,
):
    print(
        f"Sync request returned snapshot_id: "
        f"{snapshot_id}"
    )

    while True:
        progress_url = (
            f"{API_BASE}/datasets/v3/progress/"
            f"{snapshot_id}"
        )

        _, progress = _get_json(
            progress_url,
            api_key,
        )

        status = progress.get("status")

        print(f"Snapshot status: {status}")

        if status == "ready":
            break

        if status == "failed":
            raise RuntimeError(
                f"Bright Data snapshot failed: {progress}"
            )

        time.sleep(2)

    download_url = (
        f"{API_BASE}/datasets/v3/snapshot/"
        f"{snapshot_id}?format=json"
    )

    _, items = _get_json(
        download_url,
        api_key,
    )

    return items


def main() -> None:
    api_key = os.getenv("BRIGHT_DATA_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing BRIGHT_DATA_API_KEY "
            "in environment"
        )

    search_input = {
        "url": "https://www.booking.com",
        "location": "Baku",
        "check_in": "2026-09-15T00:00:00.000Z",
        "check_out": "2026-09-18T00:00:00.000Z",
        "adults": 2,
        "children": 0,
        "rooms": 1,
        "country": "AZ",
        "currency": "USD",
    }

    payload = {
        "input": [
            search_input,
        ],
        "limit_per_input": 5,
    }

    url = (
        f"{API_BASE}/datasets/v3/scrape"
        f"?dataset_id={DATASET_ID}"
        f"&notify=false"
        f"&include_errors=true"
    )

    body = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=body,
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()

    status_code, response = _request_json(
        request,
        timeout=180,
    )

    # Bright Data sync mode can fall back
    # to a snapshot for longer jobs.
    if (
        isinstance(response, dict)
        and response.get("snapshot_id")
    ):
        items = _wait_for_snapshot(
            response["snapshot_id"],
            api_key,
        )
    else:
        items = response

    latency_seconds = (
        time.perf_counter() - started
    )

    print(
        "\n=== BRIGHT DATA BOOKING "
        "SEARCH BENCHMARK ==="
    )
    print("City: Baku")
    print("Requested listings: 5")
    print(f"HTTP status: {status_code}")

    if not isinstance(items, list):
        print(
            f"Unexpected response type: "
            f"{type(items)}"
        )
        print(items)
        return

    print(f"Returned listings: {len(items)}")
    print(
        f"Total retrieval latency: "
        f"{latency_seconds:.2f} sec"
    )

    print("\n=== RETURNED HOTELS ===")

    for index, item in enumerate(
        items,
        start=1,
    ):
        name = (
            item.get("name")
            or item.get("title")
            or item.get("hotel_name")
            or "<missing name>"
        )

        print(f"{index}. {name}")

    print("\n=== RAW KEYS ===")

    if items:
        print(sorted(items[0].keys()))

    _save_debug_payload(
        actor_input=search_input,
        items=items,
    )


if __name__ == "__main__":
    main()