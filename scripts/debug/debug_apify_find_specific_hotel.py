from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TARGET_SLUG = "ryokan-sawaya-honten"
TARGET_NAME = "Ryokan Sawaya Honten"


def post_json_sync(url: str, payload: dict, timeout: int = 300):
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
        return json.loads(resp.read().decode("utf-8"))


def main():
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_TOKEN")

    actor = os.getenv("APIFY_BOOKING_ACTOR", "voyager~booking-scraper")
    api_base = os.getenv("APIFY_BASE_URL", "https://api.apify.com")

    actor_input = {
        "search": "Ryokan Sawaya Honten Kyoto",
        "currency": "USD",
        "language": "en-gb",
        "maxItems": 20,
        "checkIn": "2026-05-20",
        "checkOut": "2026-05-22",
        "adults": 2,
        "children": 0,
        "rooms": 1,
    }

    url = (
        f"{api_base}/v2/acts/{actor}/run-sync-get-dataset-items"
        f"?token={token}&format=json&clean=true&timeout=300"
    )

    print("APIFY ACTOR INPUT:")
    print(json.dumps(actor_input, ensure_ascii=False, indent=2))

    try:
        items = post_json_sync(url, actor_input, timeout=300)
    except HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Apify HTTPError {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Apify URLError: {e}") from e

    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "debug_ryokan_sawaya_all_results.json"
    all_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nReturned items: {len(items)}")
    print(f"Saved all results to: {all_path}")

    target = None

    for item in items:
        name = str(item.get("name", "")).lower()
        url_value = str(item.get("url", "")).lower()

        if TARGET_SLUG in url_value or TARGET_NAME.lower() in name:
            target = item
            break

    if not target:
        print("\nTarget hotel was NOT found.")
        print("Returned hotel names:")
        for i, item in enumerate(items, start=1):
            print(f"{i}. {item.get('name')} | type={item.get('type')} | url={item.get('url')}")
        return

    target_path = output_dir / "debug_ryokan_sawaya_honten.json"
    target_path.write_text(
        json.dumps(target, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFOUND TARGET HOTEL.")
    print(f"Saved target JSON to: {target_path}")
    print("name:", target.get("name"))
    print("type:", target.get("type"))
    print("price:", target.get("price"))
    print("currency:", target.get("currency"))
    print("url:", target.get("url"))


if __name__ == "__main__":
    main()