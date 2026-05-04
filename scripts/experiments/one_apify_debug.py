from __future__ import annotations

import asyncio
from app.retrieval import get_candidates
from app.schemas.query import SearchRequest

async def main():
    req = SearchRequest(
        user_message="1 night at Nagasaki with kitchen",
        city="Nagasaki",
        check_in="2026-02-01",
        check_out="2026-02-02",
        adults=2,
        children=0,
        rooms=1,
        currency="USD",
        budget_max=None,
        must_have_fields=[],
        nice_to_have_fields=[],
        forbidden_fields=[],
        min_guest_rating=None,
        property_types=None,
    )

    listings = await get_candidates(req, max_items=5, source="apify")

    print(f"Got {len(listings)} listings\n")

    if not listings:
        return

    # Печатаем ключи первого объекта, чтобы понять где URL
    first = listings[0]
    d = first.model_dump()  # ListingRaw pydantic
    print("TOP-LEVEL KEYS:", sorted(d.keys()))

    # Часто url может лежать не в url, а в других полях — покажем кандидаты на ссылку
    url_like_keys = [k for k in d.keys() if "url" in k.lower() or "link" in k.lower() or "booking" in k.lower()]
    print("URL-LIKE KEYS:", url_like_keys)

    # И покажем их значения
    for k in url_like_keys:
        print(f"{k} = {d.get(k)}")

    # На всякий случай: краткий вывод первых 3
    print("\nFIRST 3 NAMES + URL FIELD:")
    for i, lst in enumerate(listings[:3], start=1):
        dd = lst.model_dump()
        print(f"{i}. name={dd.get('name')}")
        print(f"   url={dd.get('url')}")
        # если есть альтернативные:
        for k in ["bookingUrl", "booking_url", "link", "href", "landingPageUrl"]:
            if k in dd:
                print(f"   {k}={dd.get(k)}")

if __name__ == "__main__":
    asyncio.run(main())
