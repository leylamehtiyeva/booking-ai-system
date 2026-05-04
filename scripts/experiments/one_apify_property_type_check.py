from __future__ import annotations
import asyncio
from app.retrieval import get_candidates
from app.schemas.query import SearchRequest

async def main():
    req = SearchRequest(
        user_message="Ryokan in Matsuyama",
        city="Matsuyama",
        check_in="2026-02-09",
        check_out="2026-02-10",
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

    listings = await get_candidates(req, max_items=7, source="apify")
    print(f"Got {len(listings)} listings\n")

    for i, lst in enumerate(listings[:5], start=1):
        d = lst.model_dump()
        print(f"{i}. {d.get('name')}")
        print("   property_type:", d.get("property_type"))
        print("   type:", d.get("type"))
        print("   url:", d.get("url"))
        print()

if __name__ == "__main__":
    asyncio.run(main())
