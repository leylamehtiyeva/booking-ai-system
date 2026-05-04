from __future__ import annotations

import asyncio
from app.tools.orchestrate_search_tool import orchestrate_search
from app.schemas.fallback_policy import FallbackPolicy

async def main():
    intent = {
        "city": "Baku",
        "check_in": "2026-04-08",
        "check_out": "2026-04-15",
        "must_have_fields": ["private_bathroom", "kitchen"],
        "nice_to_have_fields": [],
        "unknown_requests": ["2 bedrooms", "city center", "80 sqm"],
    }

    out = await orchestrate_search(
        user_text="Apartment in Baku city center with 2 bedrooms, 80 sqm, April 8-15",
        intent=intent,
        top_n=3,
        fallback_policy=FallbackPolicy(enabled=True, top_k=5),
        max_items=5,
        source="apify",
    )
    print(out)

if __name__ == "__main__":
    asyncio.run(main())
