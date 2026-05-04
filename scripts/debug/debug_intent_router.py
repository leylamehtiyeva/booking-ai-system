from __future__ import annotations

import asyncio
import json

from app.logic.intent_router import (
    build_search_request_adk_async,
    route_intent_adk_async,
)
from app.tools.orchestrate_search_tool import orchestrate_search


async def main():
    queries = [
        "I want an apartment in Baku from 2026-04-08 to 2026-04-15 with private bathroom and kitchen",
        "I want an apartment in Baku from 2026-04-08 to 2026-04-15 with private bathroom, kitchen, at least 2 bedrooms and at least 80 sqm",
    ]

    for i, q in enumerate(queries, start=1):
        print(f"\n\n######## TEST {i} ########")
        print("USER QUERY:", q)

        intent = await route_intent_adk_async(q)
        print("\n=== PARSED INTENT ===")
        print(json.dumps(intent.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))

        req = await build_search_request_adk_async(q)
        print("\nFINAL REQUEST JSON:")
        print(json.dumps(req.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))

        result = await orchestrate_search(
            q,
            intent.model_dump(mode="json", exclude_none=True),
            source="apify",
            max_items=10,
        )

        print("\n=== ORCHESTRATE RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())