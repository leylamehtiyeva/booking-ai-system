from __future__ import annotations

import asyncio
from app.logic.intent_router import route_intent_adk_async
from app.tools.orchestrate_search_tool import orchestrate_search
from app.schemas.fallback_policy import FallbackPolicy

USER_TEXT = (
    "Нужен рекан в Мацуяма с 9 по 10 февраля 2026 года максимум 300 долларов с горячим источником"
)

async def main():
    intent_obj = await route_intent_adk_async(USER_TEXT)
    intent = intent_obj.model_dump()

    # full pipeline apify  + structured + fallback
    out = await orchestrate_search(
        user_text=USER_TEXT,
        intent=intent,
        source="apify",
        max_items=7,       
        top_n=2,            
        fallback_policy=FallbackPolicy(enabled=True, top_k=2),
    )

    print("INTENT:", intent)
    print("\nOUTPUT:", out)

if __name__ == "__main__":
    asyncio.run(main())
