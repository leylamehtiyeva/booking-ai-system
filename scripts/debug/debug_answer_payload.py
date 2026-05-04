from __future__ import annotations

import asyncio
import json
from app.schemas.fallback_policy import FallbackPolicy
from app.logic.answer_generation import build_user_answer
from app.logic.answer_generation_llm import generate_user_answer_with_llm
from app.logic.build_answer_payload import build_answer_payload
from app.logic.intent_router import route_intent_adk_async
from app.schemas.search_response import NormalizedSearchResponse
from app.tools.orchestrate_search_tool import orchestrate_search


USER_TEXT = (
"I want an apartment in Baku from 2026-04-08 to 2026-04-15 with satellite TV and ironing facilities."

    )


async def main():
    print("\n=== USER TEXT ===")
    print(USER_TEXT)

    intent = await route_intent_adk_async(USER_TEXT)
    intent_dict = intent.model_dump(mode="json", exclude_none=True)

    print("\n=== INTENT ===")
    print(json.dumps(intent_dict, ensure_ascii=False, indent=2))

    result = await orchestrate_search(
        user_text=USER_TEXT,
        intent=intent_dict,
        source="fixtures",
        max_items=10,
        fallback_policy=FallbackPolicy(enabled=True, top_k=0),
    )

    print("\n=== NORMALIZED SEARCH RESPONSE ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    response_obj = NormalizedSearchResponse.model_validate(result)

    payload = build_answer_payload(
        response_obj,
        latest_user_query=USER_TEXT,
        top_k=3,
    )

    print("\n=== LLM-READY ANSWER PAYLOAD ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    print("\n=== ACTIVE INTENT ===")
    print(json.dumps(payload.get("active_intent"), ensure_ascii=False, indent=2))

    print("\n=== DETERMINISTIC USER-FACING ANSWER ===")
    answer = build_user_answer(payload)
    print(answer)

    print("\n=== LLM USER-FACING ANSWER ===")
    print("\n=== Ответ закоментирован ===")

    # llm_answer = await generate_user_answer_with_llm(payload)
    # print(llm_answer)


if __name__ == "__main__":
    asyncio.run(main())