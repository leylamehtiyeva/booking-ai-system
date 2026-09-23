"""
Smoke check: does the current ConversationActionDecision structured
output schema (decision_status/clarification_reason/result_scope/
target_result_id, all required with sentinel defaults - no Optional
anywhere) actually pass through BOTH provider profiles' structured
output validators?

One real call per profile. Makes real LLM API calls - run only when you
intend to spend the corresponding quota/cost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.logic.conversation_router import (
    ConversationRoutingError,
    route_conversation_async,
)
from app.observability.trace import RequestTrace
from app.schemas.conversation_route import RouterInput

PROFILES = ["gemini_default", "groq_gpt_oss_20b"]


async def smoke_one(profile: str) -> None:
    router_input = RouterInput(
        user_message="Only with parking",
        current_search=None,
        latest_result_context={
            "has_shown_results": True,
            "shown_results": [
                {"position": 1, "result_id": "H1", "title": "Hilton Baku"},
                {"position": 2, "result_id": "H2", "title": "Marriott Baku"},
            ],
        },
    )

    print("=" * 78)
    print(f"profile: {profile}")

    try:
        decision = await route_conversation_async(
            router_input=router_input,
            trace=RequestTrace(),
            llm_profile_name=profile,
        )
    except ConversationRoutingError as exc:
        print(f"FAILED: {exc.code} ({exc.internal_detail})")
        return
    except Exception as exc:  # schema-level provider errors surface here
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return

    print("OK:", decision.model_dump(mode="json"))


async def main() -> None:
    for profile in PROFILES:
        await smoke_one(profile)


if __name__ == "__main__":
    asyncio.run(main())
