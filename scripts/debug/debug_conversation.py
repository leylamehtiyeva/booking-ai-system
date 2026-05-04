from __future__ import annotations

import asyncio
import json

from app.logic.conversation_flow import handle_user_message
from app.schemas.query import SearchRequest
from app.schemas.fallback_policy import FallbackPolicy


def _print_block(title: str, payload):
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


async def main():
    print("Booking AI debug chat")
    print("Commands:")
    print("  /reset  -> reset conversation state")
    print("  /exit   -> quit")

    state: SearchRequest | None = None

    while True:
        user_message = input("\nYou: ").strip()

        if not user_message:
            continue

        if user_message.lower() == "/exit":
            print("Bye.")
            break

        if user_message.lower() == "/reset":
            state = None
            print("State reset.")
            continue

        try:
            result = await handle_user_message(
                user_message,
                previous_state=state,
                source="fixtures",
                top_n=5,
                fallback_policy=FallbackPolicy(enabled=True, top_k=5),
                max_items=10,
            )
        except Exception as e:
            _print_block("ERROR", str(e))
            continue

        _print_block("RESULT", result)

        state_payload = result.get("state")
        if state_payload:
            state = SearchRequest.model_validate(state_payload)
            _print_block("CURRENT STATE", state.model_dump(mode="json", exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())