"""
Step 4 (conversation_flow.py integration) smoke trace: a real
handle_user_message() call for a LISTING_QUESTION turn, with 3 fake shown
results. Makes real LLM calls (router + factual extraction) - no prose
generation involved, this only shows the structured payload.

Run: python3 scripts/debug/listing_question_flow_smoke.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from app.logic.conversation_flow import handle_user_message
from app.schemas.listing import ListingRaw
from app.schemas.shown_result_set import ShownResult, ShownResultSet

shown_result_set = ShownResultSet(
    result_set_id="rs-smoke",
    items=[
        ShownResult(
            result_id="H1",
            listing=ListingRaw(
                id="H1",
                name="Hilton Baku",
                facilities=[{"name": "Private parking"}],
            ),
        ),
        ShownResult(
            result_id="H2",
            listing=ListingRaw(
                id="H2",
                name="Marriott Baku",
                description="No parking available on site.",
            ),
        ),
        ShownResult(
            result_id="H3",
            listing=ListingRaw(
                id="H3",
                name="Boulevard Hotel",
                description="A cozy hotel close to the seaside boulevard.",
            ),
        ),
    ],
)


async def main() -> None:
    out = await handle_user_message(
        "Which of these have parking?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    print("=== router decision ===")
    print(json.dumps(out["parsed_intent"]["router"], indent=2, ensure_ascii=False))

    print("\n=== response_type / need_clarification ===")
    print(out.get("response_type"), out.get("need_clarification"))

    print("\n=== ResultQuery ===")
    print(json.dumps(out["listing_query_result"]["query"], indent=2, ensure_ascii=False))

    print("\n=== matcher decisions per candidate ===")
    for match in out["listing_query_result"]["matches"]:
        for cr in match["constraint_results"]:
            evidence = cr["evidence"][0]["snippet"] if cr["evidence"] else None
            print(f"  {match['result_id']} -> {cr['value']} (evidence: {evidence!r})")

    print("\n=== invariants ===")
    print("shown_result_set in payload (must be False):", "shown_result_set" in out)
    print("result_scope:", out.get("result_scope"))
    print("target_result_id:", out.get("target_result_id"))


if __name__ == "__main__":
    asyncio.run(main())
