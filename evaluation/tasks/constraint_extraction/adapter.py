from __future__ import annotations

from typing import Any

from app.config.clock import use_reference_date
from app.logic.intent_router import route_intent_adk_async
from evaluation.core.reference_date import EVAL_REFERENCE_DATE


async def run_case(user_message: str) -> dict[str, Any]:
    with use_reference_date(EVAL_REFERENCE_DATE):
        intent = await route_intent_adk_async(user_message)
    return intent.model_dump(mode="json", exclude_none=True)
