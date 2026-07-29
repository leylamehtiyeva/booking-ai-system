# evaluation/tasks/end_to_end/adapter.py

from __future__ import annotations

from typing import Any

from google.api_core.exceptions import ServerError, ServiceUnavailable, TooManyRequests
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.logic.conversation_flow import handle_user_message
from app.schemas.fallback_policy import FallbackPolicy
from evaluation.tasks.end_to_end.dataset import Decision, EndToEndEvalCase


class EndToEndPrediction(BaseModel):
    decision: Decision
    selected_listing_id: str | None = None
    selected_listing_ids: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any]


def _extract_selected_listing_ids(response: dict[str, Any]) -> list[str]:
    results = response.get("results") or []

    ids: list[str] = []

    for item in results:
        listing_id = (
            item.get("result_id")
            or item.get("id")
            or item.get("listing_id")
        )

        if listing_id:
            ids.append(str(listing_id))

    return ids


def _infer_decision(response: dict[str, Any]) -> Decision:
    results = response.get("results") or []

    if results:
        return "YES"

    if response.get("need_clarification") is True:
        questions = response.get("questions") or []
        debug_notes = response.get("debug_notes") or []

        text = " ".join(questions + debug_notes).lower()

        no_result_markers = [
            "ничего не найдено",
            "no listings remained",
            "no results",
            "nothing found",
            "no suitable listings",
        ]

        if any(marker in text for marker in no_result_markers):
            return "NO"

        return "UNCERTAIN"

    return "NO"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(
        (
            ServerError,
            ServiceUnavailable,
            TooManyRequests,
            TimeoutError,
        )
    ),
    reraise=True,
)
async def run_case(
    case: EndToEndEvalCase,
    *,
    source: str = "fixtures",
    top_n: int = 5,
    max_items: int = 40,
    fallback_policy: FallbackPolicy | None = None,
) -> EndToEndPrediction:
    if fallback_policy is None:
        fallback_policy = FallbackPolicy(
            enabled=True,
            top_k=5,
            must_only=True,
        )

    response = await handle_user_message(
        user_message=case.user_query,
        previous_state=None,
        source=source,
        top_n=top_n,
        max_items=max_items,
        fallback_policy=fallback_policy,
    )

    selected_ids = _extract_selected_listing_ids(response)
    decision = _infer_decision(response)

    return EndToEndPrediction(
        decision=decision,
        selected_listing_id=selected_ids[0] if selected_ids else None,
        selected_listing_ids=selected_ids,
        raw_response=response,
    )