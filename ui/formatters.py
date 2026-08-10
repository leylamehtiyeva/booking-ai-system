from __future__ import annotations

from typing import Any

from app.logic.answer_generation import build_user_answer
from app.logic.build_answer_payload import build_answer_payload
from app.schemas.search_response import NormalizedSearchResponse
from typing import Any


def build_result_links(
    answer_payload: dict[str, Any] | None,
) -> str:
    if not answer_payload:
        return ""

    top_results = answer_payload.get("top_results") or []

    links: list[str] = []

    for position, item in enumerate(top_results, start=1):
        title = item.get("title")
        url = item.get("url")

        if not title or not url:
            continue

        links.append(
            f"{position}. [{title}]({url})"
        )

    return "\n".join(links)

_DIRECT_RESPONSE_TYPES = {
    "other",
    "listing_question",
    "routing_unavailable",
}


def build_display_answer(
    result: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:

    if result.get("need_clarification"):
        payload = {
            "need_clarification": True,
            "questions": result.get("questions", []),
            "request_summary": None,
            "top_results": [],
            "results_count": 0,
            "active_intent": result.get(
                "active_intent",
                result.get("state", {}),
            ),
            "debug_notes": result.get("debug_notes", []),
        }
        return build_user_answer(payload), payload

    if result.get("response_type") in _DIRECT_RESPONSE_TYPES:
        answer = result.get("answer")

        if isinstance(answer, str) and answer.strip():
            return answer, None

        return (
            "I couldn't prepare a response right now. Please try again.",
            None,
        )

    normalized = NormalizedSearchResponse.model_validate(result)

    payload = build_answer_payload(
        normalized,
        latest_user_query=None,
        top_k=3,
    )

    return build_user_answer(payload), payload