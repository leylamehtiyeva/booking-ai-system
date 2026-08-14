from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.agents.conversation_response_agent import (
    build_conversation_response_agent,
)
from app.config.llm import get_llm_profile
from app.config.settings import CONVERSATION_RESPONSE_LLM_PROFILE
from app.llm.model_factory import build_adk_model
from app.logic.build_answer_payload import build_answer_payload
from app.logic.conversation_response_generator import (
    generate_deterministic_conversation_response,
)
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)

from dataclasses import dataclass
from typing import Literal

from contextlib import nullcontext

from app.observability.llm_usage import (
    record_llm_call_estimated,
    record_llm_call_from_response,
)
from app.observability.trace import RequestTrace


APP_NAME = "booking-ai-agent"
USER_ID = "local-user"

CONVERSATION_RESPONSE_TIMEOUT_SECONDS = 10.0

@dataclass(frozen=True)
class ConversationResponseGenerationResult:
    text: str
    source: Literal["llm", "deterministic_fallback"]


def _extract_event_text(event: Any) -> str | None:
    content = getattr(event, "content", None)

    if content is None:
        return None

    parts = getattr(content, "parts", None)

    if not parts:
        return None

    text_parts: list[str] = []

    for part in parts:
        text = getattr(part, "text", None)

        if isinstance(text, str) and text:
            text_parts.append(text)

    if not text_parts:
        return None

    return "".join(text_parts)

async def _collect_final_response(
    events: AsyncIterator[Any],
) -> tuple[str | None, Any | None]:
    final_text: str | None = None
    usage_event: Any | None = None

    async for event in events:
        if getattr(event, "usage_metadata", None) is not None:
            usage_event = event

        if not event.is_final_response():
            continue

        event_text = _extract_event_text(event)

        if event_text:
            final_text = event_text

    return final_text, usage_event


def _build_llm_payload(
    response_input: ConversationResponseInput,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "current_user_message": response_input.user_message,
        "action": response_input.action.value,
        "recent_messages": [
            message.model_dump(mode="json")
            for message in response_input.recent_messages
        ],
        "current_search": (
            response_input.current_search.model_dump(
                mode="json",
                exclude_none=True,
            )
            if response_input.current_search is not None
            else None
        ),
    }

    outcome = response_input.outcome

    if isinstance(
        outcome,
        GeneralChatConversationOutcome,
    ):
        payload["outcome"] = {
            "kind": "general_chat",
        }

        return payload

    if isinstance(
        outcome,
        ClarificationConversationOutcome,
    ):
        payload["outcome"] = {
            "kind": "clarification",
            "questions": outcome.questions,
        }

        return payload

    if isinstance(
        outcome,
        SearchConversationOutcome,
    ):
        answer_payload = build_answer_payload(
            outcome.search_response,
            latest_user_query=None,
            top_k=3,
        )

        compact_results: list[dict[str, Any]] = []

        for result in answer_payload.get(
            "top_results",
            [],
        ):
            compact_results.append(
                {
                    "result_id": result.get("result_id"),
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "price_summary": result.get(
                        "price_summary"
                    ),
                    "budget_summary": result.get(
                        "budget_summary"
                    ),
                    "key_facts": result.get("key_facts"),
                    "answer_explanation": result.get(
                        "answer_explanation"
                    ),
                    "selection_reasons": result.get(
                        "selection_reasons"
                    ),
                }
            )

        payload["outcome"] = {
            "kind": "search",
            "status": answer_payload.get(
                "search_status"
            ),
            "results_count": answer_payload.get(
                "results_count",
                0,
            ),
            "results": compact_results,
        }

        return payload

    raise TypeError(
        "Unsupported conversation outcome: "
        f"{type(outcome).__name__}"
    )
    
    
def _build_llm_prompt(
    response_input: ConversationResponseInput,
) -> str:
    payload = _build_llm_payload(
        response_input
    )

    return (
        "Write the next assistant message using only the "
        "structured context below.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    
    
async def generate_conversation_response_with_llm(
    response_input: ConversationResponseInput,
    *,
    llm_profile_name: str | None = None,
    trace: RequestTrace | None = None,
) -> ConversationResponseGenerationResult:
    deterministic_fallback = (
        generate_deterministic_conversation_response(
            response_input
        )
    )

    prompt = _build_llm_prompt(response_input)

    model_name = (
        llm_profile_name
        or CONVERSATION_RESPONSE_LLM_PROFILE
    )

    usage_event: Any | None = None
    response_text: str | None = None

    step_context = (
        trace.step("conversation_response_generation")
        if trace is not None
        else nullcontext()
    )

    with step_context:
        try:
            model_profile = get_llm_profile(
                llm_profile_name
                or CONVERSATION_RESPONSE_LLM_PROFILE
            )

            model_name = model_profile.model

            model = build_adk_model(model_profile)

            agent = build_conversation_response_agent(
                model=model
            )

            session_service = InMemorySessionService()

            runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=session_service,
            )

            session_id = (
                "conversation-response-"
                f"{uuid.uuid4().hex[:8]}"
            )

            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

            message = Content(
                role="user",
                parts=[
                    Part.from_text(text=prompt)
                ],
            )

            run_config = RunConfig(
                response_modalities=["TEXT"],
            )

            events = runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=message,
                run_config=run_config,
            )

            async with asyncio.timeout(
                CONVERSATION_RESPONSE_TIMEOUT_SECONDS
            ):
                response_text, usage_event = (
                    await _collect_final_response(events)
                )

            response_text = (
                response_text.strip()
                if response_text
                else None
            )

            if not response_text:
                record_llm_call_estimated(
                    trace=trace,
                    step="conversation_response_generation",
                    model=model_name,
                    prompt_text=prompt,
                    response_text=None,
                    success=False,
                    error="empty_response",
                )

                return ConversationResponseGenerationResult(
                    text=deterministic_fallback,
                    source="deterministic_fallback",
                )

            if usage_event is not None:
                record_llm_call_from_response(
                    trace=trace,
                    step="conversation_response_generation",
                    model=model_name,
                    response=usage_event,
                    success=True,
                )
            else:
                record_llm_call_estimated(
                    trace=trace,
                    step="conversation_response_generation",
                    model=model_name,
                    prompt_text=prompt,
                    response_text=response_text,
                    success=True,
                )

            return ConversationResponseGenerationResult(
                text=response_text,
                source="llm",
            )

        except Exception as exc:
            record_llm_call_estimated(
                trace=trace,
                step="conversation_response_generation",
                model=model_name,
                prompt_text=prompt,
                response_text=response_text,
                success=False,
                error=str(exc),
            )

            return ConversationResponseGenerationResult(
                text=deterministic_fallback,
                source="deterministic_fallback",
            )