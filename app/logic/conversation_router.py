from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional
from app.config.llm import get_gemini_model
from app.observability.trace import RequestTrace
from app.observability.llm_usage import record_llm_call_estimated
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.agents.conversation_router_agent import build_conversation_router_agent
from app.schemas.conversation_route import ConversationRouteDecision
from app.schemas.query import SearchRequest
from collections.abc import AsyncIterator
import logging
from app.agents.conversation_router_agent import (
    CONVERSATION_ROUTER_INSTRUCTION,
    build_conversation_router_agent,
)

APP_NAME = "booking-ai-agent"
USER_ID = "local-user"

logger = logging.getLogger(__name__)



class ConversationRoutingError(RuntimeError):
    """
    Raised when the conversation router cannot produce
    a valid domain routing decision.

    This is an infrastructure/runtime failure,
    not a user intent.
    """

    def __init__(
        self,
        code: str,
        *,
        internal_detail: str | None = None,
    ) -> None:
        super().__init__(f"Conversation routing failed: {code}")
        self.code = code
        self.internal_detail = internal_detail

def _extract_event_text(event: Any) -> str | None:
    """
    Extract textual parts from one ADK event.

    Parts inside one event belong to the same content object,
    so they may be joined together.
    """
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


async def _collect_final_response_text(
    events: AsyncIterator[Any],
) -> str | None:
    """
    Consume the whole ADK event stream and keep only
    non-empty final response text.
    """
    final_text: str | None = None

    async for event in events:
        if not event.is_final_response():
            continue

        event_text = _extract_event_text(event)

        if event_text:
            final_text = event_text

    return final_text


def _ensure_gemini_key() -> None:
    if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    return t


def _build_router_prompt(
    *,
    user_message: str,
    previous_state: SearchRequest | None,
    latest_result_context: dict[str, Any] | None = None,
) -> str:
    state_json = (
        json.dumps(previous_state.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2)
        if previous_state is not None
        else "null"
    )

    result_json = json.dumps(latest_result_context or {}, ensure_ascii=False, indent=2)

    return f"""
Current search state:
{state_json}

Latest shown result context:
{result_json}

Latest user message:
{user_message}
""".strip()


def _record_router_llm_call(
    *,
    trace: RequestTrace | None,
    model: str,
    prompt: str,
    response_text: str | None,
    success: bool,
    error: str | None,
) -> None:
    """
    Record one completed router LLM attempt.

    Telemetry failure must not change the business result
    or hide the original routing error.
    """
    estimated_input_text = (
        f"System instruction:\n{CONVERSATION_ROUTER_INSTRUCTION}\n\n"
        f"User input:\n{prompt}"
    )

    try:
        record_llm_call_estimated(
            trace=trace,
            step="conversation_routing",
            model=model,
            prompt_text=estimated_input_text,
            response_text=response_text,
            success=success,
            error=error,
        )
    except Exception:
        logger.exception(
            "Failed to record conversation router telemetry"
        )


async def route_conversation_async(
    *,
    user_message: str,
    previous_state: SearchRequest | None,
    latest_result_context: dict[str, Any] | None = None,
    trace: RequestTrace | None = None,
) -> ConversationRouteDecision:
    _ensure_gemini_key()

    agent = build_conversation_router_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session_id = f"conversation-router-{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    prompt = _build_router_prompt(
        user_message=user_message,
        previous_state=previous_state,
        latest_result_context=latest_result_context,
    )

    msg = Content(
        role="user",
        parts=[Part.from_text(text=prompt)],
    )
    cfg = RunConfig(response_modalities=["TEXT"])

    model_name = get_gemini_model()

    final_text: str | None = None
    routing_success = False
    routing_error_code: str | None = None

    try:
        events = runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=msg,
            run_config=cfg,
        )

        final_text = await _collect_final_response_text(events)

        if not final_text:
            routing_error_code = "empty_response"

            raise ConversationRoutingError(
                code=routing_error_code,
            )

        clean = _strip_json_fence(final_text)

        try:
            decision = (
                ConversationRouteDecision.model_validate_json(
                    clean
                )
            )
        except Exception as exc:
            routing_error_code = "invalid_response"

            raise ConversationRoutingError(
                code=routing_error_code,
                internal_detail=clean[:500],
            ) from exc

        routing_success = True

        return decision

    except ConversationRoutingError as exc:
        if routing_error_code is None:
            routing_error_code = exc.code

        raise

    except Exception as exc:
        routing_error_code = "provider_error"

        raise ConversationRoutingError(
            code=routing_error_code,
        ) from exc

    finally:
        _record_router_llm_call(
            trace=trace,
            model=model_name,
            prompt=prompt,
            response_text=final_text,
            success=routing_success,
            error=routing_error_code,
        )