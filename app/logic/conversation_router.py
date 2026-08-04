from __future__ import annotations
from app.schemas.conversation_route import (
    ConversationAction,
    RouterInput,
)
import json
import os
import uuid
from typing import Any
from app.config.llm import get_gemini_model
from app.observability.trace import RequestTrace
from app.observability.llm_usage import record_llm_call_estimated
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
import asyncio
from app.config.settings import (
    CONVERSATION_ROUTER_MAX_ATTEMPTS,
    CONVERSATION_ROUTER_RETRY_DELAY_SECONDS,
    CONVERSATION_ROUTER_TIMEOUT_SECONDS,
)
from google.genai.errors import APIError
from app.schemas.conversation_route import (
    ConversationActionDecision,
    RouterInput,
)
from collections.abc import AsyncIterator
import logging
from app.agents.conversation_router_agent import (
    CONVERSATION_ROUTER_INSTRUCTION,
    build_conversation_router_agent,
)


APP_NAME = "booking-ai-agent"
USER_ID = "local-user"
RETRYABLE_ROUTER_ERROR_CODES = frozenset(
    {
        "rate_limited",
        "provider_unavailable",
    }
)

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
        
        
def _classify_api_error(exc: APIError) -> str:
    code = getattr(exc, "code", None)

    if code == 429:
        return "rate_limited"

    if code in {401, 403}:
        return "authentication_error"

    if isinstance(code, int) and 500 <= code < 600:
        return "provider_unavailable"

    return "provider_error"

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
    router_input: RouterInput,
) -> str:
    current_search = router_input.current_search

    state_json = (
        json.dumps(
            current_search.model_dump(
                mode="json",
                exclude_none=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
        if current_search is not None
        else "null"
    )

    result_json = json.dumps(
        router_input.latest_result_context or {},
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Current search:
{state_json}

Latest shown result context:
{result_json}

Latest user message:
{router_input.user_message}
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


async def _run_router_attempt(
    *,
    runner: Runner,
    session_id: str,
    message: Content,
    run_config: RunConfig,
    prompt: str,
    model_name: str,
    trace: RequestTrace | None,
) -> ConversationActionDecision:
    final_text: str | None = None
    routing_success = False
    routing_error_code: str | None = None

    try:
        events = runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=message,
            run_config=run_config,
        )

        async with asyncio.timeout(
            CONVERSATION_ROUTER_TIMEOUT_SECONDS
        ):
            final_text = await _collect_final_response_text(
                events
            )

        if not final_text:
            routing_error_code = "empty_response"

            raise ConversationRoutingError(
                code=routing_error_code,
            )

        clean = _strip_json_fence(final_text)

        try:
            decision = (
                ConversationActionDecision.model_validate_json(
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

    except TimeoutError as exc:
        routing_error_code = "timeout"

        raise ConversationRoutingError(
            code=routing_error_code,
        ) from exc

    except APIError as exc:
        routing_error_code = _classify_api_error(exc)

        raise ConversationRoutingError(
            code=routing_error_code,
        ) from exc

    except ConversationRoutingError as exc:
        if routing_error_code is None:
            routing_error_code = exc.code

        raise

    except Exception:
        routing_error_code = "unexpected_error"
        raise

    finally:
        _record_router_llm_call(
            trace=trace,
            model=model_name,
            prompt=prompt,
            response_text=final_text,
            success=routing_success,
            error=routing_error_code,
        )

async def route_conversation_async(
    *,
    router_input: RouterInput,
    trace: RequestTrace | None = None,
) -> ConversationActionDecision:
    _ensure_gemini_key()

    agent = build_conversation_router_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    prompt = _build_router_prompt(
        router_input=router_input,
    )

    message = Content(
        role="user",
        parts=[Part.from_text(text=prompt)],
    )
    run_config = RunConfig(
        response_modalities=["TEXT"],
    )
    model_name = get_gemini_model()

    for attempt in range(
        1,
        CONVERSATION_ROUTER_MAX_ATTEMPTS + 1,
    ):
        session_id = (
            f"conversation-router-"
            f"{uuid.uuid4().hex[:8]}"
        )

        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

        try:
            return await _run_router_attempt(
                runner=runner,
                session_id=session_id,
                message=message,
                run_config=run_config,
                prompt=prompt,
                model_name=model_name,
                trace=trace,
            )

        except ConversationRoutingError as exc:
            attempts_exhausted = (
                attempt
                >= CONVERSATION_ROUTER_MAX_ATTEMPTS
            )
            error_is_retryable = (
                exc.code
                in RETRYABLE_ROUTER_ERROR_CODES
            )

            if attempts_exhausted or not error_is_retryable:
                raise

            logger.warning(
                "Retrying conversation router after %s "
                "(attempt %s/%s)",
                exc.code,
                attempt,
                CONVERSATION_ROUTER_MAX_ATTEMPTS,
            )

            await asyncio.sleep(
                CONVERSATION_ROUTER_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Conversation router attempts exhausted "
        "without a result"
    )