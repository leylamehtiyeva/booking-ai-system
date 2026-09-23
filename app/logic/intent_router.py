from __future__ import annotations
from app.config.llm import get_gemini_model
from app.observability.trace import RequestTrace
from app.observability.llm_usage import record_llm_call_estimated
import asyncio
import json
import os
import uuid
from typing import Optional

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.agents.intent_router_agent import IntentRoute, build_intent_router_agent
from app.logic.date_normalization import normalize_intent_dates
from app.logic.request_resolution import resolve_required_search_context
from app.schemas.query import SearchRequest



APP_NAME = "booking-ai-agent"
USER_ID = "local-user"


def _clean_filters(filters):
    if filters is None:
        return None

    cleaned = filters.model_dump(
        mode="json",
        exclude_none=True,
    )

    return cleaned or None


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



async def _route_intent_via_adk(
    user_text: str,
    *,
    trace: RequestTrace | None = None,
    step: str = "initial_intent_extraction",
) -> IntentRoute:
    _ensure_gemini_key()

    max_retries = 3
    delay_seconds = 1.0

    for attempt in range(max_retries):
        try:
            agent = build_intent_router_agent()
            session_service = InMemorySessionService()
            runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

            session_id = f"intent-{uuid.uuid4().hex[:8]}"
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

            msg = Content(role="user", parts=[Part.from_text(text=user_text)])
            cfg = RunConfig(response_modalities=["TEXT"])

            final_text: Optional[str] = None
            async for ev in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=msg,
                run_config=cfg,
            ):
                content = getattr(ev, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if t:
                            final_text = (final_text or "") + t

            if not final_text:
                raise ValueError("ADK returned empty response text")

            clean = _strip_json_fence(final_text)
            
            record_llm_call_estimated(
                trace=trace,
                step=step,
                model=get_gemini_model(),
                prompt_text=user_text,
                response_text=final_text,
                success=True,
            )
            payload = json.loads(clean)

            if payload.get("filters") is None:
                payload["filters"] = {}

            if payload.get("constraints") is None:
                payload["constraints"] = []

            if payload.get("property_types") is None:
                payload["property_types"] = []

            if payload.get("occupancy_types") is None:
                payload["occupancy_types"] = []

            return IntentRoute.model_validate(payload)

        except Exception as e:
            error_text = str(e)
            is_resource_exhausted = (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
                or "_ResourceExhaustedError" in e.__class__.__name__
            )

            if is_resource_exhausted and attempt < max_retries - 1:
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
                continue

            raise


async def route_intent_adk_async(
    user_text: str,
    *,
    trace: RequestTrace | None = None,
    step: str = "initial_intent_extraction",
) -> IntentRoute:
    return await _route_intent_via_adk(user_text, trace=trace, step=step)


def route_intent_adk(user_text: str) -> IntentRoute:
    return asyncio.run(route_intent_adk_async(user_text))


async def build_search_request_adk_async(
    user_text: str,
    *,
    trace: RequestTrace | None = None,
    step: str = "initial_intent_extraction",
) -> SearchRequest:
    intent = await route_intent_adk_async(user_text, trace=trace, step=step)
    intent = normalize_intent_dates(intent, user_text)


    resolved = resolve_required_search_context(intent)
    clean_filters = _clean_filters(intent.filters)

    req = SearchRequest(
    city=resolved.city,
    check_in=resolved.check_in,
    check_out=resolved.check_out,
    adults=intent.adults or 2,
    children=intent.children or 0,
    rooms=intent.rooms or 1,
    filters=clean_filters,
    property_types=intent.property_types or None,
    occupancy_types=intent.occupancy_types or None,
    constraints=intent.constraints,
)

    return req


def build_search_request_adk(user_text: str) -> SearchRequest:
    return asyncio.run(build_search_request_adk_async(user_text))