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

APP_NAME = "booking-ai-agent"
USER_ID = "local-user"


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
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

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

    msg = Content(role="user", parts=[Part.from_text(text=prompt)])
    cfg = RunConfig(response_modalities=["TEXT"])

    final_text: Optional[str] = None
    record_llm_call_estimated(
        trace=trace,
        step="conversation_routing",
        model=get_gemini_model(),
        prompt_text=prompt,
        response_text=final_text,
        success=bool(final_text),
    )
    
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
        return ConversationRouteDecision(
            route="search_update",
            reason="fallback: empty router response",
        )

    clean = _strip_json_fence(final_text)

    try:
        return ConversationRouteDecision.model_validate_json(clean)
    except Exception:
        return ConversationRouteDecision(
            route="search_update",
            reason=f"fallback: invalid router output: {clean[:200]}",
        )