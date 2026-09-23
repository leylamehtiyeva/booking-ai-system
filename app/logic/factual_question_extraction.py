from __future__ import annotations

import os
import uuid
from typing import Optional

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.agents.factual_question_agent import build_factual_question_agent
from app.config.llm import get_gemini_model
from app.observability.llm_usage import record_llm_call_estimated
from app.observability.trace import RequestTrace
from app.schemas.result_query import ResultQuery

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


async def extract_factual_constraints_async(
    user_message: str,
    *,
    trace: RequestTrace | None = None,
) -> ResultQuery:
    """
    user_message -> ResultQuery (transient, never persisted). Structurally
    mirrors route_intent_adk_async/route_intent_update_patch_async (same
    ADK Runner/session pattern, same JSON-fence stripping), but this
    result must never reach apply_intent_patch/set_search_state - see
    ResultQuery's own docstring.

    One deliberate deviation from intent_update.py's telemetry placement:
    that file calls record_llm_call_estimated() BEFORE the response loop
    runs (response_text is always None, success is always True at that
    point regardless of outcome). That looks like a pre-existing
    telemetry bug there, not a pattern worth propagating - here the call
    is made after the response is collected, with the real text and a
    real success/failure outcome.
    """
    _ensure_gemini_key()

    agent = build_factual_question_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    session_id = f"factual-question-{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    msg = Content(role="user", parts=[Part.from_text(text=user_message)])
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
        record_llm_call_estimated(
            trace=trace,
            step="factual_question_extraction",
            model=get_gemini_model(),
            prompt_text=user_message,
            response_text=None,
            success=False,
            error="empty_response",
        )
        raise ValueError("Factual question agent returned empty response")

    record_llm_call_estimated(
        trace=trace,
        step="factual_question_extraction",
        model=get_gemini_model(),
        prompt_text=user_message,
        response_text=final_text,
        success=True,
    )

    clean = _strip_json_fence(final_text)
    return ResultQuery.model_validate_json(clean)
