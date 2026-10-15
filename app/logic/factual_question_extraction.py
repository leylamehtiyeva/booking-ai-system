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
from app.llm.call_policy import LlmCallError, call_llm_stage
from app.observability.llm_usage import record_llm_call_estimated
from app.observability.trace import RequestTrace
from app.schemas.result_query import ResultQuery

APP_NAME = "booking-ai-agent"
USER_ID = "local-user"
FACTUAL_EXTRACTION_STEP = "factual_question_extraction"


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

    Runs under the shared LLM stage failure policy (app.llm.call_policy):
    raises LlmCallError for provider failures and unusable responses.
    """
    _ensure_gemini_key()
    model_name = get_gemini_model()

    async def attempt() -> ResultQuery:
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
            raise LlmCallError("empty_response", stage=FACTUAL_EXTRACTION_STEP)

        clean = _strip_json_fence(final_text)
        try:
            result_query = ResultQuery.model_validate_json(clean)
        except ValueError as exc:  # ValidationError, incl. malformed JSON
            raise LlmCallError(
                "invalid_response", stage=FACTUAL_EXTRACTION_STEP, parse_failure=True
            ) from exc

        record_llm_call_estimated(
            trace=trace,
            step=FACTUAL_EXTRACTION_STEP,
            model=model_name,
            prompt_text=user_message,
            response_text=final_text,
            success=True,
        )
        return result_query

    return await call_llm_stage(
        attempt,
        stage=FACTUAL_EXTRACTION_STEP,
        model=model_name,
        trace=trace,
    )
