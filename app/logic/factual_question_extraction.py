from __future__ import annotations

import json
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


def _build_factual_question_message(
    user_message: str,
    original_question: str | None,
) -> str:
    """
    The extractor input. Without original_question it is the user's
    message verbatim (unchanged behaviour). With it, the turn answers a
    clarification about an earlier question: both texts are passed as
    separately labelled inputs, never concatenated, so the facts of the
    original question are kept and the reply can add more.
    """
    if original_question is None:
        return user_message

    return f"""
The user asked a question that needed clarification, then answered the
clarification. Return every fact the user is asking about across BOTH
messages: the facts of the original question, plus any facts the reply
adds. A reply that only says how to handle the question (for example, to
look at the shown results) adds no facts. Return each fact once.

Original question:
{json.dumps(original_question, ensure_ascii=False)}

Clarification reply:
{json.dumps(user_message, ensure_ascii=False)}
""".strip()


async def extract_factual_constraints_async(
    user_message: str,
    *,
    trace: RequestTrace | None = None,
    original_question: str | None = None,
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
    message_text = _build_factual_question_message(user_message, original_question)

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

        msg = Content(role="user", parts=[Part.from_text(text=message_text)])
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
            prompt_text=message_text,
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
