from __future__ import annotations



import json
import os
import uuid
from typing import Optional
from app.config.llm import get_gemini_model
from app.observability.trace import RequestTrace
from app.observability.llm_usage import record_llm_call_estimated

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.agents.intent_update_agent import build_intent_update_agent
from app.logic.apply_intent_patch import apply_intent_patch
from app.logic.date_normalization import normalize_patch_dates
from app.schemas.intent_patch import SearchIntentPatch
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





def _build_update_prompt(previous_state: SearchRequest, user_message: str) -> str:
    state_json = json.dumps(
        previous_state.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )
    return f"""
Current structured search state (source of truth):
{state_json}

New user message:
{user_message}

Task:
Return ONLY a JSON patch describing the changes caused by the new user message.

Rules:
- Do NOT return the full state
- Do NOT repeat unchanged fields
- Do NOT reconstruct or restate the whole request
- Preserve all existing values unless the user explicitly changes or removes them
- If the message changes only one slot, return only that slot change
- The user may write in any language

CONSTRAINT-CENTRIC RULES:
- constraints are the main representation for meaningful user requirements
- Prefer add_constraints for new meaningful requirements
- Prefer remove_constraint_texts when the user removes a previous requirement
- Do NOT silently drop meaningful constraints
- If a requirement cannot be represented as a structured filter/property/occupancy slot, preserve it as an unresolved constraint
- Return an empty patch only when the user message truly does not change the search state

PATCH FORMAT:
- Use only add_constraints and remove_constraint_texts for user requirements
- Do not use any legacy field-centric patch format

DATES:
- If user gives one date only, set_check_in to that date and set_nights = 1
- If user says "from X for N nights", set_check_in = X and set_nights = N
- If user gives both dates, set_check_in and set_check_out
- Do not invent dates

FILTERS:
- Use set_filters only for structured numeric changes like bedrooms, bathrooms, area, price
- Do NOT rebuild the whole filters object if only one field changes

PROPERTY / OCCUPANCY:
- Use property_types / occupancy_types slots for those concepts directly
- Do not duplicate them as constraints when a dedicated slot exists

RETURN EMPTY PATCH ONLY IF:
- the message does not change the search state
- or it is not a search update at all
""".strip()


async def route_intent_update_patch_async(
    previous_state: SearchRequest,
    user_message: str,
    *,
    trace: RequestTrace | None = None,
) -> SearchIntentPatch:
    _ensure_gemini_key()
    

    agent = build_intent_update_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    session_id = f"intent-update-{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    prompt = _build_update_prompt(previous_state, user_message)
    msg = Content(role="user", parts=[Part.from_text(text=prompt)])
    cfg = RunConfig(response_modalities=["TEXT"])

    final_text: Optional[str] = None
    record_llm_call_estimated(
        trace=trace,
        step="intent_update",
        model=get_gemini_model(),
        prompt_text=prompt,
        response_text=final_text,
        success=True,
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
        raise ValueError("Intent update agent returned empty response")

    clean = _strip_json_fence(final_text)
    patch = SearchIntentPatch.model_validate_json(clean)
    return patch


def _inherit_month_from_previous_state(
    *,
    previous_state: SearchRequest,
    patch: SearchIntentPatch,
    normalized_check_in: str | None,
    normalized_check_out: str | None,
    user_text: str,
) -> tuple[str | None, str | None]:
    """
    Temporary no-op.

    Month inheritance is intentionally disabled until we introduce
    a language-agnostic way to distinguish:
    - partial date updates ("change dates to 8-12")
    - explicit month changes ("change dates to 8-12 August")

    For now, preserve normalized dates exactly as resolved upstream.
    """
    return normalized_check_in, normalized_check_out

async def update_search_state_async(
    previous_state: SearchRequest,
    user_message: str,
    *,
    trace: RequestTrace | None = None,
) -> SearchRequest:
    patch = await route_intent_update_patch_async(
        previous_state,
        user_message,
        trace=trace,
    )


    normalized_check_in, normalized_check_out = normalize_patch_dates(
        set_check_in=patch.set_check_in,
        set_check_out=patch.set_check_out,
        set_nights=patch.set_nights,
        user_text=user_message,
    )

    if (
        patch.set_check_in is not None or patch.set_check_out is not None
    ) and previous_state.check_in is not None:
        normalized_check_in, normalized_check_out = _inherit_month_from_previous_state(
            previous_state=previous_state,
            patch=patch,
            normalized_check_in=normalized_check_in,
            normalized_check_out=normalized_check_out,
            user_text=user_message,
        )

    patch = patch.model_copy(
        update={
            "set_check_in": normalized_check_in,
            "set_check_out": normalized_check_out,
        }
    )
    
    return apply_intent_patch(previous_state, patch)