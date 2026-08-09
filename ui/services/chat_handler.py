from __future__ import annotations
from app.config.settings import MAX_ITEMS_HARD_CAP, TOP_N_DEFAULT, FALLBACK_TOP_K_DEFAULT, SOURCE_NAME 
import asyncio
import sys
from pathlib import Path
from typing import Any
from app.schemas.fallback_policy import FallbackPolicy
import streamlit as st
from app.observability.telemetry_logger import save_telemetry_record
from app.logic.conversation_flow import handle_user_message
from app.schemas.conversation_response import ConversationMessage
from app.logic.conversation_response_adapter import (
    build_conversation_response_input,
)
from app.logic.conversation_response_generator import (
    generate_deterministic_conversation_response,
)
from app.logic.conversation_response_llm import (
    generate_conversation_response_with_llm,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest

from ui.formatters import build_display_answer
from ui.state import append_message, get_search_state, set_search_state, get_messages

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logic.conversation_flow import handle_user_message
from app.schemas.query import SearchRequest

from ui.formatters import build_display_answer
from ui.state import append_message, get_search_state, set_search_state


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)

def build_recent_conversation_messages(
    messages: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[ConversationMessage]:
    """
    Convert Streamlit UI messages into the small conversation-history
    contract used by the response layer.

    Debug data and other UI-specific fields are intentionally excluded.
    """
    if limit <= 0:
        return []

    visible_messages: list[ConversationMessage] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        visible_messages.append(
            ConversationMessage(
                role=role,
                content=content,
            )
        )

    return visible_messages[-limit:]

def build_assistant_response(
    *,
    user_message: str,
    result: dict[str, Any],
    recent_messages: list[ConversationMessage] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    conversation_action = result.get("conversation_action")

    use_conversation_response_layer = conversation_action in {
        ConversationAction.START_SEARCH.value,
        ConversationAction.UPDATE_SEARCH.value,
        ConversationAction.GENERAL_CHAT.value,
    }

    if not use_conversation_response_layer:
        return build_display_answer(result)

    response_input = build_conversation_response_input(
    user_message=user_message,
    result=result,
    recent_messages=recent_messages,
)

    assistant_answer = generate_deterministic_conversation_response(
        response_input
    )

    # Preserve the existing answer payload for debug/observability.
    _, answer_payload = build_display_answer(result)

    return assistant_answer, answer_payload

def process_user_message(user_message: str) -> None:
    recent_messages = build_recent_conversation_messages(
        get_messages()
    )

    append_message("user", user_message)

    previous_state = None
    current_state = get_search_state()
    if current_state is not None:
        previous_state = SearchRequest.model_validate(current_state)

    with st.spinner("Thinking..."):
        result = run_async(
            handle_user_message(
                user_message=user_message,
                previous_state=previous_state,
                source=SOURCE_NAME,
                top_n=TOP_N_DEFAULT,
                fallback_policy=FallbackPolicy(enabled=True, top_k=FALLBACK_TOP_K_DEFAULT),
                max_items=MAX_ITEMS_HARD_CAP,
            )
        )
        
        telemetry_log_info = None

    if result.get("telemetry"):
        telemetry_log_info = save_telemetry_record(
            telemetry=result["telemetry"],
            user_message=user_message,
            source="fixtures",
            top_n=5,
            max_items=MAX_ITEMS_HARD_CAP,
            result_summary={
                "need_clarification": result.get("need_clarification"),
                "results_count": result.get("results_count"),
                "questions": result.get("questions"),
            },
        )

    conversation_action = result.get("conversation_action")

    use_conversation_response_layer = conversation_action in {
        ConversationAction.START_SEARCH.value,
        ConversationAction.UPDATE_SEARCH.value,
        ConversationAction.GENERAL_CHAT.value,
    }

    response_source = "direct"

    if use_conversation_response_layer:
        response_input = build_conversation_response_input(
            user_message=user_message,
            result=result,
            recent_messages=recent_messages,
        )

        generation_result = run_async(
            generate_conversation_response_with_llm(
                response_input
            )
        )

        assistant_answer = generation_result.text
        response_source = generation_result.source

        # Existing payload is still kept only for debug/observability.
        _, answer_payload = build_display_answer(result)

    else:
        assistant_answer, answer_payload = build_display_answer(result)
    debug_data = {
        "parsed_intent": result.get("parsed_intent"),
        "search_request": result.get("search_request"),
        "state_after": result.get("state"),
        "answer_payload": answer_payload,
        "conversation_response_source": response_source,
        "telemetry": result.get("telemetry"),
        "telemetry_log_info": telemetry_log_info,
    }

    append_message("assistant", assistant_answer, debug_data=debug_data)
    set_search_state(result.get("state"))