from __future__ import annotations
from app.config.settings import MAX_ITEMS_HARD_CAP
import asyncio
import sys
from pathlib import Path
from typing import Any
from app.schemas.fallback_policy import FallbackPolicy
import streamlit as st
from app.observability.telemetry_logger import save_telemetry_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logic.conversation_flow import handle_user_message
from app.schemas.query import SearchRequest

from ui.formatters import build_display_answer
from ui.state import append_message, get_search_state, set_search_state


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def process_user_message(user_message: str) -> None:
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
                source="apify",
                top_n=5,
                fallback_policy=FallbackPolicy(enabled=True, top_k=5),
                max_items=MAX_ITEMS_HARD_CAP,
            )
        )
        
        telemetry_log_info = None

    if result.get("telemetry"):
        telemetry_log_info = save_telemetry_record(
            telemetry=result["telemetry"],
            user_message=user_message,
            source="apify",
            top_n=5,
            max_items=MAX_ITEMS_HARD_CAP,
            result_summary={
                "need_clarification": result.get("need_clarification"),
                "results_count": result.get("results_count"),
                "questions": result.get("questions"),
            },
        )

    assistant_answer, answer_payload = build_display_answer(result)
    debug_data = {
        "parsed_intent": result.get("parsed_intent"),
        "search_request": result.get("search_request"),
        "state_after": result.get("state"),
        "answer_payload": answer_payload,
        "telemetry": result.get("telemetry"),
        "telemetry_log_info": telemetry_log_info,
    }

    append_message("assistant", assistant_answer, debug_data=debug_data)
    set_search_state(result.get("state"))