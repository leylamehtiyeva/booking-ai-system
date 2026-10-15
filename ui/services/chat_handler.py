"""
Streamlit adapter over the application service: read the session from
st.session_state, run one turn through process_turn, store the updated
session, and append what to display. No domain or response-generation
policy lives here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import streamlit as st

from app.schemas.turn import TurnConfig, TurnResult
from app.services.conversation_service import process_turn
from ui.state import (
    append_message,
    get_conversation_session,
    set_conversation_session,
)

logger = logging.getLogger(__name__)

STREAMLIT_TURN_CONFIG = TurnConfig(include_debug=True)

UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while processing your message. "
    "Your current search has not been changed. Please try again."
)


def process_user_message(user_message: str) -> None:
    append_message("user", user_message)

    session = get_conversation_session()

    try:
        with st.spinner("Thinking..."):
            turn = asyncio.run(
                process_turn(session, user_message, config=STREAMLIT_TURN_CONFIG)
            )
    except Exception:
        # UI boundary for unexpected (programming) errors: expected
        # provider/retrieval failures already come back as typed turns.
        # Nothing was committed - the stored session stays as it was.
        logger.exception("Unexpected error while processing a chat turn")
        append_message("assistant", UNEXPECTED_ERROR_MESSAGE)
        return

    set_conversation_session(turn.session)
    append_message("assistant", turn.assistant_message, debug_data=_debug_data(turn))


def _debug_data(turn: TurnResult) -> dict[str, Any] | None:
    if turn.debug is None:
        return None

    return {
        "parsed_intent": turn.debug.parsed_intent,
        "search_request": turn.debug.search_request,
        "state_after": turn.debug.state_after,
        "answer_payload": turn.debug.answer_payload,
        "conversation_response_source": turn.inspector.response_source,
        "telemetry": turn.debug.telemetry,
        "telemetry_log_info": turn.debug.telemetry_log_info,
    }
