from __future__ import annotations

import asyncio
import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from app.schemas.fallback_policy import FallbackPolicy

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logic.answer_generation import build_user_answer
from app.logic.build_answer_payload import build_answer_payload
from app.logic.conversation_flow import handle_user_message
from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse


st.set_page_config(
    page_title="AI Booking",
    page_icon="🖤",
    layout="wide",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b0b0c;
            color: white;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
        }

        .brand {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            letter-spacing: -0.02em;
        }

        .hero-wrap {
            min-height: 42vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 650;
            line-height: 1.1;
            color: white;
            margin-bottom: 0.8rem;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            line-height: 1.6;
            color: #b3b3b3;
            max-width: 760px;
            margin: 0 auto;
        }

        .chat-container {
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }

        .chat-message {
            padding: 16px 18px;
            border-radius: 18px;
            margin-bottom: 12px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 1rem;
        }

        .user-message {
            background: #151517;
            border: 1px solid #2b2b31;
        }

        .assistant-message {
            background: #111214;
            border: 1px solid #23242a;
        }

        .message-role {
            display: inline-block;
            margin-bottom: 6px;
            font-size: 0.92rem;
            font-weight: 700;
            color: #ffffff;
        }

        .input-shell {
            margin-top: 1.2rem;
            padding: 10px;
            border-radius: 26px;
            background: #1a1a1d;
            border: 1px solid #2d2d33;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        }

        .stTextInput > div > div > input {
            background: transparent !important;
            color: white !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 1.08rem !important;
            padding: 0.95rem 0.6rem !important;
        }

        .stTextInput > label {
            display: none !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 999px !important;
            border: none !important;
            background: #f3f3f3 !important;
            color: #111111 !important;
            font-weight: 700 !important;
            min-height: 52px !important;
            margin-top: 1px !important;
        }

        .stButton > button:hover {
            background: #ffffff !important;
            color: #000000 !important;
        }

        .helper-text {
            text-align: center;
            color: #8f8f95;
            font-size: 0.96rem;
            margin-top: 0.8rem;
        }

        .small-muted {
            color: #9a9aa1;
            font-size: 0.9rem;
        }

        div[data-testid="stSpinner"] > div {
            border-top-color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "search_state" not in st.session_state:
        st.session_state.search_state = None

    if "input_value" not in st.session_state:
        st.session_state.input_value = ""


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def build_display_answer(result: dict[str, Any]) -> str:
    """
    Convert structured backend result into final user-facing text.
    Reuses the project's existing answer generation pipeline.
    """
    if result.get("need_clarification"):
        payload = {
            "need_clarification": True,
            "questions": result.get("questions", []),
            "request_summary": None,
            "top_results": [],
            "results_count": 0,
            "active_intent": result.get("state", {}),
        }
        return build_user_answer(payload)

    normalized = NormalizedSearchResponse.model_validate(result)
    payload = build_answer_payload(
        normalized,
        latest_user_query=None,
        top_k=3,
    )
    return build_user_answer(payload)


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">AI Booking</div>
            <div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <div>
                <div class="hero-title">Find your perfect stay</div>
                <div class="hero-subtitle">
                    Apartments, hotels, or unique places — just describe what you want,
                    and I’ll help you narrow it down.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_messages() -> None:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        role = message["role"]
        raw_content = message["content"]

        role_title = "You" if role == "user" else "AI Booking"
        css_class = "user-message" if role == "user" else "assistant-message"

        safe_content = html.escape(raw_content).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="chat-message {css_class}">
                <div class="message-role">{role_title}</div>
                <div>{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def process_user_message(user_message: str) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    previous_state = None
    if st.session_state.search_state is not None:
        previous_state = SearchRequest.model_validate(st.session_state.search_state)

    with st.spinner("Thinking..."):
        result = run_async(
            handle_user_message(
                user_message=user_message,
                previous_state=previous_state,
                source="fixtures",
                top_n=5,
                fallback_policy=FallbackPolicy(enabled=True, top_k=5),
                max_items=10,
            )
        )

    assistant_answer = build_display_answer(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_answer,
        }
    )

    st.session_state.search_state = result.get("state")
    st.session_state.input_value = ""


def render_input_area() -> None:
    st.markdown('<div class="input-shell">', unsafe_allow_html=True)

    col1, col2 = st.columns([8, 1])

    with col1:
        user_message = st.text_input(
            label="Message",
            placeholder="Where would you like to stay?",
            key="input_value",
            label_visibility="collapsed",
        )

    with col2:
        send_clicked = st.button("Send", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="helper-text">
                Try: apartment in Baku from April 10 to April 15 for 4 people
            </div>
            """,
            unsafe_allow_html=True,
        )

    if send_clicked and user_message.strip():
        process_user_message(user_message.strip())
        st.rerun()


def main() -> None:
    apply_styles()
    init_session_state()

    render_header()

    if st.session_state.messages:
        render_messages()
    else:
        render_empty_state()

    render_input_area()


if __name__ == "__main__":
    main()