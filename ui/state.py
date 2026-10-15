import streamlit as st

from app.schemas.session import ConversationSession


def init_session_state() -> None:
    # Display transcript (UI-only: carries debug data for the expander).
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Domain state for process_turn - SearchRequest, ShownResultSet, history.
    if "conversation_session" not in st.session_state:
        st.session_state.conversation_session = ConversationSession()


def get_messages():
    return st.session_state.messages


def get_conversation_session() -> ConversationSession:
    return st.session_state.conversation_session


def set_conversation_session(session: ConversationSession) -> None:
    st.session_state.conversation_session = session


def append_message(role: str, content: str, debug_data=None) -> None:
    message = {
        "role": role,
        "content": content,
    }
    if debug_data is not None:
        message["debug_data"] = debug_data

    st.session_state.messages.append(message)
