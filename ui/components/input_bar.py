import streamlit as st

from ui.services.chat_handler import process_user_message


def render_input_area() -> None:
    user_message = st.chat_input(
        "Try: apartment in Baku from April 10 to April 15 for 4 people"
    )

    if user_message and user_message.strip():
        process_user_message(user_message.strip())
        st.rerun()