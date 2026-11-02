import streamlit as st

from app.config.settings import RETRIEVAL_SOURCE


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="brand">AI Booking System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if RETRIEVAL_SOURCE == "liteapi":
        st.caption("Live data: LiteAPI sandbox · Barcelona only · one room for two adults")
    else:
        st.caption("Sample data (fixtures)")