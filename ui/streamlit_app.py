from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.chat import render_messages
from ui.components.empty_state import render_empty_state
from ui.components.header import render_header
from ui.components.input_bar import render_input_area
from ui.state import get_messages, init_session_state
from ui.styles import apply_styles


st.set_page_config(
    page_title="AI Booking System",
    page_icon="🖤",
    layout="wide",
)


def main() -> None:
    apply_styles()
    init_session_state()

    render_header()

    if get_messages():
        render_messages()
    else:
        render_empty_state()

    st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)
    render_input_area()
    
    


if __name__ == "__main__":
    main()