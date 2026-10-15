"""
Load .env before any app module is imported - exactly what
ui/streamlit_app.py does - so the evaluation CLI runs with the same
configuration as the app. Several settings are read at import time
(ROUTER_LLM_PROFILE in app.config.settings, the FallbackPolicy.model
default), so loading .env later silently produces a different model mix.
Import this module first in the CLI entry point.
"""

from dotenv import load_dotenv

load_dotenv()
