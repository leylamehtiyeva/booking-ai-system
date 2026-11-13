"""
Runtime wiring of the conversation router (offline, no model call):

- with no environment override the router profile is gemini_default and
  resolves to gemini-2.5-flash through the Google ADK Gemini adapter;
- route_conversation_async builds its model from that profile;
- the application runtime does not import the Jev experiment;
- FastAPI and Streamlit run with the same discourse context default, and the
  API applies .env values before its settings are read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from google.adk.models.google_llm import Gemini

from app.config import settings
from app.config.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_PROFILE_NAME,
    AdkModelAdapter,
    LlmProvider,
    get_llm_profile,
)
from app.llm.model_factory import build_adk_model
from app.logic import conversation_router
from app.schemas.conversation_route import RouterInput
from app.schemas.turn import TurnConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRS = ("app", "ui")


def test_default_router_profile_is_gemini_2_5_flash():
    # conftest removes ROUTER_LLM_PROFILE / DEFAULT_LLM_PROFILE / GEMINI_MODEL,
    # so these are the code defaults a deployment gets without overrides.
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    assert settings.DEFAULT_LLM_PROFILE == DEFAULT_LLM_PROFILE_NAME == "gemini_default"
    assert settings.ROUTER_LLM_PROFILE == "gemini_default"

    profile = get_llm_profile(settings.ROUTER_LLM_PROFILE)
    assert profile.provider == LlmProvider.GOOGLE
    assert profile.adapter == AdkModelAdapter.GEMINI
    assert profile.model == "gemini-2.5-flash"

    model = build_adk_model(profile)
    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"


def test_gemini_model_env_overrides_the_default_profile(monkeypatch):
    # GEMINI_MODEL is read when the profile is resolved, so it changes the
    # router model (and every other gemini_default stage) at runtime.
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    assert get_llm_profile("gemini_default").model == "gemini-other"


async def test_router_builds_its_model_from_the_router_profile(monkeypatch):
    built: list = []

    class Stop(Exception):
        pass

    def fake_build(profile):
        built.append(profile)
        raise Stop

    monkeypatch.setattr(conversation_router, "build_adk_model", fake_build)

    with pytest.raises(Stop):
        await conversation_router.route_conversation_async(
            router_input=RouterInput(user_message="Hotel in Rome"),
        )

    assert [p.model for p in built] == ["gemini-2.5-flash"]
    assert built[0].provider == LlmProvider.GOOGLE


def test_runtime_does_not_import_the_jev_experiment():
    import app.api.main  # noqa: F401
    import app.services.conversation_service  # noqa: F401
    import ui.services.chat_handler  # noqa: F401

    assert not [name for name in sys.modules if "jev" in name.lower()]

    mentions = [
        str(path.relative_to(PROJECT_ROOT))
        for directory in RUNTIME_DIRS
        for path in (PROJECT_ROOT / directory).rglob("*.py")
        if "jev" in path.read_text(encoding="utf-8").lower()
    ]
    assert mentions == []


def test_api_and_streamlit_share_the_discourse_context_default():
    from app.api.main import _default_turn_config
    from ui.services.chat_handler import STREAMLIT_TURN_CONFIG

    assert settings.DISCOURSE_CONTEXT_ENABLED is True
    assert TurnConfig().discourse_context is True
    assert _default_turn_config().discourse_context is True
    assert STREAMLIT_TURN_CONFIG.discourse_context is True


# Runs in a fresh interpreter: network blocked, and load_dotenv replaced by a
# stand-in that "loads" BOOKING_DISCOURSE_CONTEXT=0 - no real .env is read.
_API_DOTENV_PROBE = """
import os, socket
def _blocked(*a, **k):
    raise OSError("network blocked")
socket.getaddrinfo = _blocked
socket.create_connection = _blocked

import dotenv, dotenv.main
def _fake_load_dotenv(*args, **kwargs):
    os.environ.setdefault("BOOKING_DISCOURSE_CONTEXT", "0")
    return True
dotenv.load_dotenv = _fake_load_dotenv
dotenv.main.load_dotenv = _fake_load_dotenv

import app.api.main as api
print(api.app.state.turn_config.discourse_context)
"""


def test_api_applies_dotenv_before_reading_settings():
    env = {k: v for k, v in os.environ.items() if k != "BOOKING_DISCOURSE_CONTEXT"}
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    out = subprocess.run(
        [sys.executable, "-c", _API_DOTENV_PROBE],
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert out.stdout.strip().splitlines()[-1] == "False"
