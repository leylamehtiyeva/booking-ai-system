"""
Test-suite isolation: a plain `uv run pytest` must never reach a real LLM
API or any other external service.

Everything that matters here runs at MODULE IMPORT time, not in a fixture:
pytest imports this conftest before any test module, and therefore before
any `app.*` import. That ordering matters because litellm calls
`dotenv.load_dotenv()` inside its own `__init__` (pulled in transitively by
`import app...`), so a fixture would run too late - the real keys from
`.env` would already be in os.environ by then.

Offline mode (default):
- `.env` loading is disabled, so neither credentials nor local LLM config
  (GEMINI_MODEL, ROUTER_LLM_PROFILE, ...) leak into the tests;
- every credential env var is replaced with a non-functional placeholder
  (also overriding values exported in the shell);
- outbound sockets/DNS to non-loopback hosts are blocked, and every blocked
  attempt fails the test even if application code catches the exception
  (several layers deliberately swallow provider errors and fall back).

Live mode (explicit opt-in, may cost money):
    BOOKING_AI_LIVE_TESTS=1 uv run pytest -m live
Only in this mode is `.env` loaded and the network left open. Tests that
need real services are marked `@pytest.mark.live` and are skipped otherwise.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from pathlib import Path

import dotenv
import dotenv.main
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIVE_TESTS_ENV_VAR = "BOOKING_AI_LIVE_TESTS"
LIVE_MODE = os.getenv(LIVE_TESTS_ENV_VAR) == "1"

OFFLINE_CREDENTIAL_PLACEHOLDER = "offline-test-placeholder"

# Fixed "today" for the offline suite (see app/config/clock.py), so tests
# with dated fixtures/requests do not start failing as the calendar moves.
# Tests that need another day use clock.use_reference_date(...).
OFFLINE_REFERENCE_DATE = "2026-03-01"

CREDENTIAL_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "APIFY_TOKEN",
    "BRIGHT_DATA_API_KEY",
)

# Local LLM configuration that would otherwise change which model/profile
# the code under test selects depending on the developer's machine.
LOCAL_CONFIG_ENV_VARS = (
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODELS",
    "SEMANTIC_VERIFIER_MODEL",
    "DEFAULT_LLM_PROFILE",
    "ROUTER_LLM_PROFILE",
    "CONVERSATION_RESPONSE_LLM_PROFILE",
    "LITELLM_DEV_ENV_HOT_RELOAD",
)


class BlockedNetworkCallError(RuntimeError):
    """Raised when an offline test tries to reach a non-loopback host."""


_blocked_network_attempts: list[str] = []


def _disable_dotenv() -> None:
    def _no_op_load_dotenv(*args, **kwargs) -> bool:
        return False

    dotenv.load_dotenv = _no_op_load_dotenv
    dotenv.main.load_dotenv = _no_op_load_dotenv


def _apply_offline_environment() -> None:
    for name in CREDENTIAL_ENV_VARS:
        os.environ[name] = OFFLINE_CREDENTIAL_PLACEHOLDER

    for name in LOCAL_CONFIG_ENV_VARS:
        os.environ.pop(name, None)

    os.environ["APP_REFERENCE_DATE"] = OFFLINE_REFERENCE_DATE


def _is_loopback_host(host: object) -> bool:
    if host is None:
        return True

    host_text = host.decode() if isinstance(host, bytes) else str(host)

    if host_text.lower() == "localhost":
        return True

    try:
        return ipaddress.ip_address(host_text.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _guard(host: object, port: object, api: str) -> None:
    if _is_loopback_host(host):
        return

    attempt = f"{api}({host!r}, {port!r})"
    _blocked_network_attempts.append(attempt)
    raise BlockedNetworkCallError(
        f"Offline test attempted a network call: {attempt}. Mock the external "
        f"dependency, or mark the test @pytest.mark.live and run it with "
        f"{LIVE_TESTS_ENV_VAR}=1."
    )


def _install_network_guard() -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        _guard(host, port, "getaddrinfo")
        return original_getaddrinfo(host, port, *args, **kwargs)

    def _guard_address(sock: socket.socket, address, api: str) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _guard(address[0], address[1], api)

    def guarded_connect(self, address):
        _guard_address(self, address, "socket.connect")
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        _guard_address(self, address, "socket.connect_ex")
        return original_connect_ex(self, address)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


if LIVE_MODE:
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
else:
    _disable_dotenv()
    _apply_offline_environment()
    _install_network_guard()


def pytest_collection_modifyitems(config, items):
    if LIVE_MODE:
        return

    skip_live = pytest.mark.skip(
        reason=f"live test - run with {LIVE_TESTS_ENV_VAR}=1 uv run pytest -m live"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """
    Fail the test on any blocked network attempt, including ones the code
    under test caught and silently degraded from (e.g. an LLM fallback).
    """
    attempts_before = len(_blocked_network_attempts)
    try:
        return (yield)
    finally:
        new_attempts = _blocked_network_attempts[attempts_before:]
        if new_attempts:
            pytest.fail(
                "Offline test attempted network access "
                f"({len(new_attempts)} blocked): " + "; ".join(new_attempts),
                pytrace=False,
            )


@pytest.fixture(autouse=True)
def _offline_credentials(monkeypatch):
    """Re-assert placeholders per test in case anything rewrote os.environ."""
    if LIVE_MODE:
        return

    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.setenv(name, OFFLINE_CREDENTIAL_PLACEHOLDER)
