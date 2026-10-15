"""
Self-checks for the offline isolation installed by tests/conftest.py, so
the guard itself cannot silently stop working.
"""

from __future__ import annotations

import os
import socket

import dotenv
import pytest

import conftest

pytestmark = pytest.mark.skipif(
    conftest.LIVE_MODE,
    reason="offline isolation is intentionally disabled in live mode",
)


def test_credentials_are_placeholders():
    for name in conftest.CREDENTIAL_ENV_VARS:
        assert os.environ[name] == conftest.OFFLINE_CREDENTIAL_PLACEHOLDER


def test_dotenv_loading_is_disabled(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("BOOKING_AI_DOTENV_PROBE=loaded\n")

    assert dotenv.load_dotenv(env_file, override=True) is False
    assert "BOOKING_AI_DOTENV_PROBE" not in os.environ


def test_local_llm_config_is_not_inherited():
    for name in conftest.LOCAL_CONFIG_ENV_VARS:
        assert name not in os.environ


@pytest.mark.parametrize(
    "attempt",
    [
        lambda: socket.getaddrinfo("generativelanguage.googleapis.com", 443),
        lambda: socket.create_connection(("93.184.216.34", 443), timeout=1),
    ],
    ids=["dns", "connect"],
)
def test_external_network_is_blocked(attempt):
    attempts_before = len(conftest._blocked_network_attempts)

    with pytest.raises(conftest.BlockedNetworkCallError):
        attempt()

    # The block was expected here - drop it so the per-test network check
    # does not fail this test for doing exactly what it asserts.
    assert len(conftest._blocked_network_attempts) == attempts_before + 1
    del conftest._blocked_network_attempts[attempts_before:]


def test_loopback_is_allowed():
    assert socket.getaddrinfo("127.0.0.1", 0)
