"""
Usage limits of the public chat API (app/api/usage_limits.py).

Limits are enforced before process_turn, so a rejected request makes no
LLM call and changes no conversation state. process_turn is replaced by a
counting fake here; the limits never depend on what it does.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import create_app
from app.api.usage_limits import (
    ConcurrencyGate,
    UsageLedger,
    UsageLimitConfig,
    client_address,
)
from app.schemas.turn import TurnConfig
from app.services import conversation_service

HOUR = 3600


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:  # 2027-01-15 08:00 UTC
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def turns(monkeypatch):
    """Fake process_turn: records calls, returns a real TurnResult shape."""
    calls: list[str] = []
    real = conversation_service.process_turn

    async def fake(session, message, *, config=None):
        calls.append(message)
        from app.logic import conversation_flow
        from app.schemas.conversation_route import (
            ClarificationReason,
            ConversationAction,
            ConversationActionDecision,
            ConversationDecisionStatus,
            ResultScope,
        )

        async def route(**kwargs):
            return ConversationActionDecision(
                action=ConversationAction.GENERAL_CHAT,
                reason="t",
                decision_status=ConversationDecisionStatus.RESOLVED,
                clarification_reason=ClarificationReason.NONE,
                result_scope=ResultScope.NOT_APPLICABLE,
                target_result_id="",
            )

        monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
        monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", _no_model)
        return await real(session, message, config=config)

    monkeypatch.setattr(api_main, "process_turn", fake)
    return calls


def _no_model(*args, **kwargs):
    from app.llm.call_policy import LlmCallError

    raise LlmCallError("configuration_error", stage="conversation_response_generation")


def _app(tmp_path, clock=None, **limits):
    config = UsageLimitConfig(db_path=str(tmp_path / "usage.sqlite3"), **limits)
    ledger = UsageLedger(config.db_path, clock=clock or Clock())
    return create_app(turn_config=TurnConfig(persist_telemetry=False), usage_config=config, ledger=ledger)


def _chat(client: TestClient, message="hello", session_id=None, ip="203.0.113.7"):
    body = {"message": message} if session_id is None else {"message": message, "session_id": session_id}
    return client.post("/api/chat", json=body, headers={"x-test-ip": ip})


# --- per-visitor limit ------------------------------------------------------------------


def test_visitor_limit_returns_429_before_any_turn_runs(tmp_path, turns):
    app = _app(tmp_path, visitor_turns=2, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        first = _chat(client)
        sid = first.json()["session_id"]
        assert _chat(client, session_id=sid).status_code == 200
        before = app.state.sessions._entries[sid].session.model_dump()

        rejected = _chat(client, "third", session_id=sid)

    assert rejected.status_code == 429
    assert rejected.json()["detail"] == {
        "code": "rate_limited",
        "scope": "visitor",
        "message": "You have reached the demo's message limit for today. Please come back tomorrow.",
    }
    assert int(rejected.headers["Retry-After"]) > 0
    assert turns == ["hello", "hello"]  # the rejected message never reached process_turn
    assert app.state.sessions._entries[sid].session.model_dump() == before


def test_new_sessions_do_not_reset_the_visitor_limit(tmp_path, turns):
    app = _app(tmp_path, visitor_turns=2, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client).status_code == 200
        assert _chat(client).status_code == 200  # a second, fresh session
        sessions_before = len(app.state.sessions)

        rejected = _chat(client)  # and a third

    assert rejected.status_code == 429
    assert len(app.state.sessions) == sessions_before  # no session created for a rejected request
    assert len(turns) == 2


def test_visitors_are_counted_separately(tmp_path, turns):
    app = _app(tmp_path, visitor_turns=1, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client, ip="203.0.113.7").status_code == 200
        assert _chat(client, ip="198.51.100.4").status_code == 200
        assert _chat(client, ip="203.0.113.7").status_code == 429


def test_visitor_window_is_rolling(tmp_path, turns):
    clock = Clock()
    app = _app(tmp_path, clock=clock, visitor_turns=1, global_turns_per_day=100, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client).status_code == 200
        clock.now += 23 * HOUR
        assert _chat(client).status_code == 429
        clock.now += 1 * HOUR + 1
        assert _chat(client).status_code == 200


def test_ipv6_addresses_in_one_slash_64_share_a_limit(tmp_path, turns):
    app = _app(tmp_path, visitor_turns=1, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client, ip="2001:db8:1:2::1").status_code == 200
        assert _chat(client, ip="2001:db8:1:2:ffff::9").status_code == 429
        assert _chat(client, ip="2001:db8:1:3::1").status_code == 200


# --- global daily limit -----------------------------------------------------------------


def test_global_limit_applies_across_visitors_and_survives_a_restart(tmp_path, turns):
    clock = Clock()
    app = _app(tmp_path, clock=clock, global_turns_per_day=3, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        for i in range(3):
            assert _chat(client, ip=f"203.0.113.{i}").status_code == 200

    restarted = _app(tmp_path, clock=clock, global_turns_per_day=3, client_ip_header="x-test-ip")
    with TestClient(restarted) as client:
        rejected = _chat(client, ip="198.51.100.99")

    assert rejected.status_code == 429
    assert rejected.json()["detail"]["scope"] == "global"
    assert len(turns) == 3


def test_global_limit_resets_at_utc_midnight(tmp_path, turns):
    clock = Clock(1_800_057_600.0 - 60)  # 2027-01-15 23:59 UTC
    app = _app(tmp_path, clock=clock, global_turns_per_day=1, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client, ip="203.0.113.1").status_code == 200
        rejected = _chat(client, ip="203.0.113.2")
        assert rejected.status_code == 429
        assert int(rejected.headers["Retry-After"]) == 60
        clock.now += 61
        assert _chat(client, ip="203.0.113.2").status_code == 200


def test_the_visitor_identity_survives_a_restart(tmp_path, turns):
    """The HMAC salt is stored with the counts, so a restart cannot reset a visitor."""
    first = UsageLedger(tmp_path / "usage.sqlite3")
    second = UsageLedger(tmp_path / "usage.sqlite3")

    assert first.visitor_id("203.0.113.7") == second.visitor_id("203.0.113.7")
    assert first.visitor_id("203.0.113.7") != first.visitor_id("203.0.113.8")


def test_raw_ip_addresses_are_not_stored(tmp_path, turns):
    app = _app(tmp_path, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        _chat(client, ip="203.0.113.7")

    assert b"203.0.113.7" not in (tmp_path / "usage.sqlite3").read_bytes()


# --- atomicity and concurrency ----------------------------------------------------------


def test_parallel_reservations_never_exceed_the_limit(tmp_path):
    """Many threads racing on one database: exactly `limit` turns are recorded."""
    import threading

    config = UsageLimitConfig(db_path=str(tmp_path / "usage.sqlite3"), visitor_turns=100, global_turns_per_day=5)
    ledgers = [UsageLedger(config.db_path) for _ in range(4)]  # separate connections, like separate processes
    results: list[bool] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        decision = ledgers[i % 4].try_record_turn(f"v{i}", config)
        with lock:
            results.append(decision.allowed)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 5


async def test_concurrency_limit_returns_busy_without_charging_quota(tmp_path, monkeypatch):
    release = asyncio.Event()
    started = 0

    async def slow_turn(session, message, *, config=None):
        nonlocal started
        started += 1
        await release.wait()
        raise AssertionError("not reached in this test")

    monkeypatch.setattr(api_main, "process_turn", slow_turn)
    config = UsageLimitConfig(db_path=str(tmp_path / "usage.sqlite3"), max_concurrent_turns=2, global_turns_per_day=10)
    app = create_app(turn_config=TurnConfig(persist_telemetry=False), usage_config=config)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        first = [asyncio.create_task(http.post("/api/chat", json={"message": f"m{i}"})) for i in range(2)]
        while started < 2:
            await asyncio.sleep(0.01)
        busy = await http.post("/api/chat", json={"message": "third"})

        assert busy.status_code == 429
        assert busy.json()["detail"]["scope"] == "busy"
        assert busy.headers["Retry-After"] == "5"
        assert app.state.gate.active == 2

        release.set()
        await asyncio.gather(*first)

    assert app.state.gate.active == 0  # slots are released even when a turn fails
    ledger = app.state.ledger
    visitor = ledger.visitor_id("127.0.0.1")
    assert ledger.try_record_turn(visitor, config).allowed  # 2 recorded + this one = 3 of 10


def test_gate_counts_slots():
    gate = ConcurrencyGate(2)
    assert gate.try_acquire() and gate.try_acquire()
    assert not gate.try_acquire()
    gate.release()
    assert gate.try_acquire()


# --- validation order and identification ------------------------------------------------


def test_invalid_or_unknown_requests_use_no_quota(tmp_path, turns):
    app = _app(tmp_path, visitor_turns=1, client_ip_header="x-test-ip")
    with TestClient(app) as client:
        assert _chat(client, "   ").status_code == 422
        assert _chat(client, "x" * 2001).status_code == 422
        assert _chat(client, session_id="expired").status_code == 404
        assert _chat(client).status_code == 200  # the one allowed turn is still available


def test_client_address_uses_only_a_configured_header():
    headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.7", "fly-client-ip": "198.51.100.4"}

    assert client_address("10.0.0.1", headers, UsageLimitConfig()) == "10.0.0.1"  # headers ignored by default
    assert client_address("10.0.0.1", headers, UsageLimitConfig(client_ip_header="fly-client-ip")) == "198.51.100.4"
    # X-Forwarded-For: only the entry appended by our own proxy is trusted.
    assert client_address("10.0.0.1", headers, UsageLimitConfig(client_ip_header="x-forwarded-for")) == "203.0.113.7"


def test_limits_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("BOOKING_API_VISITOR_TURNS", "3")
    monkeypatch.setenv("BOOKING_API_VISITOR_WINDOW_HOURS", "12")
    monkeypatch.setenv("BOOKING_API_GLOBAL_TURNS_PER_DAY", "50")
    monkeypatch.setenv("BOOKING_API_MAX_CONCURRENT_TURNS", "1")
    monkeypatch.setenv("BOOKING_API_USAGE_DB", "/data/usage.sqlite3")
    monkeypatch.setenv("BOOKING_API_CLIENT_IP_HEADER", "Fly-Client-IP")

    config = UsageLimitConfig.from_env()

    assert (config.visitor_turns, config.visitor_window_seconds, config.global_turns_per_day) == (3, 12 * HOUR, 50)
    assert (config.max_concurrent_turns, config.db_path, config.client_ip_header) == (1, "/data/usage.sqlite3", "fly-client-ip")


def test_defaults_match_the_first_public_version(monkeypatch):
    for name in ("BOOKING_API_VISITOR_TURNS", "BOOKING_API_GLOBAL_TURNS_PER_DAY", "BOOKING_API_MAX_CONCURRENT_TURNS"):
        monkeypatch.delenv(name, raising=False)

    config = UsageLimitConfig.from_env()

    assert (config.visitor_turns, config.visitor_window_seconds, config.global_turns_per_day, config.max_concurrent_turns) == (
        8,
        24 * HOUR,
        30,
        2,
    )


# --- the public API never calls a paid retrieval provider -------------------------------


def test_public_api_always_searches_the_sample_listings(monkeypatch):
    monkeypatch.delenv("BOOKING_API_PERSIST_TELEMETRY", raising=False)
    app = create_app(usage_config=UsageLimitConfig(db_path=":memory:"))

    assert app.state.turn_config.source == "fixtures"
    assert app.state.turn_config.persist_telemetry is False  # telemetry files would store message text
