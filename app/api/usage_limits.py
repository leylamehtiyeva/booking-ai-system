"""
Server-side usage limits for the public chat API.

Every accepted chat turn may call a paid LLM several times, so turns are
limited before process_turn runs:

- per visitor: `visitor_turns` turns in a rolling `visitor_window_seconds`;
- globally: `global_turns_per_day` turns per UTC calendar day;
- concurrency: at most `max_concurrent_turns` turns in flight.

These count turns, not money. A turn's LLM cost varies (router, extraction,
evidence fallback, response wording), and the telemetry cost is an estimate,
so the provider-side spend cap stays the real financial limit.

Visitors are identified by client IP address, never by session_id (a new
session is free to create). The address is stored only as an HMAC with a
random salt kept in the database; IPv6 addresses are reduced to their /64
network so one host cannot rotate through its own block.

Counts live in SQLite (stdlib), so they survive a restart as long as the
database file is on persistent storage. Check-and-record runs in one
`BEGIN IMMEDIATE` transaction, which is atomic across threads and across
processes using the same file. The concurrency gate is per process - the
MVP runs one worker.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

LimitScope = Literal["visitor", "global", "busy"]

DAY_SECONDS = 24 * 3600
# Rows older than this are no longer needed by either limit.
RETENTION_SECONDS = 2 * DAY_SECONDS


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class UsageLimitConfig:
    visitor_turns: int = 8
    visitor_window_seconds: int = DAY_SECONDS
    global_turns_per_day: int = 30
    max_concurrent_turns: int = 2
    db_path: str = "var/usage.sqlite3"
    # Header holding the real client IP when behind a trusted proxy that
    # sets it (e.g. "fly-client-ip", "cf-connecting-ip"). Unset: the TCP
    # peer address is used. Never trust a header the client can set.
    client_ip_header: str | None = None

    @classmethod
    def from_env(cls) -> UsageLimitConfig:
        header = os.getenv("BOOKING_API_CLIENT_IP_HEADER", "").strip().lower() or None
        return cls(
            visitor_turns=_env_int("BOOKING_API_VISITOR_TURNS", cls.visitor_turns),
            visitor_window_seconds=_env_int("BOOKING_API_VISITOR_WINDOW_HOURS", 24, minimum=1) * 3600,
            global_turns_per_day=_env_int("BOOKING_API_GLOBAL_TURNS_PER_DAY", cls.global_turns_per_day),
            max_concurrent_turns=_env_int("BOOKING_API_MAX_CONCURRENT_TURNS", cls.max_concurrent_turns, minimum=1),
            db_path=os.getenv("BOOKING_API_USAGE_DB", cls.db_path),
            client_ip_header=header,
        )


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    scope: LimitScope | None = None
    retry_after_seconds: int = 0


class UsageLedger:
    """Persistent turn counts; one row per accepted turn."""

    def __init__(self, db_path: str | Path, *, clock=time.time) -> None:
        self._path = str(db_path)
        self._clock = clock
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY,
                    visitor TEXT NOT NULL,
                    ts REAL NOT NULL,
                    day TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS turns_visitor_ts ON turns (visitor, ts);
                CREATE INDEX IF NOT EXISTS turns_day ON turns (day);
                """
            )
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('salt', ?)", (secrets.token_hex(32),))
            self._salt = bytes.fromhex(conn.execute("SELECT value FROM meta WHERE key = 'salt'").fetchone()[0])

    def _connect(self) -> sqlite3.Connection:
        # Autocommit mode: transactions are opened explicitly below.
        return sqlite3.connect(self._path, timeout=5, isolation_level=None)

    def visitor_id(self, client_address: str) -> str:
        return hmac.new(self._salt, _network_key(client_address).encode(), hashlib.sha256).hexdigest()[:32]

    def try_record_turn(self, visitor: str, config: UsageLimitConfig) -> LimitDecision:
        """Atomically check both limits and, if allowed, record the turn."""
        now = self._clock()
        today = datetime.fromtimestamp(now, tz=timezone.utc)
        day = today.strftime("%Y-%m-%d")
        window_start = now - config.visitor_window_seconds

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            (global_count,) = conn.execute("SELECT COUNT(*) FROM turns WHERE day = ?", (day,)).fetchone()
            if global_count >= config.global_turns_per_day:
                conn.execute("ROLLBACK")
                midnight = (today + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                return LimitDecision(False, "global", max(1, int(midnight.timestamp() - now)))

            visitor_count, oldest = conn.execute(
                "SELECT COUNT(*), MIN(ts) FROM turns WHERE visitor = ? AND ts > ?", (visitor, window_start)
            ).fetchone()
            if visitor_count >= config.visitor_turns:
                conn.execute("ROLLBACK")
                retry = (oldest or now) + config.visitor_window_seconds - now
                return LimitDecision(False, "visitor", max(1, int(retry)))

            conn.execute("INSERT INTO turns (visitor, ts, day) VALUES (?, ?, ?)", (visitor, now, day))
            conn.execute("DELETE FROM turns WHERE ts < ?", (now - RETENTION_SECONDS,))
            conn.execute("COMMIT")
            return LimitDecision(True)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


class ConcurrencyGate:
    """Non-blocking cap on turns in flight in this process."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    def try_acquire(self) -> bool:
        # Single event loop and no await between check and increment:
        # the check-then-act is atomic.
        if self._active >= self._limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        self._active = max(0, self._active - 1)


def client_address(peer: str | None, headers, config: UsageLimitConfig) -> str:
    if config.client_ip_header:
        value = headers.get(config.client_ip_header)
        if value:
            # For X-Forwarded-For the rightmost entry is the one our own
            # proxy appended; earlier entries are client-controlled.
            return value.split(",")[-1].strip()
    return peer or "unknown"


def _network_key(address: str) -> str:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return f"raw:{address}"
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


async def run_in_thread(func, *args):
    """SQLite calls are short but blocking; keep them off the event loop."""
    return await asyncio.to_thread(func, *args)
