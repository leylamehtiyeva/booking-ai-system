"""
In-memory ConversationSession store for the API MVP.

The store only keeps the sessions process_turn returns; it holds no
conversation logic of its own. Each session has a lock so two requests
in the same session run one after the other and the second one sees the
first one's result (no lost update); different sessions run concurrently.

MVP limitations, by design:
- sessions live in this process only: a restart loses them, and with
  several workers (uvicorn --workers N, several replicas) a session exists
  in one worker only. Run a single worker, or replace this class with a
  shared store (Redis, a database) behind the same three methods;
- idle sessions expire after `ttl_seconds`; beyond `max_sessions` the
  least recently used one is dropped.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from app.schemas.session import ConversationSession


class SessionNotFoundError(KeyError):
    """The session id is unknown or expired (e.g. after a server restart)."""


@dataclass
class _Entry:
    session: ConversationSession
    touched_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InMemorySessionStore:
    def __init__(self, *, max_sessions: int = 1000, ttl_seconds: float = 3600, clock=time.monotonic) -> None:
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def __len__(self) -> int:
        return len(self._entries)

    def create(self) -> str:
        self._expire()
        while len(self._entries) >= self._max_sessions and not self._oldest_locked():
            del self._entries[next(iter(self._entries))]
        session = ConversationSession()
        self._entries[session.session_id] = _Entry(session=session, touched_at=self._clock())
        return session.session_id

    @asynccontextmanager
    async def checkout(self, session_id: str) -> AsyncIterator[_Entry]:
        """
        Hold the session for one turn. The caller reads `entry.session` and
        assigns the updated session to it; nothing is assigned on error, so
        the stored session stays as it was.
        """
        entry = self._get(session_id)
        async with entry.lock:
            yield entry
            entry.touched_at = self._clock()

    def _get(self, session_id: str) -> _Entry:
        self._expire()
        entry = self._entries.get(session_id)
        if entry is None:
            raise SessionNotFoundError(session_id)
        self._entries.move_to_end(session_id)
        return entry

    def _oldest_locked(self) -> bool:
        return bool(self._entries) and self._entries[next(iter(self._entries))].lock.locked()

    def _expire(self) -> None:
        now = self._clock()
        for session_id, entry in list(self._entries.items()):
            if now - entry.touched_at > self._ttl_seconds and not entry.lock.locked():
                del self._entries[session_id]
