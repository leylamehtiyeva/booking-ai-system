"""
FastAPI adapter over the application service.

One chat turn = one call to app.services.conversation_service.process_turn.
The API only loads the caller's ConversationSession, runs the turn and
stores the session process_turn returns; routing, retrieval, matching,
ranking and LLM calls all stay behind process_turn.

Run (single worker - see app/api/session_store.py):

    uv run uvicorn app.api.main:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.session_store import InMemorySessionStore, SessionNotFoundError
from app.api.usage_limits import (
    ConcurrencyGate,
    UsageLedger,
    UsageLimitConfig,
    client_address,
    run_in_thread,
)
from app.schemas.turn import TurnConfig, TurnResult
from app.services.conversation_service import process_turn

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000
DEFAULT_CORS_ORIGINS = "http://localhost:4321,http://127.0.0.1:4321"  # Astro dev server

UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while processing your message. "
    "Your current search has not been changed. Please try again."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    # Omit to start a new conversation; send back the session_id of the
    # previous response to continue it.
    session_id: str | None = Field(default=None, max_length=64)


class HealthResponse(BaseModel):
    status: str = "ok"


def _cors_origins() -> list[str]:
    raw = os.getenv("BOOKING_API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _default_turn_config() -> TurnConfig:
    # The public demo always searches the bundled sample listings: no paid
    # retrieval provider is ever called from a visitor's message. Telemetry
    # files would store the message text, so they are opt-in.
    return TurnConfig(source="fixtures", persist_telemetry=os.getenv("BOOKING_API_PERSIST_TELEMETRY") == "1")


LIMIT_MESSAGES = {
    "visitor": "You have reached the demo's message limit for today. Please come back tomorrow.",
    "global": "The demo has reached its daily message limit. Please try again tomorrow.",
    "busy": "The demo is busy right now. Please try again in a few seconds.",
}
BUSY_RETRY_SECONDS = 5


def _rate_limited(scope: str, retry_after: int) -> HTTPException:
    logger.info("Chat turn rejected by usage limit: scope=%s", scope)
    return HTTPException(
        status_code=429,
        detail={"code": "rate_limited", "scope": scope, "message": LIMIT_MESSAGES[scope]},
        headers={"Retry-After": str(retry_after)},
    )


def create_app(
    *,
    store: InMemorySessionStore | None = None,
    turn_config: TurnConfig | None = None,
    usage_config: UsageLimitConfig | None = None,
    ledger: UsageLedger | None = None,
) -> FastAPI:
    app = FastAPI(title="Booking AI API", version="0.1.0")
    app.state.sessions = store or InMemorySessionStore()
    app.state.turn_config = turn_config or _default_turn_config()
    limits = usage_config or UsageLimitConfig.from_env()
    app.state.usage_config = limits
    app.state.gate = ConcurrencyGate(limits.max_concurrent_turns)
    app.state.ledger = ledger  # opened on first use, so importing the app writes nothing

    def usage_ledger() -> UsageLedger:
        if app.state.ledger is None:
            app.state.ledger = UsageLedger(limits.db_path)
        return app.state.ledger

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @asynccontextmanager
    async def admitted(visitor: str):
        """Reserve a concurrency slot and one turn of quota before any paid call."""
        gate: ConcurrencyGate = app.state.gate
        if not gate.try_acquire():
            raise _rate_limited("busy", BUSY_RETRY_SECONDS)
        try:
            decision = await run_in_thread(usage_ledger().try_record_turn, visitor, limits)
            if not decision.allowed:
                raise _rate_limited(decision.scope, decision.retry_after_seconds)
            yield
        finally:
            gate.release()

    async def run_turn(entry, message: str) -> TurnResult:
        try:
            turn = await process_turn(entry.session, message, config=app.state.turn_config)
        except Exception:
            # Expected provider/retrieval failures come back as typed
            # turns; this is a programming error. Nothing is stored.
            logger.exception("Unexpected error while processing a chat turn")
            raise HTTPException(
                status_code=500,
                detail={"code": "internal_error", "message": UNEXPECTED_ERROR_MESSAGE},
            ) from None
        entry.session = turn.session
        return turn

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/chat", response_model=TurnResult)
    async def chat(request: ChatRequest, http_request: Request) -> TurnResult:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail={"code": "empty_message"})

        sessions: InMemorySessionStore = app.state.sessions
        peer = http_request.client.host if http_request.client else None
        visitor = usage_ledger().visitor_id(client_address(peer, http_request.headers, limits))

        # Nothing is created or changed for a rejected request: an unknown
        # session is a 404 before any quota is used, and a new session is
        # only created once the turn has been admitted.
        try:
            if request.session_id:
                async with sessions.checkout(request.session_id) as entry:
                    async with admitted(visitor):
                        return await run_turn(entry, message)
            async with admitted(visitor):
                async with sessions.checkout(sessions.create()) as entry:
                    return await run_turn(entry, message)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "Start a new conversation without session_id."},
            ) from None

    return app


app = create_app()
