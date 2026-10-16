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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.session_store import InMemorySessionStore, SessionNotFoundError
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


def create_app(
    *,
    store: InMemorySessionStore | None = None,
    turn_config: TurnConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="Booking AI API", version="0.1.0")
    app.state.sessions = store or InMemorySessionStore()
    app.state.turn_config = turn_config or TurnConfig()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/chat", response_model=TurnResult)
    async def chat(request: ChatRequest) -> TurnResult:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail={"code": "empty_message"})

        sessions: InMemorySessionStore = app.state.sessions
        session_id = request.session_id or sessions.create()

        try:
            async with sessions.checkout(session_id) as entry:
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
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "Start a new conversation without session_id."},
            ) from None

        return turn

    return app


app = create_app()
