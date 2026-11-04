from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.schemas.conversation_response import ConversationMessage
from app.schemas.conversation_route import DiscourseContext, PendingClarification
from app.schemas.query import SearchRequest
from app.schemas.shown_result_set import ShownResultSet

# Upper bound on stored history; the response layer itself only reads the
# last few messages (see TurnConfig.history_limit).
MAX_SESSION_MESSAGES = 20


class ConversationSession(BaseModel):
    """
    Server-side conversation state carried between turns - the two domain
    state objects, the visible chat history, and an optional one-turn
    pending clarification (None for sessions stored before it existed).

    Internal contract: shown_result_set holds full ListingRaw (provider
    payload included), so a session is stored by the caller and must not
    be returned to a browser as-is - TurnResult's public payload exposes
    only session_id.
    """

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    search_request: SearchRequest | None = None
    shown_result_set: ShownResultSet | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    pending_clarification: PendingClarification | None = None
    # Only set when TurnConfig.discourse_context is enabled (experiment).
    discourse_context: DiscourseContext | None = None
