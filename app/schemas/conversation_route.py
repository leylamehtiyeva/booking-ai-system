from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel
from app.schemas.query import SearchRequest


class ConversationAction(str, Enum):
    START_SEARCH = "start_search"
    UPDATE_SEARCH = "update_search"
    LISTING_QUESTION = "listing_question"
    GENERAL_CHAT = "general_chat"


class RouterInput(BaseModel):
    user_message: str
    current_search: SearchRequest | None = None
    latest_result_context: dict[str, Any] | None = None


class ConversationActionDecision(BaseModel):
    action: ConversationAction
    reason: str

