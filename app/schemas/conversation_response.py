from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SearchConversationOutcome(BaseModel):
    kind: Literal["search"] = "search"
    search_response: NormalizedSearchResponse


class GeneralChatConversationOutcome(BaseModel):
    kind: Literal["general_chat"] = "general_chat"


class ClarificationConversationOutcome(BaseModel):
    kind: Literal["clarification"] = "clarification"
    questions: list[str]


class ConversationFailureOutcome(BaseModel):
    kind: Literal["failure"] = "failure"
    user_safe_message: str


ConversationOutcome = Annotated[
    SearchConversationOutcome
    | GeneralChatConversationOutcome
    | ClarificationConversationOutcome
    | ConversationFailureOutcome,
    Field(discriminator="kind"),
]


class ConversationResponseInput(BaseModel):
    user_message: str
    action: ConversationAction
    recent_messages: list[ConversationMessage] = Field(
        default_factory=list
    )
    current_search: SearchRequest | None = None
    outcome: ConversationOutcome