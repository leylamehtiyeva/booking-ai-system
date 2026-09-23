from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.result_query_match import ResultQueryMatch
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


class ListingQueryResponseItem(BaseModel):
    """
    One shown candidate's already-final factual result, paired with its
    presentation identity - position/title only, never ListingRaw. `match`
    is the real ResultQueryMatch (not a copy/subset of its fields), so
    there is exactly one collection to keep in sync, not two parallel
    lists that could drift apart.
    """

    position: int
    title: str | None = None
    match: ResultQueryMatch


class ListingQueryConversationOutcome(BaseModel):
    """
    ResultQueryMatch[] is the final factual source of truth (see
    result_query_matching.py) - this outcome carries it through unchanged,
    only joined with presentation metadata so a renderer never has to
    reach back into ListingRaw/ShownResultSet for a title. items is
    already sorted by position; the join/validation happens once, in
    conversation_response_adapter.py.
    """

    kind: Literal["listing_query"] = "listing_query"
    items: list[ListingQueryResponseItem]


class InformationalConversationOutcome(BaseModel):
    """
    A resolved, non-clarification, non-failure turn whose text is fully
    determined by a semantic code - not a pre-built message string, so
    the response layer (not conversation_flow.py) owns the actual
    wording. Covers UPDATE_SEARCH no-ops and unsupported listing
    questions today; both are "nothing went wrong, there's just nothing
    more to compute", which is why they share one outcome kind instead of
    each getting its own.
    """

    kind: Literal["informational"] = "informational"
    code: Literal["update_no_op", "listing_query_unsupported"]


class ConversationFailureOutcome(BaseModel):
    kind: Literal["failure"] = "failure"
    user_safe_message: str


ConversationOutcome = Annotated[
    SearchConversationOutcome
    | GeneralChatConversationOutcome
    | ClarificationConversationOutcome
    | ListingQueryConversationOutcome
    | InformationalConversationOutcome
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