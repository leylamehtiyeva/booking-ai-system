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


class ResultScope(str, Enum):
    """
    Only meaningful when action == LISTING_QUESTION.

    NOT_APPLICABLE: default/sentinel - no specific-result semantics apply
    (used instead of Optional/None: some structured-output providers,
    e.g. Groq, reject an Optional[Enum] field's JSON Schema outright).
    CURRENT_RESULTS: the question is about the whole shown set
    ("which of these...").
    SPECIFIC_RESULT: the question is about exactly one shown listing
    (ordinal, pronoun, or name reference).
    """

    NOT_APPLICABLE = "not_applicable"
    CURRENT_RESULTS = "current_results"
    SPECIFIC_RESULT = "specific_result"


class ConversationDecisionStatus(str, Enum):
    """
    The router is not required to always resolve action/reference. A
    genuinely ambiguous turn should abstain (NEEDS_CLARIFICATION) rather
    than guess - guessing a persistent mutation (update_search) when the
    user actually meant a transient query is the most dangerous failure
    mode this status exists to prevent.
    """

    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"


class ClarificationReason(str, Enum):
    """
    NONE is the sentinel for decision_status == RESOLVED (kept as a real
    enum member, not Optional, for the same provider-schema reason as
    ResultScope.NOT_APPLICABLE above).
    """

    NONE = "none"
    AMBIGUOUS_ACTION = "ambiguous_action"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"


class RouterInput(BaseModel):
    user_message: str
    current_search: SearchRequest | None = None
    latest_result_context: dict[str, Any] | None = None


class ConversationActionDecision(BaseModel):
    """
    Every field is required, with no Python-side default, on purpose:
    Groq's (OpenAI-compatible) strict structured output rejects a schema
    unless every property is listed in "required", regardless of whether
    it has a default - Pydantic only puts default-less fields there, and
    there is no working per-model override for this in the ADK/LiteLLM
    schema path (json_schema_serialization_defaults_required does not
    apply - verified empirically against a live Groq call). Gemini has no
    such restriction, so a single required-everywhere schema is the only
    shape confirmed to work for both providers - the sentinel values
    below (rather than Optional/None) exist for the same reason.
    """

    action: ConversationAction
    reason: str

    # Explicit abstention contract - no numeric confidence, just a status
    # plus a small closed set of reasons. decision_status is checked
    # BEFORE action is ever dispatched on: NEEDS_CLARIFICATION always
    # short-circuits to a read-only clarification turn regardless of what
    # action/result_scope/target_result_id otherwise contain.
    decision_status: ConversationDecisionStatus
    clarification_reason: ClarificationReason

    # Router's own (unvalidated) semantic reference resolution - only
    # meaningful when action == LISTING_QUESTION. target_result_id is the
    # LLM's best guess at a result_id from the shown_results context; it is
    # NOT trusted as-is - conversation_flow.py deterministically checks it
    # against the actual latest ShownResultSet before use. "" is the
    # sentinel for "not set" (see ResultScope.NOT_APPLICABLE docstring).
    result_scope: ResultScope
    target_result_id: str

