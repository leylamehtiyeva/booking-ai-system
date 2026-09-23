from __future__ import annotations

from pydantic import BaseModel, Field as PydanticField, field_validator

from app.schemas.constraints import UserConstraint


class ResultQuery(BaseModel):
    """
    Transient, one-turn factual query over already-shown listings
    (LISTING_QUESTION). Reuses UserConstraint as-is - no new constraint
    model, no new Field vocabulary, no new priority semantics.

    This is deliberately NOT SearchRequest and NOT SearchIntentPatch, and
    carries no scope/target_result_id/SearchRequest fields - operation/
    scope/reference are already decided by the router before this object
    is built (ConversationActionDecision.result_scope/target_result_id).

    Must never be passed to apply_intent_patch, set_search_state, or
    orchestrate_search_request - it has no persistent-state identity at
    all, only a list of facts being asked about.
    """

    constraints: list[UserConstraint] = PydanticField(default_factory=list)

    @field_validator("constraints", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v
