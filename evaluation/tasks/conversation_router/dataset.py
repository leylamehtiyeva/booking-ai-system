from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationDecisionStatus,
    ResultScope,
)
from evaluation.core.io import load_jsonl


class ShownResultStub(BaseModel):
    """
    Synthetic shown-result entry for eval cases - deliberately not a
    ListingRaw. Position in the case's `shown_results` list is the
    display order (1-indexed when projected for the router).
    """

    model_config = ConfigDict(extra="forbid")

    result_id: str
    title: str


class ConversationRouterEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_message: str
    has_current_search: bool
    has_shown_results: bool
    expected_action: ConversationAction
    category: str
    notes: str

    # Optional - only populated by reference-resolution style datasets.
    # Absent/empty on router_eval_dev.jsonl's existing rows. These are
    # ground-truth fields for eval scoring, loaded from plain JSONL - not
    # sent to any LLM, so Optional[Enum] here has none of the structured
    # output schema restrictions ConversationActionDecision has to avoid.
    shown_results: list[ShownResultStub] = Field(default_factory=list)
    expected_result_scope: ResultScope | None = None
    expected_target_result_id: str | None = None
    expected_decision_status: ConversationDecisionStatus = (
        ConversationDecisionStatus.RESOLVED
    )
    expected_clarification_reason: ClarificationReason = ClarificationReason.NONE


def load_conversation_router_dataset(
    path: str | Path,
) -> list[ConversationRouterEvalCase]:
    rows = load_jsonl(path)
    cases: list[ConversationRouterEvalCase] = []

    for row in rows:
        row.pop("critical_if_wrong", None)

        cases.append(
            ConversationRouterEvalCase.model_validate(row)
        )

    return cases