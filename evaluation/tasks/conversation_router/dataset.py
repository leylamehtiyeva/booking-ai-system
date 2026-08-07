from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.schemas.conversation_route import ConversationAction
from evaluation.core.io import load_jsonl


class ConversationRouterEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_message: str
    has_current_search: bool
    has_shown_results: bool
    expected_action: ConversationAction
    category: str
    notes: str


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