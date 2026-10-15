# evaluation/tasks/end_to_end/adapter.py

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.clock import use_reference_date
from app.logic.conversation_flow import handle_user_message
from evaluation.core.reference_date import EVAL_REFERENCE_DATE
from evaluation.tasks.end_to_end.contract import (
    CONTRACT_VERSION,
    infer_decision,
    infer_legacy_decision,
    selected_listing_ids,
)
from evaluation.tasks.end_to_end.dataset import Decision, EndToEndEvalCase
from evaluation.tasks.end_to_end.run_config import EvalRunConfig


class EvalTurnFailure(RuntimeError):
    """The turn ended in a typed failure outcome - scored as an error, not a decision."""

    def __init__(self, failure: dict[str, Any] | None, *, telemetry: dict[str, Any] | None = None) -> None:
        failure = failure or {}
        super().__init__(f"{failure.get('stage', 'unknown')}:{failure.get('code', 'unknown')}")
        self.failure = failure
        self.telemetry = telemetry or {}


class EndToEndPrediction(BaseModel):
    contract_version: str = CONTRACT_VERSION
    decision: Decision
    decision_basis: str
    legacy_decision: Decision
    selected_listing_id: str | None = None
    selected_listing_ids: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any]


def _failure_of(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("response_type") == "routing_unavailable":
        router = (response.get("parsed_intent") or {}).get("router") or {}
        return {"stage": "conversation_routing", "code": router.get("error_code")}
    return response.get("failure") or {}


async def run_case(
    case: EndToEndEvalCase,
    *,
    config: EvalRunConfig | None = None,
) -> EndToEndPrediction:
    config = config or EvalRunConfig()

    with use_reference_date(EVAL_REFERENCE_DATE):
        response = await handle_user_message(
            user_message=case.user_query,
            previous_state=None,
            source=config.source,
            top_n=config.top_n,
            max_items=config.max_items,
            fallback_policy=config.fallback_policy(),
        )

    decision, basis = infer_decision(response)
    if decision is None:
        raise EvalTurnFailure(_failure_of(response), telemetry=response.get("telemetry"))

    selected_ids = selected_listing_ids(response)

    return EndToEndPrediction(
        decision=decision,
        decision_basis=basis,
        legacy_decision=infer_legacy_decision(response),
        selected_listing_id=selected_ids[0] if selected_ids else None,
        selected_listing_ids=selected_ids,
        raw_response=response,
    )
