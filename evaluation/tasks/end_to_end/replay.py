"""
Free, LLM-free replay of saved end-to-end predictions through the CURRENT
search backend.

Each saved prediction carries the SearchRequest the LLM extracted at the
time ("state"). Replay re-runs that request through today's retrieval,
matching, ranking and selection with the textual LLM fallback disabled,
then scores it with the current decision contract.

What it measures: the effect of deterministic backend changes (e.g. P0-8
fixture pricing, property-type safety net) on fixed intents, at zero cost.
What it does not: intent extraction or textual-fallback quality - it is an
approximation, never a substitute for a full `predict` run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.clock import use_reference_date
from app.logic.request_resolution import resolve_required_search_context
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.query import SearchRequest
from app.tools.orchestrate_search_tool import orchestrate_search_request
from evaluation.core.reference_date import EVAL_REFERENCE_DATE
from evaluation.tasks.end_to_end.comparator import compare_case
from evaluation.tasks.end_to_end.dataset import load_end_to_end_dataset
from evaluation.tasks.end_to_end.metrics import compute_metrics
from evaluation.tasks.end_to_end.run_config import EvalRunConfig

REPLAY_FALLBACK_POLICY = FallbackPolicy(enabled=False)


async def _replay_response(state: dict[str, Any], config: EvalRunConfig) -> dict[str, Any]:
    request = SearchRequest.model_validate(state)

    resolved = resolve_required_search_context(request)
    if resolved.need_clarification:
        return {"need_clarification": True, "questions": resolved.questions, "state": state}

    search_response, _ = await orchestrate_search_request(
        request,
        result_limit=config.top_n,
        candidate_pool_size=config.max_items,
        fallback_policy=REPLAY_FALLBACK_POLICY,
        source=config.source,
    )
    return {**search_response.model_dump(mode="json", exclude_none=True), "state": state}


async def replay_predictions(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    config: EvalRunConfig | None = None,
) -> dict[str, Any]:
    config = config or EvalRunConfig()
    cases = load_end_to_end_dataset(dataset_path)

    saved: dict[str, dict[str, Any]] = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            saved[record["case_id"]] = record

    rows = []
    skipped = []
    with use_reference_date(EVAL_REFERENCE_DATE):
        for case in cases:
            record = saved.get(case.case_id)
            state = (
                ((record or {}).get("prediction") or {}).get("raw_response") or {}
            ).get("state")

            if record is None or record.get("status") != "ok" or not state:
                skipped.append(case.case_id)
                continue

            response = await _replay_response(state, config)
            rows.append(
                compare_case(
                    case=case,
                    prediction_record={
                        "case_id": case.case_id,
                        "status": "ok",
                        "prediction": {"raw_response": response},
                    },
                )
            )

    return {
        "metrics": compute_metrics(rows),
        "rows": rows,
        "skipped_case_ids": skipped,
        "replay": {
            "textual_fallback": "disabled",
            "intents_from": str(predictions_path),
            "run_config": config.model_dump(),
        },
    }
