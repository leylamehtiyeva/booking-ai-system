# evaluation/tasks/end_to_end/dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Decision = Literal["YES", "NO", "UNCERTAIN"]


class EndToEndEvalCase(BaseModel):
    case_id: str
    case_type: str
    query_style: str

    user_query: str

    expected_decision: Decision
    expected_selected_id: str | None = None
    acceptable_ids: list[str] = Field(default_factory=list)

    hard_constraints: dict
    reason: str | None = None
    target_listing_id: str | None = None


def _parse_acceptable_ids(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _parse_hard_constraints(value: str | dict) -> dict:
    if isinstance(value, dict):
        return value

    return json.loads(value)


def load_end_to_end_dataset(path: str | Path) -> list[EndToEndEvalCase]:
    cases: list[EndToEndEvalCase] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw_case = json.loads(line)

            raw_case["acceptable_ids"] = _parse_acceptable_ids(
                raw_case.get("acceptable_ids")
            )

            raw_case["hard_constraints"] = _parse_hard_constraints(
                raw_case["hard_constraints"]
            )

            cases.append(EndToEndEvalCase(**raw_case))

    return cases