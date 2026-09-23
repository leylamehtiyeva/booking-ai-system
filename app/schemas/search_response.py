from __future__ import annotations

from typing import Any
from enum import Enum
from pydantic import BaseModel, Field, Field as PydanticField



class SearchStatus(str, Enum):
    RESULTS = "results"
    NO_RESULTS = "no_results"

class NormalizedRequestSummary(BaseModel):
    city: str | None = None
    check_in: str | None = None
    check_out: str | None = None

    property_types: list[str] = Field(default_factory=list)
    occupancy_types: list[str] = Field(default_factory=list)

    filters: dict[str, Any] = Field(default_factory=dict)

    # Debug / normalization residue.
    dropped_requests: list[str] = Field(default_factory=list)

    # Canonical semantic state.
    constraints: list[dict] = PydanticField(default_factory=list)


class ConstraintStatus(BaseModel):
    name: str
    status: str  # matched | uncertain | failed
    reason: str | None = None
    constraint: dict[str, Any] | None = None


class ResultFact(BaseModel):
    key: str
    value: Any
    source: str | None = None


class ConstraintResolutionEvidence(BaseModel):
    snippet: str
    source: str
    path: str | None = None


class ConstraintResolutionItem(BaseModel):
    listing_id: str | None = None
    listing_title: str | None = None
    constraint_id: str | None = None

    raw_text: str
    normalized_text: str

    resolver_type: str
    decision: str
    resolution_status: str
    confidence: float | None = None
    reason: str

    evidence: list[ConstraintResolutionEvidence] = Field(default_factory=list)

    source_stage: str = "fallback"
    structured_value_before: str | None = None
    explicit_negative: bool = False


class NormalizedSearchResult(BaseModel):
    result_id: str
    title: str
    url: str | None = None

    score: float

    matched_constraints: list[ConstraintStatus] = Field(default_factory=list)
    uncertain_constraints: list[ConstraintStatus] = Field(default_factory=list)
    failed_constraints: list[ConstraintStatus] = Field(default_factory=list)
    
    matched_requested_constraints: list[ConstraintStatus] = Field(default_factory=list)
    uncertain_requested_constraints: list[ConstraintStatus] = Field(default_factory=list)
    failed_requested_constraints: list[ConstraintStatus] = Field(default_factory=list)

    matched_derived_matches: list[ConstraintStatus] = Field(default_factory=list)
    uncertain_derived_matches: list[ConstraintStatus] = Field(default_factory=list)
    failed_derived_matches: list[ConstraintStatus] = Field(default_factory=list)
    
    facts: list[ResultFact] = Field(default_factory=list)

    eligibility_status: str | None = None
    match_tier: str | None = None
    selection_reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    why: list[str] = Field(default_factory=list)
    constraint_resolution_results: list[ConstraintResolutionItem] = Field(default_factory=list)


class NormalizedSearchResponse(BaseModel):
    status: SearchStatus

    request_summary: NormalizedRequestSummary | None = None
    results: list[NormalizedSearchResult] = Field(
        default_factory=list
    )

    debug_notes: list[str] = Field(
        default_factory=list
    )