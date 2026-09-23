"""
Step 4 smoke example: ResultQuery + selected ShownResult candidates ->
existing structured matcher, with no search/LLM/network involved at all.

Run: python3 scripts/debug/result_query_matching_smoke.py
"""

from __future__ import annotations

from app.logic.result_candidate_selection import CandidateSelectionResult
from app.logic.result_query_matching import match_result_query_against_candidates
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.result_query import ResultQuery
from app.schemas.shown_result_set import ShownResult

candidates = [
    ShownResult(
        result_id="H1",
        listing=ListingRaw(
            id="H1",
            name="Hilton Baku",
            facilities=[{"name": "Private parking"}],
        ),
    ),
    ShownResult(
        result_id="H2",
        listing=ListingRaw(
            id="H2",
            name="Marriott Baku",
            description="No parking available on site.",
        ),
    ),
    ShownResult(
        result_id="H3",
        listing=ListingRaw(
            id="H3",
            name="Boulevard Hotel",
            description="A cozy hotel close to the seaside boulevard.",
        ),
    ),
]

result_query = ResultQuery(
    constraints=[
        UserConstraint(
            raw_text="parking",
            normalized_text="parking",
            priority=ConstraintPriority.MUST,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.PARKING],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        )
    ]
)

candidate_selection = CandidateSelectionResult(candidates=candidates, status="ok")

outcome = match_result_query_against_candidates(
    result_query=result_query,
    candidate_selection=candidate_selection,
)

print(f"status: {outcome.status}")
for result in outcome.results:
    for cr in result.constraint_results:
        evidence = cr.evidence[0].snippet if cr.evidence else None
        print(f"  {result.result_id} -> {cr.value.value} (evidence: {evidence!r})")
