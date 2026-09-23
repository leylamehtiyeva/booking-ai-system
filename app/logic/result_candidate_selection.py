from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.conversation_route import ResultScope
from app.schemas.shown_result_set import ShownResult, ShownResultSet


@dataclass
class CandidateSelectionResult:
    """
    candidates: the ShownResult entries (result_id + full ListingRaw) in
    scope for this turn - never re-derived/re-ranked, always a direct
    subset of the already-stored ShownResultSet.

    status:
    - "ok": candidates is non-empty, safe to match against.
    - "no_shown_results": CURRENT_RESULTS was requested but the shown set
      is empty - there is nothing to match, not a signal to guess.
    - "invalid_target": SPECIFIC_RESULT was requested but target_result_id
      is missing or does not match any shown item - never falls back to
      the first/nearest listing.
    - "not_applicable": result_scope is neither CURRENT_RESULTS nor
      SPECIFIC_RESULT.
    """

    candidates: list[ShownResult] = field(default_factory=list)
    status: str = "ok"


def select_shown_candidates(
    *,
    result_scope: ResultScope,
    target_result_id: str | None,
    shown_result_set: ShownResultSet | None,
) -> CandidateSelectionResult:
    """
    Deterministic, no LLM, no re-resolution of reference - target_result_id
    must already be validated (see conversation_flow._resolve_target_reference)
    before this function is called; it only selects, never decides identity.
    """
    items = shown_result_set.items if shown_result_set is not None else []

    if result_scope == ResultScope.CURRENT_RESULTS:
        if not items:
            return CandidateSelectionResult(candidates=[], status="no_shown_results")
        return CandidateSelectionResult(candidates=list(items), status="ok")

    if result_scope == ResultScope.SPECIFIC_RESULT:
        if target_result_id is None:
            return CandidateSelectionResult(candidates=[], status="invalid_target")

        for item in items:
            if item.result_id == target_result_id:
                return CandidateSelectionResult(candidates=[item], status="ok")

        return CandidateSelectionResult(candidates=[], status="invalid_target")

    return CandidateSelectionResult(candidates=[], status="not_applicable")
