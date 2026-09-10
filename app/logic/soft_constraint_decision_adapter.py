"""
Smallest clean adapter from the new deterministic SoftConstraintDecision
into the existing constraint_resolution_results contract (the plain-dict
shape produced by app.logic.constraint_evidence_resolution.ConstraintResolutionResult.model_dump()
and consumed by app.logic.listing_evaluation's scoring/filtering and
app.logic.normalize_search_response's ConstraintResolutionItem.model_validate()).

Downstream code must not need to know whether a decision came from the
legacy textual fallback or the new soft-evidence policy - this adapter
is the only place that shape is constructed for the new path.
"""

from __future__ import annotations

from typing import Any

from app.schemas.soft_constraint_decision import ConstraintDecisionValue, FactualState, SoftConstraintDecision

_RESOLUTION_STATUS_BY_DECISION: dict[ConstraintDecisionValue, str] = {
    ConstraintDecisionValue.YES: "matched",
    ConstraintDecisionValue.NO: "failed",
    ConstraintDecisionValue.UNCERTAIN: "uncertain",
}

SOFT_EVIDENCE_SOURCE_STAGE = "soft_evidence"


def soft_constraint_decision_to_resolution_dict(
    decision: SoftConstraintDecision,
    *,
    listing_id: str | None,
    listing_title: str | None,
) -> dict[str, Any]:
    """
    Mirrors ConstraintResolutionResult's field shape exactly (a plain
    dict, not that pydantic model, matching how
    resolve_listing_constraints_with_fallback's results are already
    stored via .model_dump(mode="json") - see _apply_constraint_fallback_layer).

    mapped_fields is always [] here: soft-evidence constraints are not
    mapped to a canonical structured Field, so
    _fails_constraint_resolution's FORBIDDEN inverted-polarity lookup
    finds nothing special and falls back to its default polarity
    (violating_decision="YES") - correct, since PreferenceDirection was
    already applied once inside decide_constraint, so `decision` here
    is already the final, direction-aware verdict.
    """
    return {
        "listing_id": listing_id,
        "listing_title": listing_title,
        "constraint_id": decision.constraint_id,
        "raw_text": decision.raw_text,
        "normalized_text": decision.normalized_text,
        "resolver_type": "soft_evidence",
        "mapped_fields": [],
        "decision": decision.decision.value,
        "resolution_status": _RESOLUTION_STATUS_BY_DECISION[decision.decision],
        "confidence": decision.confidence,
        "reason": decision.reason,
        "evidence": [
            {"snippet": ref.evidence_text, "source": ref.source_type or "other", "path": ref.source_path}
            for ref in decision.evidence
        ],
        "source_stage": SOFT_EVIDENCE_SOURCE_STAGE,
        "structured_value_before": None,
        "explicit_negative": decision.factual_state == FactualState.VIOLATED,
        # Read by _apply_constraint_resolution_scoring / _fails_constraint_resolution
        # exactly like the legacy ConstraintResolutionResult.priority field -
        # not part of ConstraintResolutionItem's own schema (extra fields
        # there are ignored, matching how the legacy dict already carries
        # this key through model_validate() today).
        "priority": decision.priority.value,
    }
