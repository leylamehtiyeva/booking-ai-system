from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.logic.constraint_evidence_resolution import (
    ConstraintResolutionResult,
    resolve_listing_constraints_with_fallback,
)
from app.logic.field_rules import FIELD_RULES
from app.logic.matcher_structured import _match_field_via_rules
from app.logic.result_candidate_selection import CandidateSelectionResult
from app.observability.llm_usage import record_llm_call_failed
from app.observability.trace import RequestTrace
from app.schemas.constraints import ConstraintMappingStatus, UserConstraint
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field as CanonicalField
from app.schemas.listing import ListingRaw
from app.schemas.match import Evidence, EvidenceSource, FieldMatch, Ternary
from app.schemas.result_query import ResultQuery
from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch

logger = logging.getLogger(__name__)


@dataclass
class ResultQueryMatchOutcome:
    """
    status:
    - "ok": results is one ResultQueryMatch per candidate, each with one
      ConstraintMatchResult per constraint - safe to hand to a response
      layer. Each constraint's value/evidence is the FINAL verdict after
      structured matching and (where eligible) the existing textual LLM
      fallback - a response layer never needs to know which stage decided
      it.
    - "no_constraints": ResultQuery.constraints was empty - there is
      nothing to answer, not a signal to run the matcher anyway.
    - "candidate_selection_failed": candidate_selection.status != "ok" -
      this layer never re-derives *why* selection failed, it only refuses
      to match against a selection it can't trust.
    """

    results: list[ResultQueryMatch] = field(default_factory=list)
    status: str = "ok"


def _build_listing_question_fallback_policy() -> FallbackPolicy:
    """
    Distinct from listing_evaluation.py's search-tuned fallback policy.

    must_only=False: search gates fallback on UserConstraint.priority
    (must/forbidden vs nice), but factual_question_agent always sets
    priority="must" as a documented, behaviorally-inert placeholder (see
    ResultQuery/factual_question_agent.py) - this is a factual question,
    not a search requirement, so must_only is semantically meaningless
    here rather than incidentally-always-true.

    top_k is intentionally left at the FallbackPolicy default: it is only
    read by listing_evaluation.py's ranked-list slicing
    (_apply_constraint_fallback_layer), which LISTING_QUESTION never
    calls - candidate_selection.py has already bounded the candidate set
    to the shown listings actually being asked about.

    max_constraints_per_listing=3 mirrors listing_evaluation.py's own
    default - a conservative per-listing LLM-call cap that comfortably
    fits a realistic single chat question (rarely more than a couple of
    amenities asked about at once).
    """
    return FallbackPolicy(
        enabled=True,
        must_only=False,
        run_for_unresolved=True,
        run_for_structured_uncertain=True,
        max_constraints_per_listing=3,
    )


def _constraint_match_from_structured(
    constraint: UserConstraint,
    *,
    field: CanonicalField | None,
    field_match: FieldMatch | None,
) -> ConstraintMatchResult:
    if field is None or field_match is None:
        return ConstraintMatchResult(
            constraint_id=constraint.id,
            raw_text=constraint.raw_text,
            field=None,
            value=Ternary.UNCERTAIN,
            evidence=[],
            reason="unresolved_constraint",
        )

    reason = "" if FIELD_RULES.get(field) is not None else "no_field_rule"
    return ConstraintMatchResult(
        constraint_id=constraint.id,
        raw_text=constraint.raw_text,
        field=field,
        value=field_match.value,
        evidence=list(field_match.evidence),
        reason=reason,
    )


def _constraint_match_from_fallback(
    constraint: UserConstraint,
    result: ConstraintResolutionResult,
) -> ConstraintMatchResult:
    """
    Final verdict comes from the fallback's own decision (already reduced
    from evidence signals to YES/NO/UNCERTAIN by
    constraint_evidence_resolution.py's deterministic _decide_from_analysis
    - never re-derived here from evidence). Evidence is carried through
    into the existing ConstraintMatchResult.evidence contract (no second
    evidence schema); source/path/snippet are preserved as far as the
    existing Evidence model allows - EvidenceSource has no per-provider
    granularity, so every fallback evidence item is tagged
    EvidenceSource.LLM_FALLBACK, with the coarse source string
    (e.g. "description"/"facilities") folded into path only when no
    concrete listing path was recorded.
    """
    field = constraint.mapped_fields[0] if constraint.mapped_fields else None

    evidence = [
        Evidence(
            source=EvidenceSource.LLM_FALLBACK,
            path=item.path or item.source,
            snippet=item.snippet,
        )
        for item in result.evidence
    ]

    return ConstraintMatchResult(
        constraint_id=constraint.id,
        raw_text=constraint.raw_text,
        field=field,
        value=Ternary(result.decision),
        evidence=evidence,
        reason="fallback_resolved",
    )


async def _match_candidate_constraints(
    listing: ListingRaw,
    constraints: list[UserConstraint],
    *,
    policy: FallbackPolicy,
    trace: RequestTrace | None,
) -> list[ConstraintMatchResult]:
    """
    structured matcher -> eligibility (existing
    is_constraint_fallback_eligible, invoked internally by
    resolve_listing_constraints_with_fallback) -> optional textual LLM
    fallback -> final ConstraintMatchResult, per constraint.

    Calls resolve_listing_constraints_with_fallback exactly ONCE for this
    listing, with every constraint - that function already skips
    constraints that are not fallback-eligible (structured YES/NO) and
    caps how many it actually resolves via max_constraints_per_listing,
    so this never re-implements eligibility or issues more LLM calls than
    the existing fallback layer already decides are needed.
    """
    structured_matches_by_field: dict[CanonicalField, FieldMatch] = {}
    matched_field_by_constraint: dict[str, CanonicalField | None] = {}

    for constraint in constraints:
        if constraint.mapping_status != ConstraintMappingStatus.KNOWN or not constraint.mapped_fields:
            matched_field_by_constraint[constraint.id] = None
            continue

        # Mirrors resolve_listing_constraints_with_fallback's own
        # convention (build_resolution_request) of using the first mapped
        # field when a constraint maps to more than one.
        matched_field = constraint.mapped_fields[0]
        matched_field_by_constraint[constraint.id] = matched_field

        if matched_field not in structured_matches_by_field:
            structured_matches_by_field[matched_field] = _match_field_via_rules(
                listing, matched_field
            )

    try:
        fallback_results = await resolve_listing_constraints_with_fallback(
            listing=listing,
            constraints=constraints,
            structured_matches_by_field=structured_matches_by_field,
            policy=policy,
            trace=trace,
        )
    except Exception as exc:
        # A genuine API/network/timeout failure (not the JSON-decode case,
        # which resolve_constraint_via_textual_evidence already turns into
        # a safe UNCERTAIN result) must not crash the whole LISTING_QUESTION
        # turn. Record it into the same telemetry stream real fallback
        # calls use (step="constraint_textual_fallback") so it stays
        # visible, then fall through to the structured verdict for every
        # constraint that would have gone through fallback - which is
        # UNCERTAIN/unresolved by construction, since only those are ever
        # eligible in the first place.
        logger.exception(
            "Listing-question textual fallback failed",
            extra={"step": "constraint_textual_fallback"},
        )
        record_llm_call_failed(
            trace=trace,
            step="constraint_textual_fallback",
            model=policy.model,
            error=str(exc),
        )
        fallback_results = []

    fallback_by_constraint_id = {
        result.constraint_id: result
        for result in fallback_results
        if result.constraint_id is not None
    }

    matches: list[ConstraintMatchResult] = []
    for constraint in constraints:
        fallback_result = fallback_by_constraint_id.get(constraint.id)

        if fallback_result is not None:
            matches.append(_constraint_match_from_fallback(constraint, fallback_result))
            continue

        matched_field = matched_field_by_constraint[constraint.id]
        matches.append(
            _constraint_match_from_structured(
                constraint,
                field=matched_field,
                field_match=(
                    structured_matches_by_field.get(matched_field)
                    if matched_field is not None
                    else None
                ),
            )
        )

    return matches


async def match_result_query_against_candidates(
    *,
    result_query: ResultQuery,
    candidate_selection: CandidateSelectionResult,
    fallback_policy: FallbackPolicy | None = None,
    trace: RequestTrace | None = None,
) -> ResultQueryMatchOutcome:
    """
    Read-only orchestration over the existing structured matcher
    (_match_field_via_rules) and the existing textual LLM fallback
    (resolve_listing_constraints_with_fallback) for each (candidate,
    constraint) pair.

    Does not decide result scope, reference identity, or candidate
    selection - candidate_selection must already be "ok" (that is
    result_candidate_selection.py's responsibility, not re-derived here).
    Does not filter, sort, rank, or collapse per-constraint decisions -
    every candidate keeps every constraint result, in input order. Does
    not run evaluate_listings/ranking/scoring/hard-fail filtering - only
    the factual-resolution primitives are reused.
    """

    if candidate_selection.status != "ok":
        return ResultQueryMatchOutcome(results=[], status="candidate_selection_failed")

    if not result_query.constraints:
        return ResultQueryMatchOutcome(results=[], status="no_constraints")

    policy = fallback_policy or _build_listing_question_fallback_policy()

    results: list[ResultQueryMatch] = []
    for candidate in candidate_selection.candidates:
        constraint_results = await _match_candidate_constraints(
            candidate.listing,
            result_query.constraints,
            policy=policy,
            trace=trace,
        )
        results.append(
            ResultQueryMatch(
                result_id=candidate.result_id,
                constraint_results=constraint_results,
            )
        )

    return ResultQueryMatchOutcome(results=results, status="ok")
