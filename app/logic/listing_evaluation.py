from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from app.logic.constraint_evidence_resolution import (
    resolve_listing_constraints_with_fallback,
)
from app.logic.matcher_structured import (
    match_listing_structured,
)
from app.logic.numeric_filters import (
    evaluate_numeric_filters,
)

from app.logic.property_semantics import (
    match_occupancy_types,
    match_property_types,
)
from app.observability.trace import RequestTrace
from app.schemas.fallback_policy import (
    FallbackPolicy,
)
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest


@dataclass
class ListingEvaluationResult:
    ranked_items: list[dict[str, Any]]
    debug_notes: list[str]
    
    
def _fails_must(matches: dict[Field, Any], must_fields: List[Field] | None) -> bool:
    """Strict must-have filter: if any must field is explicitly NO -> reject."""
    for f in (must_fields or []):
        fm = matches.get(f)
        if fm is not None and fm.value == Ternary.NO:
            return True
    return False

def _fails_numeric_filters(numeric_results: List[Any] | None) -> bool:
    """
    Strict numeric filter: if any numeric constraint is explicitly NO -> reject.
    UNCERTAIN is allowed.
    """
    for r in (numeric_results or []):
        if r.value == Ternary.NO:
            return True
    return False


def _priority_value(priority: Any) -> str:
    return getattr(priority, "value", priority)


def _mapping_status_value(mapping_status: Any) -> str:
    return getattr(mapping_status, "value", mapping_status)


def _constraints_by_priority(req: SearchRequest) -> tuple[list[Any], list[Any], list[Any]]:
    constraints = list(req.constraints or [])

    must_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "must"
    ]
    nice_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "nice"
    ]
    forbidden_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "forbidden"
    ]
    return must_constraints, nice_constraints, forbidden_constraints


def _known_mapped_fields(constraints: list[Any]) -> list[Field]:
    out: list[Field] = []
    seen: set[Field] = set()

    for c in constraints:
        if _mapping_status_value(getattr(c, "mapping_status", None)) != "known":
            continue

        for f in getattr(c, "mapped_fields", []) or []:
            if f not in seen:
                seen.add(f)
                out.append(f)

    return out

def _constraint_key(c: Any) -> str:
    constraint_id = getattr(c, "id", None)
    if constraint_id:
        return f"id:{constraint_id}"

    normalized_text = str(getattr(c, "normalized_text", "") or "").strip().casefold()
    if normalized_text:
        return f"text:{normalized_text}"

    raw_text = str(getattr(c, "raw_text", "") or "").strip().casefold()
    return f"raw:{raw_text}"


def _resolution_key(result: dict[str, Any]) -> str:
    constraint_id = result.get("constraint_id")
    if constraint_id:
        return f"id:{constraint_id}"

    normalized_text = str(result.get("normalized_text", "") or "").strip().casefold()
    if normalized_text:
        return f"text:{normalized_text}"

    raw_text = str(result.get("raw_text", "") or "").strip().casefold()
    return f"raw:{raw_text}"


def _unresolved_must_textual_constraints(req: SearchRequest) -> list[Any]:
    out: list[Any] = []

    for c in req.constraints or []:
        if _priority_value(getattr(c, "priority", None)) != "must":
            continue

        if _mapping_status_value(getattr(c, "mapping_status", None)) != "unresolved":
            continue

        evidence_strategy = getattr(c, "evidence_strategy", None)
        evidence_strategy_value = getattr(evidence_strategy, "value", evidence_strategy)

        if evidence_strategy_value not in {"textual", "none", None}:
            continue

        out.append(c)

    return out


def _make_uncertain_resolution_for_unresolved_must(
    *,
    constraint: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    listing = item.get("listing")

    return {
        "listing_id": getattr(listing, "id", None),
        "listing_title": getattr(listing, "name", None) or item.get("listing_name"),

        "constraint_id": getattr(constraint, "id", None),
        "raw_text": getattr(constraint, "raw_text", None) or getattr(constraint, "normalized_text", ""),
        "normalized_text": getattr(constraint, "normalized_text", None) or getattr(constraint, "raw_text", ""),

        "resolver_type": "textual",
        "decision": "UNCERTAIN",
        "resolution_status": "uncertain",
        "confidence": 0.0,
        "reason": "Required textual constraint was not resolved by structured matching or textual fallback.",

        "evidence": [],
        "source_stage": "coverage_normalization",
        "structured_value_before": None,
        "explicit_negative": False,

        # Internal fields used before response normalization.
        "priority": "must",
        "mapping_status": "unresolved",
        "evidence_strategy": "textual",
    }


def _ensure_unresolved_must_constraints_are_represented(
    req: SearchRequest,
    ranked: list[dict[str, Any]],
) -> None:
    """
    Safety invariant:

    Every MUST constraint from constraints[] must have a final decision signal
    before final eligibility is computed.

    Unresolved textual MUST constraints cannot silently disappear.
    If fallback did not produce YES/NO/UNCERTAIN, we add synthetic UNCERTAIN.
    """
    unresolved_must_constraints = _unresolved_must_textual_constraints(req)

    if not unresolved_must_constraints:
        return

    for item in ranked:
        results = list(item.get("constraint_resolution_results") or [])
        existing_keys = {_resolution_key(r) for r in results if isinstance(r, dict)}

        for constraint in unresolved_must_constraints:
            key = _constraint_key(constraint)

            if key in existing_keys:
                continue

            results.append(
                _make_uncertain_resolution_for_unresolved_must(
                    constraint=constraint,
                    item=item,
                )
            )

        item["constraint_resolution_results"] = results




def _rank_structured(req: SearchRequest, listings: List[ListingRaw]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []

    must_constraints, nice_constraints, _ = _constraints_by_priority(req)
    structured_must_fields = _known_mapped_fields(must_constraints)
    
    for lst in listings:
        report = match_listing_structured(lst, req)
        numeric_results = evaluate_numeric_filters(
            lst,
            req.filters,
            check_in=req.check_in,
            check_out=req.check_out,
        )
        property_result = match_property_types(lst, req.property_types)
        occupancy_result = match_occupancy_types(lst, req.occupancy_types)

        # Apply strict filtering only to mapped canonical MUST constraints
        if _fails_must(report.matches, structured_must_fields):
            continue


        score, must_yes, must_total, why = _score_listing(
            req,
            report.matches,
            numeric_results=numeric_results,
        )

        if property_result is not None:
            why.append(property_result.why)

        if occupancy_result is not None:
            why.append(occupancy_result.why)

        ranked.append(
            {
                "listing_name": lst.name,
                "listing_id": getattr(lst, "id", None),
                "report": report,
                "matches": report.matches,
                "numeric_results": numeric_results,
                "property_result": property_result,
                "occupancy_result": occupancy_result,
                "score": score,
                "matched_must_count": must_yes,
                "matched_must_total": must_total,
                "why": why,
                "listing": lst,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


    
def _format_match_why(field: Field, fm: Any) -> str:
    if fm is None:
        return f"{field.name}: missing match"

    if fm.value == Ternary.YES:
        if fm.evidence and fm.evidence[0].snippet:
            return f"{field.name}: {fm.evidence[0].snippet}"
        return f"{field.name}: matched"

    if fm.value == Ternary.UNCERTAIN:
        return f"{field.name}: maybe (needs check)"

    return f"{field.name}: not found"


def _score_listing(
    req: SearchRequest,
    matches: dict[Field, Any],
    numeric_results: List[Any] | None = None,
) -> Tuple[float, int, int, List[str]]:
    """
    Canonical scoring.

    Structured scoring is applied only to canonical constraints that:
    - have priority must/nice
    - are known
    - have mapped_fields

    Unresolved constraints are handled later by fallback and
    constraint_resolution_results scoring.
    """
    score = 0.0
    why: List[str] = []

    must_constraints, nice_constraints, _ = _constraints_by_priority(req)

    structured_must_fields = _known_mapped_fields(must_constraints)
    structured_nice_fields = _known_mapped_fields(nice_constraints)

    must_total = len(structured_must_fields)
    must_yes = 0

    for f in structured_must_fields:
        fm = matches.get(f)

        if fm is None:
            why.append(_format_match_why(f, fm))
            continue

        if fm.value == Ternary.YES:
            score += 10
            must_yes += 1
        elif fm.value == Ternary.UNCERTAIN:
            pass
        else:
            score -= 100

        why.append(_format_match_why(f, fm))

    for f in structured_nice_fields:
        fm = matches.get(f)
        if fm and fm.value == Ternary.YES:
            score += 1
            if fm.evidence and fm.evidence[0].snippet:
                why.append(f"+ {f.name}: {fm.evidence[0].snippet}")
            else:
                why.append(f"+ {f.name}: matched")

    for nr in (numeric_results or []):
        if nr.value == Ternary.YES:
            score += 10
        elif nr.value == Ternary.UNCERTAIN:
            pass
        else:
            score -= 100

        why.append(nr.why)

    return score, must_yes, must_total, why


def _build_fallback_policy(
    *,
    fallback_top_k: int,
) -> FallbackPolicy:
    return FallbackPolicy(
        enabled=True,
        top_k=fallback_top_k,
        must_only=True,
        run_for_unresolved=True,
        run_for_structured_uncertain=True,
        max_constraints_per_listing=3,
    )

async def _apply_constraint_fallback_layer(
    req: SearchRequest,
    ranked: list[dict],
    *,
    policy: FallbackPolicy,
    trace: RequestTrace | None = None,
) -> None:
    if not policy.enabled:
        for item in ranked:
            item["constraint_resolution_results"] = []
        return

    top_k = policy.normalized_top_k()

    for item in ranked[:top_k]:
        listing = item.get("listing")
        if listing is None:
            item["constraint_resolution_results"] = []
            continue

        results = await resolve_listing_constraints_with_fallback(
            listing=listing,
            constraints=req.constraints or [],
            structured_matches_by_field=item.get("matches", {}),
            policy=policy,
            trace=trace,
        )

        item["constraint_resolution_results"] = [
            r.model_dump(mode="json") for r in results
        ]

    for item in ranked[top_k:]:
        item["constraint_resolution_results"] = []


def _apply_constraint_resolution_scoring(ranked_items: list[dict]) -> list[dict]:
    for item in ranked_items:
        delta = 0.0
        extra_why: list[str] = []

        for r in item.get("constraint_resolution_results", []) or []:
            label = r.get("normalized_text") or "constraint"
            decision = str(r.get("decision") or "").upper()
            priority = str(r.get("priority") or "").lower()

            if priority == "must":
                if decision == "YES":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: {label} confirmed by listing text")
                elif decision == "NO":
                    delta -= 100.0
                    extra_why.append(f"CONSTRAINT_FAIL: must constraint '{label}' not satisfied")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: must constraint '{label}' not confirmed")

            elif priority in {"nice", "nice_to_have"}:
                if decision == "YES":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: nice-to-have '{label}' confirmed by listing text")
                elif decision == "NO":
                    extra_why.append(f"CONSTRAINT_NO_MATCH: nice-to-have '{label}' not satisfied")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: nice-to-have '{label}' not confirmed")

            elif priority == "forbidden":
                if decision == "YES":
                    delta -= 100.0
                    extra_why.append(f"CONSTRAINT_FAIL: forbidden constraint '{label}' detected")
                elif decision == "NO":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: forbidden constraint '{label}' not detected")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: forbidden constraint '{label}' unclear")

            else:
                # Defensive fallback: do not reward unknown priority.
                if decision == "YES":
                    extra_why.append(f"CONSTRAINT_MATCH: {label} confirmed, but priority is unknown")
                elif decision == "NO":
                    extra_why.append(f"CONSTRAINT_NO_MATCH: {label} not satisfied, but priority is unknown")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: {label} not confirmed")

        if delta != 0:
            item["score"] = float(item.get("score", 0.0)) + delta

        why = list(item.get("why") or [])
        why.extend(extra_why)
        item["why"] = why

    ranked_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked_items



async def evaluate_listings(
    req: SearchRequest,
    listings: list[ListingRaw],
    *,
    fallback_policy: FallbackPolicy | None = None,
    trace: RequestTrace | None = None,
) -> ListingEvaluationResult:
    """
    Evaluate retrieved listings against the current SearchRequest.

    This stage owns:
    - structured matching
    - numeric/property/occupancy evaluation
    - preliminary scoring
    - textual fallback
    - constraint coverage normalization
    - final deterministic filtering

    Retrieval, final result selection and response normalization
    are outside this stage.
    """

    if trace is None:
        trace = RequestTrace()


    if not listings:
        return ListingEvaluationResult(
            ranked_items=[],
            debug_notes=[
                (
                    "No listings remained after "
                    "initial city/date/occupancy "
                    "filtering."
                ),
                (
                    f"city={req.city}, "
                    f"check_in={req.check_in}, "
                    f"check_out={req.check_out}"
                ),
            ],
        )

    # Structured matching + preliminary ranking
    with trace.step(
        "structured_ranking",
        listings_count=len(listings),
    ):
        ranked = _rank_structured(
            req,
            listings,
        )

    # Textual fallback policy
    if fallback_policy is None:
        fallback_policy = (
            _build_fallback_policy(
                fallback_top_k=5,
            )
        )

    # Resolve textual / uncertain evidence
    with trace.step(
        "constraint_fallback_layer",
        ranked_count=len(ranked),
    ):
        await _apply_constraint_fallback_layer(
            req,
            ranked,
            policy=fallback_policy,
            trace=trace,
        )

    # Ensure unresolved MUST constraints
    # cannot silently disappear.
    with trace.step(
        "constraint_coverage_normalization",
        ranked_count=len(ranked),
    ):
        (
            _ensure_unresolved_must_constraints_are_represented(
                req,
                ranked,
            )
        )

    # Apply fallback evidence to score.
    with trace.step(
        "fallback_scoring"
    ):
        ranked = (
            _apply_constraint_resolution_scoring(
                ranked
            )
        )

    # Final deterministic filtering
    with trace.step(
        "post_fallback_structured_filtering",
        ranked_count=len(ranked),
    ):
        must_constraints, _, _ = (
            _constraints_by_priority(req)
        )

        structured_must_fields = (
            _known_mapped_fields(
                must_constraints
            )
        )

        ranked = [
            item
            for item in ranked
            if not _fails_must(
                item["matches"],
                structured_must_fields,
            )
            and not _fails_numeric_filters(
                item.get(
                    "numeric_results"
                )
            )
        ]

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

    if not ranked:
        debug_notes = [
            (
                "No listings remained after "
                "structured filtering."
            )
        ]

        price = (
            req.filters.price
            if req.filters
            else None
        )

        if (
            price
            and price.max_amount is not None
        ):
            debug_notes.append(
                (
                    "Active price filter: "
                    f"max {price.max_amount} "
                    f"{price.currency or 'USD'} "
                    f"{price.scope or ''}"
                ).strip()
            )

        if (
            req.filters
            and req.filters.bedrooms_min
            is not None
        ):
            debug_notes.append(
                (
                    "Active bedrooms filter: "
                    f"min "
                    f"{req.filters.bedrooms_min}"
                )
            )

        if (
            req.filters
            and req.filters.area_sqm_min
            is not None
        ):
            debug_notes.append(
                (
                    "Active area filter: "
                    f"min "
                    f"{req.filters.area_sqm_min} sqm"
                )
            )

        if req.property_types:
            debug_notes.append(
                "Active property types: "
                + ", ".join(
                    property_type.value
                    for property_type
                    in req.property_types
                )
            )

        must_constraint_names = [
            constraint.normalized_text
            for constraint
            in (req.constraints or [])
            if _priority_value(
                constraint.priority
            )
            == "must"
        ]

        if must_constraint_names:
            debug_notes.append(
                "Active must constraints: "
                + ", ".join(
                    must_constraint_names
                )
            )

        nice_constraint_names = [
            constraint.normalized_text
            for constraint
            in (req.constraints or [])
            if _priority_value(
                constraint.priority
            )
            == "nice"
        ]

        if nice_constraint_names:
            debug_notes.append(
                "Optional constraints: "
                + ", ".join(
                    nice_constraint_names
                )
            )

        return ListingEvaluationResult(
            ranked_items=[],
            debug_notes=debug_notes,
        )

    return ListingEvaluationResult(
        ranked_items=ranked,
        debug_notes=[],
    )