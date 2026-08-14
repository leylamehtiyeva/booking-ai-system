from __future__ import annotations

from typing import Any, List
from app.schemas.search_response import SearchStatus
from app.logic.result_ids import build_result_id
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.search_response import (
    ConstraintResolutionItem,
    ConstraintStatus,
    NormalizedRequestSummary,
    NormalizedSearchResponse,
    NormalizedSearchResult,
    ResultFact,
)

FIELD_DISPLAY_NAMES = {
    "kitchen": "Kitchen",
    "private_bathroom": "Private bathroom",
    "pet_friendly": "Pet policy",
    "parking": "Parking",
    "washing_machine": "Washing machine",
    "balcony": "Balcony",
    "elevator": "Elevator",
    "smoking_allowed": "Smoking policy",
    "parties_allowed": "Party / event policy",
}


def default_uncertain_reason(field_name: str | None) -> str:
    if not field_name:
        return "This requirement is not explicitly confirmed in the listing."

    label = FIELD_DISPLAY_NAMES.get(
        field_name,
        field_name.replace("_", " ").capitalize(),
    )

    return f"{label} is not explicitly confirmed in the listing."


def _request_summary(req: SearchRequest, dropped_requests: List[str]) -> NormalizedRequestSummary:
    """
    Build the normalized request summary from the canonical constraint-centric request.

    Contract:
    - constraints is the only semantic source of truth
    - dropped_requests is debug / normalization residue
    - no legacy field-centric projections are included
    """

    return NormalizedRequestSummary(
        city=req.city,
        check_in=req.check_in.isoformat() if req.check_in else None,
        check_out=req.check_out.isoformat() if req.check_out else None,
        property_types=[x.value if hasattr(x, "value") else str(x) for x in (req.property_types or [])],
        occupancy_types=[x.value if hasattr(x, "value") else str(x) for x in (req.occupancy_types or [])],
        filters=req.filters.model_dump() if req.filters else {},
        dropped_requests=list(dropped_requests or []),
        constraints=[
            {
                "raw_text": c.raw_text,
                "normalized_text": c.normalized_text,
                "priority": c.priority.value,
                "category": c.category.value,
                "mapping_status": c.mapping_status.value,
                "mapped_fields": [f.value for f in c.mapped_fields],
                "evidence_strategy": c.evidence_strategy.value,
            }
            for c in (req.constraints or [])
        ],
    )

def _humanize_constraint_name(name: str | None) -> str:
    if not name:
        return "This requirement"

    mapping = {
        "kitchen": "Kitchen",
        "private_bathroom": "Private bathroom",
        "pet_friendly": "Pet policy",
        "parking": "Parking",
        "washing_machine": "Washing machine",
        "balcony": "Balcony",
        "elevator": "Elevator",
        "smoking_allowed": "Smoking policy",
        "parties_allowed": "Party / event policy",
        "children_allowed": "Child policy",
        "property_type": "Property type",
        "occupancy_type": "Occupancy type",
        "bedrooms": "Bedroom count",
        "bathrooms": "Bathroom count",
        "area_sqm": "Area",
        "price_total": "Price",
    }
    return mapping.get(name, name.replace("_", " ").capitalize())


def _default_uncertain_reason(name: str | None) -> str:
    label = _humanize_constraint_name(name)
    return f"{label} is not explicitly confirmed in the listing."


def _status_bucket(
    name: str,
    value: Any,
    reason: str | None,
    *,
    constraint_meta: dict[str, Any] | None = None,
) -> ConstraintStatus:
    final_reason = reason

    if value == "uncertain" and not final_reason:
        final_reason = _default_uncertain_reason(name)

    return ConstraintStatus(
        name=name,
        status=value,
        reason=final_reason,
        constraint=constraint_meta,
    )
    
    
def _normalize_constraint_key(value: str | None) -> str:
    if not value:
        return ""

    return (
        str(value)
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _build_constraint_lookup(req: SearchRequest) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for c in (req.constraints or []):
        constraint_meta = {
            "id": c.id,
            "raw_text": c.raw_text,
            "normalized_text": c.normalized_text,
            "priority": c.priority.value,
            "category": c.category.value,
            "mapping_status": c.mapping_status.value,
            "mapped_fields": [f.value for f in (c.mapped_fields or [])],
            "evidence_strategy": c.evidence_strategy.value,
        }

        aliases = {
            _normalize_constraint_key(c.normalized_text),
            _normalize_constraint_key(c.raw_text),
        }

        for field in (c.mapped_fields or []):
            aliases.add(_normalize_constraint_key(field.value if hasattr(field, "value") else str(field)))

        for alias in aliases:
            if alias:
                lookup[alias] = constraint_meta

    return lookup


def _resolve_constraint_meta(
    *,
    lookup: dict[str, dict[str, Any]],
    name: str | None = None,
    raw_text: str | None = None,
    normalized_text: str | None = None,
    mapped_fields: list[str] | None = None,
) -> dict[str, Any] | None:
    candidates: list[str] = []

    if normalized_text:
        candidates.append(_normalize_constraint_key(normalized_text))
    if raw_text:
        candidates.append(_normalize_constraint_key(raw_text))
    if name:
        candidates.append(_normalize_constraint_key(name))

    for field_name in (mapped_fields or []):
        candidates.append(_normalize_constraint_key(field_name))

    for key in candidates:
        if key and key in lookup:
            return lookup[key]

    return None


def _constraint_key_for_status(status: ConstraintStatus) -> str:
    if status.constraint:
        constraint_id = status.constraint.get("id")
        if constraint_id:
            return f"id:{constraint_id}"

        normalized_text = status.constraint.get("normalized_text")
        if normalized_text:
            return f"text:{_normalize_constraint_key(normalized_text)}"

    return f"name:{_normalize_constraint_key(status.name)}"


def _status_rank(status: ConstraintStatus) -> tuple[int, int]:
    explicit_negative = bool((status.constraint or {}).get("explicit_negative", False))

    if status.status == "failed" and explicit_negative:
        return (4, 0)
    if status.status == "failed":
        return (3, 0)
    if status.status == "matched":
        return (2, 0)
    return (1, 0)


def _merge_same_constraint_statuses(statuses: list[ConstraintStatus]) -> list[ConstraintStatus]:
    grouped: dict[str, list[ConstraintStatus]] = {}

    for status in statuses:
        key = _constraint_key_for_status(status)
        grouped.setdefault(key, []).append(status)

    merged: list[ConstraintStatus] = []

    for group in grouped.values():
        winner = max(group, key=_status_rank)

        constraint_meta = dict(winner.constraint or {})
        if not constraint_meta:
            for candidate in group:
                if candidate.constraint:
                    constraint_meta = dict(candidate.constraint)
                    break

        canonical_name = (
            constraint_meta.get("normalized_text")
            or winner.name
        )

        reason = winner.reason
        if not reason:
            for candidate in group:
                if candidate.reason:
                    reason = candidate.reason
                    break

        merged.append(
            ConstraintStatus(
                name=canonical_name,
                status=winner.status,
                reason=reason,
                constraint=constraint_meta or None,
            )
        )

    return merged


def _merge_constraint_resolution_statuses(
    item: dict[str, Any],
    matched: list[ConstraintStatus],
    uncertain: list[ConstraintStatus],
    failed: list[ConstraintStatus],
) -> tuple[list[ConstraintStatus], list[ConstraintStatus], list[ConstraintStatus]]:
    all_statuses: list[ConstraintStatus] = []
    all_statuses.extend(matched)
    all_statuses.extend(uncertain)
    all_statuses.extend(failed)

    resolution_results = item.get("constraint_resolution_results") or []
    for raw in resolution_results:
        item_obj = ConstraintResolutionItem.model_validate(raw)

        all_statuses.append(
            ConstraintStatus(
                name=item_obj.normalized_text,
                status=item_obj.resolution_status,
                reason=item_obj.reason,
                constraint={
                    "id": item_obj.constraint_id,
                    "raw_text": item_obj.raw_text,
                    "normalized_text": item_obj.normalized_text,
                    "explicit_negative": item_obj.explicit_negative,
                    "source_stage": item_obj.source_stage,
                },
            )
        )

    merged = _merge_same_constraint_statuses(all_statuses)

    final_matched: list[ConstraintStatus] = []
    final_uncertain: list[ConstraintStatus] = []
    final_failed: list[ConstraintStatus] = []

    for status in merged:
        if status.status == "matched":
            final_matched.append(status)
        elif status.status == "failed":
            final_failed.append(status)
        else:
            final_uncertain.append(status)

    return final_matched, final_uncertain, final_failed

def _split_requested_vs_derived(
    statuses: list[ConstraintStatus],
) -> tuple[list[ConstraintStatus], list[ConstraintStatus]]:
    requested: list[ConstraintStatus] = []
    derived: list[ConstraintStatus] = []

    for status in statuses:
        if status.constraint:
            requested.append(status)
        else:
            derived.append(status)

    return requested, derived


def _collect_constraint_statuses(
    item: dict[str, Any],
    req: SearchRequest,
) -> tuple[list[ConstraintStatus], list[ConstraintStatus], list[ConstraintStatus]]:
    matched: list[ConstraintStatus] = []
    uncertain: list[ConstraintStatus] = []
    failed: list[ConstraintStatus] = []

    constraint_lookup = _build_constraint_lookup(req)

    matches = item.get("matches") or {}
    for field, fm in matches.items():
        if fm is None:
            continue

        field_name = field.value if hasattr(field, "value") else str(field)
        constraint_meta = _resolve_constraint_meta(
            lookup=constraint_lookup,
            name=field_name,
            mapped_fields=[field_name],
        )

        display_name = (
            constraint_meta.get("normalized_text")
            if constraint_meta and constraint_meta.get("normalized_text")
            else field_name
        )

        reason = None
        if getattr(fm, "evidence", None):
            ev = fm.evidence[0]
            reason = getattr(ev, "snippet", None) or getattr(fm, "why", None)
        else:
            reason = getattr(fm, "why", None)

        if fm.value == Ternary.YES:
            matched.append(_status_bucket(display_name, "matched", reason, constraint_meta=constraint_meta))
        elif fm.value == Ternary.UNCERTAIN:
            uncertain.append(_status_bucket(display_name, "uncertain", reason, constraint_meta=constraint_meta))
        else:
            failed.append(_status_bucket(display_name, "failed", reason, constraint_meta=constraint_meta))

    for nr in (item.get("numeric_results") or []):
        attr_name = getattr(nr, "attribute", "numeric_constraint")
        constraint_meta = _resolve_constraint_meta(
            lookup=constraint_lookup,
            name=attr_name,
            mapped_fields=[attr_name],
        )

        display_name = (
            constraint_meta.get("normalized_text")
            if constraint_meta and constraint_meta.get("normalized_text")
            else attr_name
        )

        reason = getattr(nr, "why", None)

        if nr.value == Ternary.YES:
            matched.append(_status_bucket(display_name, "matched", reason, constraint_meta=constraint_meta))
        elif nr.value == Ternary.UNCERTAIN:
            uncertain.append(_status_bucket(display_name, "uncertain", reason, constraint_meta=constraint_meta))
        else:
            failed.append(_status_bucket(display_name, "failed", reason, constraint_meta=constraint_meta))

    property_result = item.get("property_result")
    if property_result is not None:
        constraint_meta = _resolve_constraint_meta(
            lookup=constraint_lookup,
            name="property_type",
            mapped_fields=["property_type"],
        )
        display_name = (
            constraint_meta.get("normalized_text")
            if constraint_meta and constraint_meta.get("normalized_text")
            else "property_type"
        )

        if property_result.value == Ternary.YES:
            matched.append(_status_bucket(display_name, "matched", property_result.why, constraint_meta=constraint_meta))
        elif property_result.value == Ternary.UNCERTAIN:
            uncertain.append(_status_bucket(display_name, "uncertain", property_result.why, constraint_meta=constraint_meta))
        else:
            failed.append(_status_bucket(display_name, "failed", property_result.why, constraint_meta=constraint_meta))

    occupancy_result = item.get("occupancy_result")
    if occupancy_result is not None:
        constraint_meta = _resolve_constraint_meta(
            lookup=constraint_lookup,
            name="occupancy_type",
            mapped_fields=["occupancy_type"],
        )
        display_name = (
            constraint_meta.get("normalized_text")
            if constraint_meta and constraint_meta.get("normalized_text")
            else "occupancy_type"
        )

        if occupancy_result.value == Ternary.YES:
            matched.append(_status_bucket(display_name, "matched", occupancy_result.why, constraint_meta=constraint_meta))
        elif occupancy_result.value == Ternary.UNCERTAIN:
            uncertain.append(_status_bucket(display_name, "uncertain", occupancy_result.why, constraint_meta=constraint_meta))
        else:
            failed.append(_status_bucket(display_name, "failed", occupancy_result.why, constraint_meta=constraint_meta))

    return _merge_constraint_resolution_statuses(item, matched, uncertain, failed)

def _collect_facts(item: dict[str, Any], req: SearchRequest) -> list[ResultFact]:
    listing = item.get("listing")
    facts: list[ResultFact] = []

    property_result = item.get("property_result")
    if property_result is not None and getattr(property_result, "actual_value", None) is not None:
        facts.append(
            ResultFact(
                key="property_type",
                value=property_result.actual_value,
                source="property_semantics",
            )
        )

    occupancy_result = item.get("occupancy_result")
    if occupancy_result is not None and getattr(occupancy_result, "actual_value", None) is not None:
        facts.append(
            ResultFact(
                key="occupancy_type",
                value=occupancy_result.actual_value,
                source="property_semantics",
            )
        )

    for nr in (item.get("numeric_results") or []):
        attr = getattr(nr, "attribute", None)
        actual_value = getattr(nr, "actual_value", None)
        if attr is not None and actual_value is not None:
            facts.append(
                ResultFact(
                    key=attr,
                    value=actual_value,
                    source="numeric_filters",
                )
            )

    if req.check_in and req.check_out:
        night_count = (req.check_out - req.check_in).days
        if night_count > 0:
            facts.append(
                ResultFact(
                    key="night_count",
                    value=night_count,
                    source="request",
                )
            )

    if listing is not None:
        price = getattr(listing, "price", None)
        currency = getattr(listing, "currency", None)

        if price is not None:
            facts.append(ResultFact(key="listing_price_total", value=price, source="listing"))
        if currency is not None:
            facts.append(ResultFact(key="listing_currency", value=currency, source="listing"))

        if price is not None and req.check_in and req.check_out:
            night_count = (req.check_out - req.check_in).days
            if night_count > 0:
                facts.append(
                    ResultFact(
                        key="listing_price_per_night_derived",
                        value=round(float(price) / night_count, 2),
                        source="derived",
                    )
                )

    if req.filters and req.filters.price:
        pf = req.filters.price
        if pf.max_amount is not None:
            if pf.scope == "per_night" and req.check_in and req.check_out:
                night_count = (req.check_out - req.check_in).days
                if night_count > 0:
                    facts.append(
                        ResultFact(
                            key="budget_total_derived",
                            value=round(float(pf.max_amount) * night_count, 2),
                            source="derived",
                        )
                    )
            elif pf.scope == "total_stay":
                facts.append(
                    ResultFact(
                        key="budget_total_derived",
                        value=float(pf.max_amount),
                        source="derived",
                    )
                )

        if pf.currency is not None:
            facts.append(
                ResultFact(
                    key="budget_currency",
                    value=pf.currency,
                    source="request",
                )
            )

        if pf.scope is not None:
            facts.append(
                ResultFact(
                    key="budget_scope",
                    value=pf.scope,
                    source="request",
                )
            )

    return facts


def normalize_search_response(
    req: SearchRequest,
    ranked: list[dict[str, Any]],
    *,
    top_n: int,
    dropped_requests: list[str],
    debug_notes: list[str] | None = None,
) -> NormalizedSearchResponse:
    request_summary = _request_summary(req, dropped_requests)
    results: list[NormalizedSearchResult] = []

    for item in ranked[: max(0, top_n)]:
        listing = item.get("listing")
        matched, uncertain, failed = _collect_constraint_statuses(item, req)
        facts = _collect_facts(item, req)

        matched_requested_constraints, matched_derived_matches = _split_requested_vs_derived(matched)
        uncertain_requested_constraints, uncertain_derived_matches = _split_requested_vs_derived(uncertain)
        failed_requested_constraints, failed_derived_matches = _split_requested_vs_derived(failed)

        results.append(
            NormalizedSearchResult(
                result_id=build_result_id(listing),
                title=item.get("listing_name") or getattr(listing, "name", "Unknown"),
                url=getattr(listing, "url", None),
                score=float(item.get("score", 0.0)),
                eligibility_status=item.get("eligibility_status"),
                match_tier=item.get("match_tier"),
                selection_reasons=item.get("selection_reasons") or [],
                blocking_reasons=item.get("blocking_reasons") or [],
                constraint_resolution_results=item.get("constraint_resolution_results", []),
                matched_constraints=matched,
                uncertain_constraints=uncertain,
                failed_constraints=failed,
                matched_requested_constraints=matched_requested_constraints,
                uncertain_requested_constraints=uncertain_requested_constraints,
                failed_requested_constraints=failed_requested_constraints,
                matched_derived_matches=matched_derived_matches,
                uncertain_derived_matches=uncertain_derived_matches,
                failed_derived_matches=failed_derived_matches,
                facts=facts,
                why=item.get("why") or [],
            )
        )

    return NormalizedSearchResponse(
        status=SearchStatus.RESULTS,
        request_summary=request_summary,
        results=results,
        debug_notes=debug_notes or [],
    )