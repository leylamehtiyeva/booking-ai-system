from __future__ import annotations

from typing import Any


def summarize_selection_signals(item: dict[str, Any]) -> dict[str, int]:
    matched_constraints = item.get("matched_constraints") or []
    uncertain_constraints = item.get("uncertain_constraints") or []
    failed_constraints = item.get("failed_constraints") or []

    if not matched_constraints and not uncertain_constraints and not failed_constraints:
        matched_constraints, uncertain_constraints, failed_constraints = _derive_constraint_buckets(item)

    must_total = int(item.get("matched_must_total", 0))
    must_matched = int(item.get("matched_must_count", 0))
    must_failed = 0
    must_uncertain = 0
    
    uncertain_names = {
        str(c.get("name") if isinstance(c, dict) else getattr(c, "name", "")).strip().casefold()
        for c in uncertain_constraints
    }
    failed_names = {
        str(c.get("name") if isinstance(c, dict) else getattr(c, "name", "")).strip().casefold()
        for c in failed_constraints
    }

    unknown_found_count = 0
    unknown_uncertain_count = 0
    explicit_negative_count = 0

    for result in item.get("constraint_resolution_results", []) or []:
        resolution_status = str(result.get("resolution_status", "")).strip().lower()
        explicit_negative = bool(result.get("explicit_negative", False))

        if resolution_status == "matched":
            unknown_found_count += 1
        elif resolution_status == "uncertain":
            unknown_uncertain_count += 1
        elif resolution_status == "failed":
            must_failed += 1

        if explicit_negative:
            explicit_negative_count += 1

    # Count uncertain/failed canonical must constraints
    for name in failed_names:
        if name:
            must_failed += 1

    for name in uncertain_names:
        if name:
            must_uncertain += 1

    return {
        "must_total": must_total,
        "must_matched": must_matched,
        "must_uncertain": must_uncertain,
        "must_failed": must_failed,
        "unknown_found_count": unknown_found_count,
        "unknown_uncertain_count": unknown_uncertain_count,
        "explicit_negative_count": explicit_negative_count,
    }


def classify_ranked_item(item: dict[str, Any]) -> dict[str, Any]:
    signals = summarize_selection_signals(item)

    must_total = signals["must_total"]
    must_matched = signals["must_matched"]
    must_uncertain = signals["must_uncertain"]
    must_failed = signals["must_failed"]
    unknown_uncertain_count = signals["unknown_uncertain_count"]
    explicit_negative_count = signals["explicit_negative_count"]

    property_result = item.get("property_result")
    occupancy_result = item.get("occupancy_result")

    property_value = getattr(property_result, "value", None)
    occupancy_value = getattr(occupancy_result, "value", None)

    property_value_str = str(getattr(property_value, "value", property_value)).lower()
    occupancy_value_str = str(getattr(occupancy_value, "value", occupancy_value)).lower()

    property_confirmed = property_value_str == "yes"
    occupancy_confirmed = occupancy_value_str == "yes"

    property_failed = property_value_str == "no"
    occupancy_failed = occupancy_value_str == "no"

    property_uncertain = property_value_str == "uncertain"
    occupancy_uncertain = occupancy_value_str == "uncertain"

    selection_reasons: list[str] = []
    blocking_reasons: list[str] = []

    resolution_results = item.get("constraint_resolution_results") or []

    has_negative_resolution = any(
        str(getattr(result.get("status") if isinstance(result, dict) else result, "value", result.get("status") if isinstance(result, dict) else result)).upper()
        == "NO"
        for result in resolution_results
    )

    if must_failed > 0:
        blocking_reasons.append("failed required constraints")

    if explicit_negative_count > 0:
        blocking_reasons.append("explicit negative evidence for requested constraints")

    if has_negative_resolution:
        blocking_reasons.append("negative constraint resolution result")

    if property_failed:
        blocking_reasons.append("property type does not match requested type")

    if occupancy_failed:
        blocking_reasons.append("occupancy type does not match requested type")

    if blocking_reasons:
        eligibility_status = "ineligible"
        match_tier = "weak"

    else:
        eligibility_status = "eligible"

        no_uncertainty = (
            must_uncertain == 0
            and unknown_uncertain_count == 0
            and not property_uncertain
            and not occupancy_uncertain
        )

        no_failures = (
            must_failed == 0
            and explicit_negative_count == 0
            and not has_negative_resolution
            and not property_failed
            and not occupancy_failed
        )

        all_must_confirmed = must_total > 0 and must_matched == must_total

        has_confirmed_core_signal = (
            all_must_confirmed
            or property_confirmed
            or occupancy_confirmed
        )

        if no_failures and no_uncertainty and has_confirmed_core_signal:
            match_tier = "strong"

            if all_must_confirmed:
                selection_reasons.append("all required constraints are confirmed")
            else:
                selection_reasons.append("core request is confirmed")

        elif must_total > 0 and must_matched == 0:
            match_tier = "weak"

            if (
                must_uncertain > 0
                or unknown_uncertain_count > 0
                or property_uncertain
                or occupancy_uncertain
            ):
                selection_reasons.append("no required constraints are confirmed")
            else:
                selection_reasons.append("weak match for required constraints")

        elif no_failures:
            match_tier = "partial"

            if (
                must_uncertain > 0
                or unknown_uncertain_count > 0
                or property_uncertain
                or occupancy_uncertain
            ):
                selection_reasons.append("some requested constraints are not fully confirmed")
            else:
                selection_reasons.append("matches the core request")

        else:
            match_tier = "weak"

    classified = dict(item)
    classified["selection_signals"] = signals
    classified["eligibility_status"] = eligibility_status
    classified["match_tier"] = match_tier
    classified["selection_reasons"] = selection_reasons
    classified["blocking_reasons"] = blocking_reasons

    return classified


def select_ranked_items(items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    classified = [classify_ranked_item(item) for item in items]

    def is_eligible(x):
        return x.get("eligibility_status") == "eligible"

    strong = [x for x in classified if is_eligible(x) and x.get("match_tier") == "strong"]
    partial = [x for x in classified if is_eligible(x) and x.get("match_tier") == "partial"]

    
    weak = [
        x for x in classified
        if is_eligible(x)
        and x.get("match_tier") == "weak"
        and not x.get("blocking_reasons") 
    ]

    def sort_by_score(items):
        return sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    strong = sort_by_score(strong)
    partial = sort_by_score(partial)
    weak = sort_by_score(weak)

    selected = []

    for bucket in (strong, partial, weak):
        remaining = max(0, top_n - len(selected))
        if remaining <= 0:
            break
        selected.extend(bucket[:remaining])

    return selected


def _derive_constraint_buckets(
    item: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    matches = item.get("matches") or {}
    for field, fm in matches.items():
        if fm is None:
            continue

        name = field.value if hasattr(field, "value") else str(field)
        ternary_value = getattr(getattr(fm, "value", None), "value", None) or str(getattr(fm, "value", "")).lower()

        status_item = {"name": name}

        if ternary_value == "yes":
            matched.append(status_item)
        elif ternary_value == "uncertain":
            uncertain.append(status_item)
        elif ternary_value == "no":
            failed.append(status_item)

    for result in item.get("numeric_results", []) or []:
        name = str(getattr(result, "attribute", "")).strip()
        ternary_value = getattr(getattr(result, "value", None), "value", None) or str(getattr(result, "value", "")).lower()
        if not name:
            continue

        status_item = {"name": name}

        if ternary_value == "yes":
            matched.append(status_item)
        elif ternary_value == "uncertain":
            uncertain.append(status_item)
        elif ternary_value == "no":
            failed.append(status_item)

    for attr_name in ("property_result", "occupancy_result"):
        result = item.get(attr_name)
        if result is None:
            continue

        name = attr_name.replace("_result", "")
        ternary_value = getattr(getattr(result, "value", None), "value", None) or str(getattr(result, "value", "")).lower()

        status_item = {"name": name}

        if ternary_value == "yes":
            matched.append(status_item)
        elif ternary_value == "uncertain":
            uncertain.append(status_item)
        elif ternary_value == "no":
            failed.append(status_item)

    return matched, uncertain, failed