from __future__ import annotations

from typing import Any

_MATCHES_EXCLUDED_LABELS = {"Budget", "Price", "Price per night"}
_MATCHES_EXCLUDED_NAMES = {
    "price_total",
    "price_per_night",
    "listing_price_total",
    "listing_price_per_night_derived",
}

def _is_displayable_match(item: dict) -> bool:
    label = item.get("label")
    name = item.get("name")
    return label not in _MATCHES_EXCLUDED_LABELS and name not in _MATCHES_EXCLUDED_NAMES

def _format_status_line(status_label: str | None, status_text: str | None) -> str | None:
    if not status_text:
        return None

    if status_label == "fully_confirmed_match":
        return f"✅ {status_text}"
    elif status_label == "partially_confirmed_match":
        return f"⚠️ {status_text}"
    elif status_label == "not_matched":
        return f"❌ {status_text}"

    return status_text


def _format_bullets(items: list[str] | None, prefix: str = "- ") -> list[str]:
    out: list[str] = []
    for item in items or []:
        if item:
            out.append(f"{prefix}{item}")
    return out


def _format_constraint_rows(items: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []

    for item in items or []:
        label = item.get("label") or "Requested detail"
        reason = item.get("reason")

        if reason and reason != label:
            out.append(f"- {label} — {reason}")
        else:
            out.append(f"- {label}")

    return out


def _format_constraint_resolution_points(
    result: dict[str, Any],
) -> list[str]:
    """
    Convert canonical constraint_resolution_results into short, user-facing lines.

    Priority:
    1. explicit matches from constraint_resolution_results
    2. fallback to unresolved_constraint_points if already provided upstream

    This keeps answer_generation aligned with canonical payloads.
    """
    points: list[str] = []

    for item in result.get("constraint_resolution_results") or []:
        if not isinstance(item, dict):
            continue

        label = item.get("normalized_text")
        if not label:
            continue

        reason = item.get("reason")
        if isinstance(reason, str) and reason:
            points.append(f"{label} — {reason}")
        else:
            points.append(str(label))

    if points:
        seen: set[str] = set()
        unique: list[str] = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:4]

    fallback_points = result.get("unresolved_constraint_points") or []
    out: list[str] = []
    for item in fallback_points:
        if item:
            text = str(item)
            if text not in out:
                out.append(text)
    return out[:4]


def _format_top_result(result: dict[str, Any], rank: int) -> str:
    title = result.get("title") or "Unknown option"
    url = result.get("url")

    explanation = result.get("answer_explanation") or {}
    status_label = explanation.get("status_label")
    status_text = explanation.get("status_text")

    confirmed = [
        item
        for item in (explanation.get("confirmed") or [])
        if _is_displayable_match(item)
    ]

    needs_confirmation = [
        item
        for item in (explanation.get("needs_confirmation") or [])
        if _is_displayable_match(item)
    ]

    not_satisfied = [
        item
        for item in (explanation.get("not_satisfied") or [])
        if _is_displayable_match(item)
    ]

    lines = [f"{rank}. {title}"]

    status_line = _format_status_line(status_label, status_text)
    if status_line:
        lines.append(status_line)

    price_summary = result.get("price_summary")
    if price_summary:
        lines.append(f"Price: {price_summary}")

    if confirmed:
        lines.append("Matches:")
        lines.extend(_format_constraint_rows(confirmed))

    if needs_confirmation:
        lines.append("Needs confirmation:")
        lines.extend(_format_constraint_rows(needs_confirmation))

    if not_satisfied:
        lines.append("Does not match:")
        lines.extend(_format_constraint_rows(not_satisfied))

    resolved_requested_details = _format_constraint_resolution_points(result)
    if resolved_requested_details:
        existing_text = "\n".join(_format_constraint_rows(confirmed)).casefold()
        extra_details = [
            p for p in resolved_requested_details if p.casefold() not in existing_text
        ]
        if extra_details:
            lines.append("Also confirmed:")
            lines.extend(_format_bullets(extra_details))

    if url:
        lines.append(f"Link: {url}")

    return "\n".join(lines)

def _get_request_context(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Prefer active_intent as canonical runtime state.
    Fall back to request_summary for tests / normalized payloads that do not
    include active_intent.
    """
    return (payload.get("active_intent") or payload.get("request_summary") or {})


def _build_intro(payload: dict[str, Any]) -> str:
    request_ctx = _get_request_context(payload)
    city = request_ctx.get("city")
    check_in = request_ctx.get("check_in")
    check_out = request_ctx.get("check_out")
    results_count = payload.get("results_count") or 0
    shown_count = len(payload.get("top_results") or [])

    parts: list[str] = []

    if city:
        parts.append(f"in {city}")

    if check_in and check_out:
        parts.append(f"for {check_in} to {check_out}")

    tail = ""
    if parts:
        tail = " " + " ".join(parts)

    if results_count > shown_count > 0:
        return (
            f"I found {results_count} option(s) that match your requirements{tail}. "
            f"Here are the top {shown_count} matches."
        )

    return f"I found {shown_count} option(s) that match your requirements{tail}."


def _build_refinement_hint(payload: dict[str, Any]) -> str:
    request_ctx = _get_request_context(payload)
    filters = request_ctx.get("filters") or {}
    constraints = request_ctx.get("constraints") or []

    suggestions: list[str] = []

    if constraints:
        suggestions.append("focus on listings with more fully confirmed requested constraints")
    if filters.get("price"):
        suggestions.append("tighten or relax the budget")
    suggestions.append("narrow by location")
    suggestions.append("change bedroom / bathroom / area requirements")

    unique: list[str] = []
    for s in suggestions:
        if s not in unique:
            unique.append(s)

    return "I can also refine this further — for example, I can " + ", ".join(unique[:3]) + "."


def build_user_answer(payload: dict[str, Any]) -> str:
    """
    Deterministic user-facing formatter.

    This formatter should remain stable, explicit, and safe.
    It prefers active_intent as the source of truth, but can fall back to
    request_summary for normalized/test payloads.
    """
    if payload.get("need_clarification"):
        questions = payload.get("questions") or []
        debug_notes = payload.get("debug_notes") or []

        base_text = ""
        if not questions:
            base_text = "I need one more detail to continue."
        elif len(questions) == 1:
            base_text = questions[0]
        else:
            base_text = "I need a few more details:\n- " + "\n- ".join(questions)

        if debug_notes:
            return base_text + "\n\nDebug notes:\n- " + "\n- ".join(debug_notes)

        return base_text

    top_results = payload.get("top_results") or []
    request_ctx = _get_request_context(payload)
    city = request_ctx.get("city")
    check_in = request_ctx.get("check_in")
    check_out = request_ctx.get("check_out")

    if not top_results:
        if city and check_in and check_out:
            return (
                f"I couldn’t find suitable options in {city} "
                f"for {check_in} to {check_out}. "
                "I can help relax the budget, area, or amenity constraints."
            )
        return "I couldn’t find suitable options. I can help relax or clarify the constraints."

    lines = [_build_intro(payload)]
    lines.append("")

    for idx, result in enumerate(top_results, start=1):
        lines.append(_format_top_result(result, idx))
        if idx < len(top_results):
            lines.append("")

    lines.append("")
    lines.append(_build_refinement_hint(payload))

    return "\n".join(lines)