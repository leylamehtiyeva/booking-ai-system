"""
A weekly or monthly rate ("$900 per month") cannot be expressed as a price
filter, which is per night or for the whole stay. Extraction is asked to
keep such a rate out of filters.price, but the guard below does not trust
that: it reads the user's own words and

- drops a price whose amount the user did not state outside the rate
  ($900 per month must not become $900 per night, nor $30 per night);
- keeps a requirement marker so the application asks for a supported budget.
"""

from __future__ import annotations

import re

from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)

PERIOD_BUDGET_TEXTS = {"month": "budget per month", "week": "budget per week"}

_CURRENCY = r"(?:[$€£₼]|usd|eur|gbp|azn|myr|dollars?|euros?|pounds?|manat)"
_AMOUNT = rf"(?:{_CURRENCY}\s?)?\d[\d,]*(?:\.\d+)?(?:\s?{_CURRENCY})?"

_PERIOD_RATE_PATTERNS = [
    # "$900 per month", "900 USD a month", "$900/month", "$250 per week"
    re.compile(rf"(?P<rate>{_AMOUNT}\s*(?:per|a|an|each|every|/)\s*(?P<period>month|week))\b", re.I),
    # "$900 monthly"
    re.compile(rf"(?P<rate>{_AMOUNT}\s+(?P<period>month|week)ly)\b", re.I),
    # "monthly budget is $900", "weekly rent of 250 EUR"
    re.compile(
        rf"(?P<rate>(?P<period>month|week)ly\s+(?:budget|rent|rate|price)\s*(?:is|of|:|=)?\s*{_AMOUNT})",
        re.I,
    ),
]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _number(text: str) -> float | None:
    match = _NUMBER.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def find_period_rates(text: str | None) -> list[tuple[str, str]]:
    """(the user's words, "month" | "week") for each weekly or monthly rate."""
    found: list[tuple[int, int, str, str]] = []
    for pattern in _PERIOD_RATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            start, end = match.span("rate")
            if any(start < e and s < end for s, e, _, _ in found):
                continue
            found.append((start, end, match.group("rate").strip(), match.group("period").lower()))
    return [(rate, period) for _, _, rate, period in sorted(found)]


def is_period_budget(constraint: UserConstraint) -> bool:
    normalized = (constraint.normalized_text or "").strip().lower()
    return normalized in PERIOD_BUDGET_TEXTS.values() or bool(find_period_rates(constraint.raw_text))


def _stated_amounts_outside(text: str, rates: list[str]) -> set[float]:
    for rate in rates:
        text = text.replace(rate, " ")
    return {float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)}


def _marker(rate: str, period: str) -> UserConstraint:
    return UserConstraint(
        raw_text=rate,
        normalized_text=PERIOD_BUDGET_TEXTS[period],
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.NUMERIC,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.NONE,
    )


def guard_period_budget(state, user_text: str, *, previous_price=None):
    """
    Return `state` (an IntentRoute or SearchRequest) without a price filter
    taken from a weekly/monthly rate, and with a marker for that rate.
    A price the user stated separately, or one unchanged from
    `previous_price`, is kept.
    """
    rates = find_period_rates(user_text)
    if not rates:
        return state

    update: dict = {}
    filters = state.filters
    price = filters.price if filters is not None else None
    if price is not None and price != previous_price:
        stated = _stated_amounts_outside(user_text, [rate for rate, _ in rates])
        amounts = [a for a in (price.min_amount, price.max_amount) if a is not None]
        if any(float(a) not in stated for a in amounts) or not amounts:
            update["filters"] = filters.model_copy(update={"price": None})

    kept_price = price if "filters" not in update else None
    has_amount = kept_price is not None and (kept_price.min_amount is not None or kept_price.max_amount is not None)
    constraints = list(state.constraints or [])
    if not has_amount and not any(is_period_budget(c) for c in constraints):
        rate, period = rates[0]
        update["constraints"] = [*constraints, _marker(rate, period)]

    return state.model_copy(update=update) if update else state


def period_budget_text(state) -> str | None:
    for constraint in getattr(state, "constraints", None) or []:
        if is_period_budget(constraint):
            return constraint.raw_text
    return None
