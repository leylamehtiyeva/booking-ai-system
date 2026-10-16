from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.logic.budget_period import period_budget_text


@dataclass
class ResolvedSearchContext:
    city: str | None
    check_in: date | None
    check_out: date | None
    questions: list[str]

    @property
    def need_clarification(self) -> bool:
        return len(self.questions) > 0


def parse_iso_date(value: str | date | None) -> date | None:
    if not value:
        return None

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(value)
    except Exception:
        return None


def resolve_required_search_context(intent) -> ResolvedSearchContext:
    questions: list[str] = []

    city = getattr(intent, "city", None)
    city = (city or "").strip() or None

    check_in = parse_iso_date(getattr(intent, "check_in", None))
    check_out = parse_iso_date(getattr(intent, "check_out", None))
    nights = getattr(intent, "nights", None)

    if not city:
        questions.append("Which city should I search in?")

    if check_in and check_out:
        if check_out <= check_in:
            questions.append("Check-out must be later than check-in.")

    elif check_in and nights is not None:
        if nights <= 0:
            questions.append("Number of nights must be greater than 0.")
        else:
            check_out = check_in + timedelta(days=nights)

    elif check_in and not check_out:
        # Never assume a one-night stay ("long stay starting April 1st").
        questions.append(
            "What is your check-out date? You can also tell me how many nights you'll stay."
        )

    elif check_out and not check_in:
        questions.append("Please specify the check-in date.")

    else:
        questions.append(
            "Please specify the travel dates. Example: 2026-04-20 for 6 nights."
        )

    # One question at a time: the budget is only asked about once city and
    # dates are settled, because the stay length decides whether it matters.
    if not questions:
        budget_question = _ambiguous_budget_question(intent, check_in, check_out)
        if budget_question:
            questions.append(budget_question)

    return ResolvedSearchContext(
        city=city,
        check_in=check_in,
        check_out=check_out,
        questions=questions,
    )


def _ambiguous_budget_question(intent, check_in: date | None, check_out: date | None) -> str | None:
    """
    A budget without a stated scope is ambiguous for a multi-night stay
    ($200 a night and $200 in total select very different listings). For a
    single night both readings are the same, so nothing is asked.
    """
    filters = getattr(intent, "filters", None)
    price = getattr(filters, "price", None) if filters is not None else None
    has_amount = price is not None and (price.max_amount is not None or price.min_amount is not None)

    period_budget = period_budget_text(intent)
    if period_budget and not has_amount:
        return (
            f"You mentioned {period_budget}. I can filter by a nightly or total-stay budget. "
            "What budget would you like me to use for your dates?"
        )

    if price is None or price.scope is not None:
        return None

    amount = price.max_amount if price.max_amount is not None else price.min_amount
    if amount is None or check_in is None or check_out is None:
        return None
    if (check_out - check_in).days <= 1:
        return None

    budget = _format_amount(amount, price.currency)
    return f"Do you mean {budget} per night or {budget} for the whole stay?"


def _format_amount(amount: float, currency: str | None) -> str:
    value = f"{amount:g}"
    if currency in {"$", "USD", "usd"}:
        return f"${value}"
    return f"{value} {currency}" if currency else value