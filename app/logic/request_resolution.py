from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta



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
        check_out = check_in + timedelta(days=1)

    elif check_out and not check_in:
        questions.append("Please specify the check-in date.")

    else:
        questions.append(
            "Please specify the travel dates. Example: 2026-04-20 for 6 nights."
        )

    return ResolvedSearchContext(
        city=city,
        check_in=check_in,
        check_out=check_out,
        questions=questions,
    )