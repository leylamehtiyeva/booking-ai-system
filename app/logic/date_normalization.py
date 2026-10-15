from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from app.config import clock


ISO_DATE_PATTERN = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")

# When the user gives no year, check-out is placed on the first matching
# day after check-in. If that rolled check-out into the next year AND made
# the stay longer than this, the user's check-out was actually before
# check-in ("April 15 to April 10"). Without a rollover a long stay is just
# long ("April 2 until the end of the year").
MAX_INFERRED_STAY_NIGHTS = 90

_LEAP_YEAR = 2000
_MAX_YEAR_LOOKAHEAD = 8


_CURRENCY_SYMBOLS = "$€£¥₽₼₺₾"
_CURRENCY_WORDS = (
    "usd|eur|gbp|azn|rub|try|gel|aed|jpy|thb|"
    "dollars?|euros?|manats?|rubles?|lari|lira"
)


def count_stated_year(year: int, user_text: str) -> int:
    """
    How many times the user wrote this exact year - money amounts like
    "$2000" / "2000 USD" and longer numbers do not count.
    """
    pattern = (
        rf"(?<![\d{_CURRENCY_SYMBOLS}]){year}(?!\d)"
        rf"(?!\s*(?:[{_CURRENCY_SYMBOLS}]|(?:{_CURRENCY_WORDS})\b))"
    )
    return len(re.findall(pattern, user_text, flags=re.IGNORECASE))


class StayDatesIssue(str, Enum):
    INVALID_DATE = "invalid_date"
    CHECK_OUT_NOT_AFTER_CHECK_IN = "check_out_not_after_check_in"
    IN_THE_PAST = "in_the_past"
    NON_POSITIVE_NIGHTS = "non_positive_nights"


_ISSUE_QUESTIONS = {
    StayDatesIssue.INVALID_DATE: (
        "I couldn't understand those dates. Could you give the check-in and "
        "check-out dates again?"
    ),
    StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN: (
        "The check-out date must be after the check-in date. Could you "
        "confirm your travel dates?"
    ),
    StayDatesIssue.IN_THE_PAST: (
        "Those dates are in the past. Which dates would you like to travel?"
    ),
    StayDatesIssue.NON_POSITIVE_NIGHTS: "Number of nights must be greater than 0.",
}


class InvalidStayDatesError(ValueError):
    """
    The user's dates cannot form a valid future stay. Carries a
    deterministic clarification question for the user.

    partial_state is set by START_SEARCH (the new request with its dates
    cleared, so city/constraints are not lost); for UPDATE_SEARCH it stays
    None and the caller keeps the previous state.
    """

    def __init__(self, issue: StayDatesIssue, *, partial_state=None) -> None:
        super().__init__(f"Invalid stay dates: {issue.value}")
        self.issue = issue
        self.question = _ISSUE_QUESTIONS[issue]
        self.partial_state = partial_state


@dataclass(frozen=True)
class StayDates:
    check_in: date | None
    check_out: date | None


def _parse_date_parts(raw: str | date | None) -> tuple[int, int, int] | None:
    """
    (year, month, day) from an ISO string, without requiring the year/day
    combination to exist yet: when the user gave no year, the LLM's year
    is replaced anyway, so "2026-02-29" must still read as Feb 29.
    """
    if raw is None or raw == "":
        return None

    if isinstance(raw, date):
        return raw.year, raw.month, raw.day

    match = ISO_DATE_PATTERN.match(str(raw))
    if match is None:
        raise InvalidStayDatesError(StayDatesIssue.INVALID_DATE)

    year, month, day = (int(part) for part in match.groups())

    try:
        date(_LEAP_YEAR, month, day)
    except ValueError:
        raise InvalidStayDatesError(StayDatesIssue.INVALID_DATE) from None

    return year, month, day


def _exact_date(parts: tuple[int, int, int]) -> date:
    try:
        return date(*parts)
    except ValueError:
        raise InvalidStayDatesError(StayDatesIssue.INVALID_DATE) from None


def _next_occurrence(month: int, day: int, *, on_or_after: date) -> date:
    for year in range(on_or_after.year, on_or_after.year + _MAX_YEAR_LOOKAHEAD + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue  # Feb 29 outside a leap year
        if candidate >= on_or_after:
            return candidate

    raise InvalidStayDatesError(StayDatesIssue.INVALID_DATE)


def resolve_stay_dates(
    *,
    check_in: str | date | None,
    check_out: str | date | None,
    nights: int | None,
    user_text: str,
    today: date | None = None,
    anchor_check_in: date | None = None,
) -> StayDates:
    """
    Turn LLM-extracted dates into a valid future stay, or raise
    InvalidStayDatesError - never returns check_out <= check_in.

    Decided per date, by whether the user actually wrote that date's year
    (see count_stated_year - "$2000" is not a year):
    - Year not stated: the LLM's year is ignored; check-in is the next
      occurrence on/after today, and check-out the next occurrence
      on/after check-in (or anchor_check_in, the existing check-in for a
      partial UPDATE_SEARCH) - which is what handles cross-year ranges
      like "Dec 28 - Jan 3".
    - Year stated: the date is taken as given, and a stay in the past is
      rejected rather than silently moved to the future.
    """
    today = today or clock.today()

    check_in_parts = _parse_date_parts(check_in)
    check_out_parts = _parse_date_parts(check_out)

    if nights is not None and nights <= 0:
        raise InvalidStayDatesError(StayDatesIssue.NON_POSITIVE_NIGHTS)

    resolved_check_in = None
    if check_in_parts:
        if count_stated_year(check_in_parts[0], user_text) > 0:
            resolved_check_in = _exact_date(check_in_parts)
        else:
            resolved_check_in = _next_occurrence(
                check_in_parts[1], check_in_parts[2], on_or_after=today
            )

    # "from Dec 28 2026 to Jan 3": the LLM tends to copy the one stated year
    # onto check-out. A year written once and shared by both dates belongs
    # to check-in; check-out is then inferred from it.
    check_out_year_stated = False
    if check_out_parts:
        check_out_year_count = count_stated_year(check_out_parts[0], user_text)
        shares_single_year_with_check_in = (
            check_in_parts is not None
            and check_in_parts[0] == check_out_parts[0]
            and check_out_year_count < 2
        )
        check_out_year_stated = (
            check_out_year_count > 0 and not shares_single_year_with_check_in
        )

    resolved_check_out = None
    if check_out_parts:
        if check_out_year_stated:
            resolved_check_out = _exact_date(check_out_parts)
        else:
            stay_start = resolved_check_in or anchor_check_in or today
            resolved_check_out = _next_occurrence(
                check_out_parts[1], check_out_parts[2], on_or_after=stay_start
            )
            rolled_into_next_year = resolved_check_out.year > stay_start.year
            if (
                rolled_into_next_year
                and (resolved_check_out - stay_start).days > MAX_INFERRED_STAY_NIGHTS
            ):
                raise InvalidStayDatesError(StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN)

    # No check-out and no nights: left open - the caller asks for it
    # rather than assuming a one-night stay.
    if resolved_check_in is not None and resolved_check_out is None and nights is not None:
        resolved_check_out = resolved_check_in + timedelta(days=nights)

    stay_start = resolved_check_in or anchor_check_in
    if (
        stay_start is not None
        and resolved_check_out is not None
        and resolved_check_out <= stay_start
    ):
        raise InvalidStayDatesError(StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN)

    if resolved_check_in is not None and resolved_check_in < today:
        raise InvalidStayDatesError(StayDatesIssue.IN_THE_PAST)

    if resolved_check_out is not None and resolved_check_out <= today:
        raise InvalidStayDatesError(StayDatesIssue.IN_THE_PAST)

    return StayDates(check_in=resolved_check_in, check_out=resolved_check_out)


def normalize_intent_dates(intent, user_text: str, *, today: date | None = None):
    """
    START_SEARCH: normalize the first-turn intent's dates. A check-in without
    check-out or nights stays open (asked for by required-context
    resolution). Raises InvalidStayDatesError.
    """
    stay = resolve_stay_dates(
        check_in=getattr(intent, "check_in", None),
        check_out=getattr(intent, "check_out", None),
        nights=getattr(intent, "nights", None),
        user_text=user_text,
        today=today,
    )

    data = intent.model_dump()
    data["check_in"] = stay.check_in.isoformat() if stay.check_in else None
    data["check_out"] = stay.check_out.isoformat() if stay.check_out else None
    return intent.__class__(**data)


def normalize_patch_dates(
    *,
    set_check_in: str | None,
    set_check_out: str | None,
    set_nights: int | None,
    user_text: str,
    today: date | None = None,
    current_check_in: date | None = None,
) -> tuple[str | None, str | None]:
    """
    UPDATE_SEARCH: same rules as START_SEARCH; a check-out-only patch is
    resolved relative to the existing check-in. Defaults for a check-in
    without check-out stay in apply_intent_patch. Raises
    InvalidStayDatesError.
    """
    stay = resolve_stay_dates(
        check_in=set_check_in,
        check_out=set_check_out,
        nights=set_nights,
        user_text=user_text,
        today=today,
        anchor_check_in=None if set_check_in else current_check_in,
    )

    return (
        stay.check_in.isoformat() if stay.check_in else None,
        stay.check_out.isoformat() if stay.check_out else None,
    )
