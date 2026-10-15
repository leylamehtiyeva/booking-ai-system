from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from app.config import clock


YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
CHINESE_YEAR_PATTERN = re.compile(r"(19|20)\d{2}\s*年")
ISO_DATE_PATTERN = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")

# When the user gives no year, check-out is placed on the first matching
# day after check-in. A result longer than this means the user's check-out
# was actually before check-in ("April 15 to April 10"), not a stay that
# spans a year - booking platforms cap stays well below this anyway.
MAX_INFERRED_STAY_NIGHTS = 90

_LEAP_YEAR = 2000
_MAX_YEAR_LOOKAHEAD = 8


def user_explicitly_mentioned_year(user_text: str) -> bool:
    return bool(
        YEAR_PATTERN.search(user_text) or CHINESE_YEAR_PATTERN.search(user_text)
    )


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
    default_one_night: bool = False,
) -> StayDates:
    """
    Turn LLM-extracted dates into a valid future stay, or raise
    InvalidStayDatesError - never returns check_out <= check_in.

    - No year in the user's text: the LLM's year is ignored; check-in is
      the next occurrence on/after today, and check-out the next
      occurrence on/after check-in (or anchor_check_in, the existing
      check-in for a partial UPDATE_SEARCH) - which is what handles
      cross-year ranges like "Dec 28 - Jan 3".
    - Year in the user's text: the dates are taken as given, and a stay
      in the past is rejected rather than silently moved to the future.
    """
    today = today or clock.today()

    check_in_parts = _parse_date_parts(check_in)
    check_out_parts = _parse_date_parts(check_out)

    if nights is not None and nights <= 0:
        raise InvalidStayDatesError(StayDatesIssue.NON_POSITIVE_NIGHTS)

    if user_explicitly_mentioned_year(user_text):
        resolved_check_in = _exact_date(check_in_parts) if check_in_parts else None
        resolved_check_out = _exact_date(check_out_parts) if check_out_parts else None
    else:
        resolved_check_in = (
            _next_occurrence(check_in_parts[1], check_in_parts[2], on_or_after=today)
            if check_in_parts
            else None
        )
        resolved_check_out = None
        if check_out_parts:
            stay_start = resolved_check_in or anchor_check_in or today
            resolved_check_out = _next_occurrence(
                check_out_parts[1], check_out_parts[2], on_or_after=stay_start
            )
            if (resolved_check_out - stay_start).days > MAX_INFERRED_STAY_NIGHTS:
                raise InvalidStayDatesError(StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN)

    if resolved_check_in is not None and resolved_check_out is None:
        if nights is not None:
            resolved_check_out = resolved_check_in + timedelta(days=nights)
        elif default_one_night:
            resolved_check_out = resolved_check_in + timedelta(days=1)

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
    START_SEARCH: normalize the first-turn intent's dates (a single date
    defaults to one night). Raises InvalidStayDatesError.
    """
    stay = resolve_stay_dates(
        check_in=getattr(intent, "check_in", None),
        check_out=getattr(intent, "check_out", None),
        nights=getattr(intent, "nights", None),
        user_text=user_text,
        today=today,
        default_one_night=True,
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
