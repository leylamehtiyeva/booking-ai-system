"""
Single reference clock for "today".

Every date decision (year inference, past-date checks, the date shown to
the LLM for relative dates) reads today() instead of date.today(), so the
application uses the real calendar while tests and evaluations can pin a
fixed day.

Precedence:
1. use_reference_date(...) - scoped override (tests, evaluation runners);
2. APP_REFERENCE_DATE=YYYY-MM-DD - process-wide override (CLI/demo runs);
3. the real current date.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date

REFERENCE_DATE_ENV_VAR = "APP_REFERENCE_DATE"

_reference_date_override: ContextVar[date | None] = ContextVar(
    "reference_date_override",
    default=None,
)


def today() -> date:
    override = _reference_date_override.get()
    if override is not None:
        return override

    raw = os.getenv(REFERENCE_DATE_ENV_VAR)
    if raw:
        try:
            return date.fromisoformat(raw.strip().strip('"'))
        except ValueError as exc:
            raise ValueError(
                f"{REFERENCE_DATE_ENV_VAR} must be an ISO date (YYYY-MM-DD), got {raw!r}"
            ) from exc

    return date.today()


@contextmanager
def use_reference_date(value: date) -> Iterator[None]:
    token = _reference_date_override.set(value)
    try:
        yield
    finally:
        _reference_date_override.reset(token)
