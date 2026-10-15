from datetime import date

import pytest

from app.agents.intent_router_agent import IntentRoute
from app.logic.date_normalization import (
    InvalidStayDatesError,
    StayDatesIssue,
    normalize_intent_dates,
)
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)


def test_normalize_intent_dates_sets_current_year_when_user_did_not_specify_year():
    intent = IntentRoute(
        city="Baku",
        check_in="2024-04-08",
        check_out="2024-04-15",
    )

    normalized = normalize_intent_dates(
        intent,
        "apartment in Baku 8 april to 15 april",
        today=date(2026, 4, 5),
    )

    assert normalized.check_in == "2026-04-08"
    assert normalized.check_out == "2026-04-15"


def test_normalize_intent_dates_keeps_explicit_future_year():
    intent = IntentRoute(
        city="Baku",
        check_in="2027-04-08",
        check_out="2027-04-15",
    )

    normalized = normalize_intent_dates(
        intent,
        "apartment in Baku from 2027-04-08 to 2027-04-15",
        today=date(2026, 4, 5),
    )

    assert normalized.check_in == "2027-04-08"
    assert normalized.check_out == "2027-04-15"


def test_normalize_intent_dates_rejects_explicit_past_year():
    """Previously kept as-is, i.e. a search for a stay in the past."""
    intent = IntentRoute(
        city="Baku",
        check_in="2024-04-08",
        check_out="2024-04-15",
    )

    with pytest.raises(InvalidStayDatesError) as exc_info:
        normalize_intent_dates(
            intent,
            "apartment in Baku from 2024-04-08 to 2024-04-15",
            today=date(2026, 4, 5),
        )

    assert exc_info.value.issue == StayDatesIssue.IN_THE_PAST


def test_normalize_intent_dates_computes_check_out_from_nights():
    intent = IntentRoute(
        city="Baku",
        check_in="2024-04-20",
        check_out=None,
        nights=6,
    )

    normalized = normalize_intent_dates(
        intent,
        "Baku 20 april for 6 nights",
        today=date(2026, 4, 5),
    )

    assert normalized.check_in == "2026-04-20"
    assert normalized.check_out == "2026-04-26"


def test_unresolved_constraints_do_not_clear_valid_dates():
    intent = IntentRoute(
        city="Baku",
        check_in="2024-04-20",
        check_out="2024-04-26",
        constraints=[
            UserConstraint(
                raw_text="satellite TV",
                normalized_text="satellite TV",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.AMENITY,
                mapping_status=ConstraintMappingStatus.UNRESOLVED,
                mapped_fields=[],
                evidence_strategy=EvidenceStrategy.TEXTUAL,
            )
        ],
    )

    normalized = normalize_intent_dates(
        intent,
        "I want an apartment in Baku from April 20 to April 26 with satellite TV",
        today=date(2026, 4, 5),
    )

    assert normalized.check_in == "2026-04-20"
    assert normalized.check_out == "2026-04-26"