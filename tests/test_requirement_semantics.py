"""
Requirement semantics behind match tiers (D3) and the breakfast mapping.

Product decisions under test:
- open-world (D1/D2): missing information is "not confirmed", never NO;
- D3: an unconfirmed MUST requirement - subjective or not - never allows
  an unconditional "strong" match; an explicit nice-to-have does not
  block "strong" when every hard requirement is confirmed;
- a meal plan ("breakfast included") has no canonical field, so it is
  never "confirmed" by an unrelated amenity such as a coffee machine.

Runs the real structured ranking, coverage normalization and selection
with the textual LLM fallback disabled (an unresolved requirement then
stays unconfirmed, which is exactly the case under test).
"""

from __future__ import annotations

from datetime import date

import pytest

from app.logic.listing_evaluation import evaluate_listings
from app.logic.result_selection import select_ranked_items
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


def _constraint(
    text: str,
    *,
    priority: ConstraintPriority = ConstraintPriority.MUST,
    category: ConstraintCategory = ConstraintCategory.AMENITY,
    fields: list[Field] | None = None,
    known: bool | None = None,
) -> UserConstraint:
    fields = fields or []
    is_known = bool(fields) if known is None else known
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=category,
        mapping_status=ConstraintMappingStatus.KNOWN if is_known else ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=fields,
        evidence_strategy=EvidenceStrategy.STRUCTURED if is_known else EvidenceStrategy.TEXTUAL,
    )


LISTING = ListingRaw(
    id="h1",
    name="Coffee Corner Hotel",
    city="Baku",
    property_type="hotel",
    description="A small hotel near the old town.",
    facilities=[{"name": "Free WiFi"}, {"name": "Coffee machine"}, {"name": "Non-smoking rooms"}],
)

WIFI = _constraint("wifi", fields=[Field.WIFI])


async def _tier(*constraints: UserConstraint) -> str:
    request = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 10),
        check_out=date(2026, 4, 15),
        constraints=list(constraints),
    )
    evaluation = await evaluate_listings(request, [LISTING], fallback_policy=FallbackPolicy(enabled=False))
    selected = select_ranked_items(evaluation.ranked_items, top_n=5)
    assert len(selected) == 1, "the listing must stay eligible (open-world)"
    return selected[0]["match_tier"]


# --- baseline: confirmed hard requirements --------------------------------------


async def test_confirmed_must_is_strong():
    assert await _tier(WIFI) == "strong"


# --- D3: unconfirmed MUST never allows strong -----------------------------------


async def test_unconfirmed_subjective_must_is_not_strong():
    assert await _tier(WIFI, _constraint("romantic atmosphere", category=ConstraintCategory.OTHER)) == "partial"


async def test_unconfirmed_structured_must_is_not_strong():
    assert await _tier(WIFI, _constraint("parking", fields=[Field.PARKING])) == "partial"


@pytest.mark.parametrize("priority", [ConstraintPriority.MUST, ConstraintPriority.FORBIDDEN])
async def test_blocking_requirement_known_but_without_any_field_is_not_ignored(priority):
    """LLM output 'known' with no mapped field: nothing checks it - it must stay unconfirmed."""
    unmapped = _constraint("guests may bring a dog", priority=priority, category=ConstraintCategory.POLICY, known=True)

    assert unmapped.mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert await _tier(WIFI, unmapped) == "partial"


@pytest.mark.parametrize(
    "category",
    [ConstraintCategory.PROPERTY_TYPE, ConstraintCategory.OCCUPANCY, ConstraintCategory.NUMERIC],
)
def test_slot_backed_constraint_without_field_is_left_to_its_slot(category):
    """'apartment' / '2 adults' / 'at least 50 sqm' are enforced by property_types / occupancy / filters."""
    constraint = _constraint("apartment", category=category, known=True)

    assert constraint.mapping_status == ConstraintMappingStatus.KNOWN


# --- D3: explicit nice-to-have does not block strong -----------------------------


async def test_unconfirmed_nice_to_have_does_not_block_strong():
    nice = _constraint("feels local", priority=ConstraintPriority.NICE, category=ConstraintCategory.OTHER)

    assert await _tier(WIFI, nice) == "strong"


async def test_failed_nice_to_have_does_not_block_strong():
    # "it would be great if smoking was allowed" on a non-smoking listing.
    nice = _constraint("smoking allowed", priority=ConstraintPriority.NICE, category=ConstraintCategory.POLICY, fields=[Field.SMOKING_ALLOWED])

    assert await _tier(WIFI, nice) == "strong"


# --- breakfast is never confirmed by a coffee machine --------------------------------


@pytest.mark.parametrize(
    "text",
    ["breakfast", "breakfast included", "free breakfast", "half board", "all inclusive"],
)
def test_meal_plan_is_never_mapped_to_an_amenity_field(text):
    constraint = _constraint(text, fields=[Field.COFFEE_MACHINE])

    assert constraint.mapped_fields == []
    assert constraint.mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert constraint.evidence_strategy == EvidenceStrategy.TEXTUAL


async def test_breakfast_is_not_confirmed_by_a_coffee_machine():
    breakfast = _constraint("breakfast", fields=[Field.COFFEE_MACHINE])

    assert await _tier(WIFI, breakfast) == "partial"


def test_real_amenity_mappings_are_kept():
    assert _constraint("coffee machine", fields=[Field.COFFEE_MACHINE]).mapped_fields == [Field.COFFEE_MACHINE]
    assert _constraint("kitchen", fields=[Field.KITCHEN]).mapped_fields == [Field.KITCHEN]


def test_bed_and_breakfast_property_type_is_not_touched():
    constraint = _constraint("bed and breakfast", category=ConstraintCategory.PROPERTY_TYPE, known=True)

    assert constraint.category == ConstraintCategory.PROPERTY_TYPE
    assert constraint.mapping_status == ConstraintMappingStatus.KNOWN
