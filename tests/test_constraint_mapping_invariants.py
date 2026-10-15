"""
UserConstraint mapping invariants (D3 / breakfast fix) must only remove
mappings that cannot be honest - never rewrite a correct canonical one.
Priority is never touched.
"""

from __future__ import annotations

import pytest

from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field

MUST, NICE, FORBIDDEN = ConstraintPriority.MUST, ConstraintPriority.NICE, ConstraintPriority.FORBIDDEN


def _known(normalized: str, priority, field: Field, *, raw: str | None = None, category=ConstraintCategory.AMENITY):
    return UserConstraint(
        raw_text=raw or normalized,
        normalized_text=normalized,
        priority=priority,
        category=category,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


@pytest.mark.parametrize(
    ("normalized", "priority", "field", "raw", "category"),
    [
        ("kitchen", MUST, Field.KITCHEN, "I need a place to cook", ConstraintCategory.AMENITY),
        ("wifi", NICE, Field.WIFI, "wifi would be great", ConstraintCategory.AMENITY),
        ("parking", MUST, Field.PARKING, "free parking is a must", ConstraintCategory.AMENITY),
        ("private bathroom", MUST, Field.PRIVATE_BATHROOM, "not sharing a bathroom", ConstraintCategory.AMENITY),
        ("balcony", NICE, Field.BALCONY, "a balcony would be great", ConstraintCategory.AMENITY),
        ("pet friendly", MUST, Field.PET_FRIENDLY, "we travel with our dog", ConstraintCategory.POLICY),
        ("smoking allowed", FORBIDDEN, Field.SMOKING_ALLOWED, "no smoking rooms, please", ConstraintCategory.POLICY),
        ("smoking allowed", NICE, Field.SMOKING_ALLOWED, "it would be great if smoking was allowed", ConstraintCategory.POLICY),
        ("coffee machine", NICE, Field.COFFEE_MACHINE, "a coffee machine would be nice", ConstraintCategory.AMENITY),
        # The meal plan word in the user's wording must not cost a real amenity its mapping.
        ("coffee machine", MUST, Field.COFFEE_MACHINE, "a coffee machine for my breakfast", ConstraintCategory.AMENITY),
        ("kettle", MUST, Field.KETTLE, "a kettle so I can make breakfast tea", ConstraintCategory.AMENITY),
    ],
)
def test_correct_canonical_mappings_are_kept(normalized, priority, field, raw, category):
    constraint = _known(normalized, priority, field, raw=raw, category=category)

    assert constraint.mapping_status == ConstraintMappingStatus.KNOWN
    assert constraint.mapped_fields == [field]
    assert constraint.evidence_strategy == EvidenceStrategy.STRUCTURED
    assert constraint.priority == priority


@pytest.mark.parametrize(
    ("normalized", "priority"),
    [
        ("breakfast included", MUST),
        ("breakfast", NICE),  # "breakfast would be nice"
        ("breakfast", FORBIDDEN),  # "no breakfast, please"
        ("half board", MUST),
        ("all-inclusive", NICE),
    ],
)
@pytest.mark.parametrize("wrong_field", [Field.COFFEE_MACHINE, Field.KITCHEN, Field.KETTLE])
def test_meal_plans_lose_any_amenity_mapping_but_keep_their_priority(normalized, priority, wrong_field):
    constraint = _known(normalized, priority, wrong_field)

    assert constraint.mapped_fields == []
    assert constraint.mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert constraint.evidence_strategy == EvidenceStrategy.TEXTUAL
    assert constraint.priority == priority


@pytest.mark.parametrize("priority", [MUST, NICE, FORBIDDEN])
def test_known_without_field_becomes_unresolved_for_every_priority(priority):
    constraint = UserConstraint(
        raw_text="quiet street",
        normalized_text="quiet street",
        priority=priority,
        category=ConstraintCategory.LOCATION,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )

    assert constraint.mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert constraint.evidence_strategy == EvidenceStrategy.TEXTUAL
    assert constraint.priority == priority


def test_unresolved_constraints_are_untouched():
    constraint = UserConstraint(
        raw_text="romantic atmosphere",
        normalized_text="romantic atmosphere",
        priority=MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    assert constraint.model_dump() == UserConstraint.model_validate(constraint.model_dump()).model_dump()


def test_invariants_hold_for_state_loaded_from_a_session():
    stored = {
        "raw_text": "breakfast",
        "normalized_text": "breakfast",
        "priority": "must",
        "category": "amenity",
        "mapping_status": "known",
        "mapped_fields": ["coffee_machine"],
        "evidence_strategy": "structured",
    }

    assert UserConstraint.model_validate(stored).mapped_fields == []
