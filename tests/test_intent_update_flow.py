from datetime import date

import pytest

from app.logic.intent_update import update_search_state_async
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.query import SearchFilters, SearchRequest


@pytest.mark.asyncio
async def test_update_search_state_adds_constraint_and_updates_filter(monkeypatch):
    async def _fake_patch(previous_state, user_message, trace=None):
        return SearchIntentPatch(
            add_constraints=[
                UserConstraint(
                    raw_text="kettle",
                    normalized_text="kettle",
                    priority=ConstraintPriority.MUST,
                    category=ConstraintCategory.AMENITY,
                    mapping_status=ConstraintMappingStatus.KNOWN,
                    mapped_fields=[Field.KETTLE],
                    evidence_strategy=EvidenceStrategy.STRUCTURED,
                )
            ],
            set_filters=SearchFilters(bedrooms_min=3),
        )

    monkeypatch.setattr(
        "app.logic.intent_update.route_intent_update_patch_async",
        _fake_patch,
    )

    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
        constraints=[
            UserConstraint(
                raw_text="kitchen",
                normalized_text="kitchen",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.AMENITY,
                mapping_status=ConstraintMappingStatus.KNOWN,
                mapped_fields=[Field.KITCHEN],
                evidence_strategy=EvidenceStrategy.STRUCTURED,
            )
        ],
        filters=SearchFilters(area_sqm_min=80, bedrooms_min=2),
    )

    updated = await update_search_state_async(
        previous_state,
        "Also I want a kettle, and now at least 3 bedrooms.",
    )

    assert updated.city == "Baku"
    assert updated.check_in == date(2026, 4, 20)
    assert updated.check_out == date(2026, 4, 26)

    assert any(c.normalized_text == "kitchen" for c in updated.constraints)
    assert any(c.normalized_text == "kettle" for c in updated.constraints)

    kettle = next(c for c in updated.constraints if c.normalized_text == "kettle")
    assert kettle.priority == ConstraintPriority.MUST
    assert kettle.mapping_status == ConstraintMappingStatus.KNOWN
    assert kettle.mapped_fields == [Field.KETTLE]
    assert kettle.evidence_strategy == EvidenceStrategy.STRUCTURED

    assert updated.filters is not None
    assert updated.filters.area_sqm_min == 80
    assert updated.filters.bedrooms_min == 3