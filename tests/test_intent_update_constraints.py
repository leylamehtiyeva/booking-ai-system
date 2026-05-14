import pytest
from datetime import date

from app.logic.intent_update import (
    _build_update_prompt,
    update_search_state_async,
)
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.query import SearchRequest


def test_build_update_prompt_includes_canonical_constraints():
    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
        adults=2,
        children=0,
        rooms=1,
        constraints=[
            UserConstraint(
                raw_text="kitchen",
                normalized_text="kitchen",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.AMENITY,
                mapping_status=ConstraintMappingStatus.KNOWN,
                mapped_fields=[Field.KITCHEN],
                evidence_strategy=EvidenceStrategy.STRUCTURED,
            ),
            UserConstraint(
                raw_text="quiet neighborhood",
                normalized_text="quiet neighborhood",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.LOCATION,
                mapping_status=ConstraintMappingStatus.UNRESOLVED,
                mapped_fields=[],
                evidence_strategy=EvidenceStrategy.TEXTUAL,
            ),
        ],
    )

    prompt = _build_update_prompt(previous_state, "also add a balcony")

    assert '"constraints"' in prompt
    assert '"normalized_text": "kitchen"' in prompt
    assert '"normalized_text": "quiet neighborhood"' in prompt


@pytest.mark.asyncio
async def test_update_search_state_async_preserves_constraint_centric_state(monkeypatch):
    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
        adults=2,
        children=0,
        rooms=1,
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
    )

    async def fake_route_intent_update_patch_async(previous_state, user_message, trace=None):
        return SearchIntentPatch(
            add_constraints=[
                UserConstraint(
                    raw_text="quiet neighborhood",
                    normalized_text="quiet neighborhood",
                    priority=ConstraintPriority.MUST,
                    category=ConstraintCategory.LOCATION,
                    mapping_status=ConstraintMappingStatus.UNRESOLVED,
                    mapped_fields=[],
                    evidence_strategy=EvidenceStrategy.TEXTUAL,
                )
            ]
        )

    monkeypatch.setattr(
        "app.logic.intent_update.route_intent_update_patch_async",
        fake_route_intent_update_patch_async,
    )

    new_state = await update_search_state_async(
        previous_state,
        "also I want a quiet neighborhood",
    )

    assert any(c.normalized_text == "kitchen" for c in new_state.constraints)
    assert any(c.normalized_text == "quiet neighborhood" for c in new_state.constraints)