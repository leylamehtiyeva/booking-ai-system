import asyncio

from app.agents.intent_router_agent import IntentRoute
from app.logic.intent_router import build_search_request_adk_async
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.filters import SearchFilters
from app.schemas.property_semantics import PropertyType


def test_build_search_request_uses_constraints_as_source_of_truth(monkeypatch):
    parsed_intent = IntentRoute(
        city="Baku",
        check_in="2026-04-10",
        check_out="2026-04-15",
        adults=4,
        children=0,
        rooms=1,
        filters=SearchFilters(),
        property_types=[PropertyType.APARTMENT],
        occupancy_types=[],
        constraints=[
            UserConstraint(
                raw_text="place for cooking",
                normalized_text="place for cooking",
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
            UserConstraint(
                raw_text="balcony",
                normalized_text="balcony",
                priority=ConstraintPriority.NICE,
                category=ConstraintCategory.AMENITY,
                mapping_status=ConstraintMappingStatus.KNOWN,
                mapped_fields=[Field.BALCONY],
                evidence_strategy=EvidenceStrategy.STRUCTURED,
            ),
        ],
    )

    async def fake_route_intent_adk_async(user_text: str, trace=None, step=None,):
        return parsed_intent

    monkeypatch.setattr(
        "app.logic.intent_router.route_intent_adk_async",
        fake_route_intent_adk_async,
    )

    req = asyncio.run(
        build_search_request_adk_async(
            "I want an apartment in Baku from 10 to 15 April for 4 people with a place for cooking"
        )
    )

    assert len(req.constraints) == len(parsed_intent.constraints)
    assert req.constraints[0].normalized_text == "place for cooking"
    assert req.constraints[1].normalized_text == "quiet neighborhood"
    assert req.constraints[1].mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert req.constraints[2].normalized_text == "balcony"
    assert req.constraints[2].priority == ConstraintPriority.NICE
    assert req.property_types == [PropertyType.APARTMENT]