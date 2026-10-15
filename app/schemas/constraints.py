from __future__ import annotations
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field as PydanticField, model_validator
from app.schemas.fields import Field


class ConstraintPriority(str, Enum):
    MUST = "must"
    NICE = "nice"
    FORBIDDEN = "forbidden"


class ConstraintMappingStatus(str, Enum):
    KNOWN = "known"
    UNRESOLVED = "unresolved"


class ConstraintCategory(str, Enum):
    AMENITY = "amenity"
    POLICY = "policy"
    LOCATION = "location"
    LAYOUT = "layout"
    NUMERIC = "numeric"
    PROPERTY_TYPE = "property_type"
    OCCUPANCY = "occupancy"
    OTHER = "other"


class EvidenceStrategy(str, Enum):
    STRUCTURED = "structured"
    TEXTUAL = "textual"
    NONE = "none"


class UserConstraint(BaseModel):
    id: str = PydanticField(default_factory=lambda: str(uuid4()))
    raw_text: str
    normalized_text: str
    priority: ConstraintPriority
    category: ConstraintCategory = ConstraintCategory.OTHER
    mapping_status: ConstraintMappingStatus = ConstraintMappingStatus.UNRESOLVED
    mapped_fields: list[Field] = PydanticField(default_factory=list)
    evidence_strategy: EvidenceStrategy = EvidenceStrategy.NONE

    @model_validator(mode="after")
    def _keep_mapping_honest(self) -> "UserConstraint":
        """
        Invariants on LLM-produced mappings, enforced wherever a constraint
        is created or loaded (search intent, update patch, factual question,
        stored session state):

        - A meal plan ("breakfast included", "half board", ...) has no
          canonical field. Any amenity it was mapped to (typically
          coffee_machine) would let that amenity "confirm" the meal plan,
          so it is checked against listing text instead.
        - "known" with no mapped field means nothing would ever check the
          requirement, so a MUST/FORBIDDEN one would be silently ignored.
          It is treated as unresolved (textual) - unless its category is
          enforced by a dedicated slot (property_types, occupancy_types,
          numeric filters), which already checks it.
        """
        if self.category in _SLOT_BACKED_CATEGORIES:
            return self

        if self.mapped_fields and _mentions_meal_plan(self):
            self._mark_unresolved()
        elif self.mapping_status == ConstraintMappingStatus.KNOWN and not self.mapped_fields:
            self._mark_unresolved()

        return self

    def _mark_unresolved(self) -> None:
        self.mapping_status = ConstraintMappingStatus.UNRESOLVED
        self.mapped_fields = []
        self.evidence_strategy = EvidenceStrategy.TEXTUAL


_SLOT_BACKED_CATEGORIES = frozenset(
    {
        ConstraintCategory.PROPERTY_TYPE,
        ConstraintCategory.OCCUPANCY,
        ConstraintCategory.NUMERIC,
    }
)

_MEAL_PLAN_TERMS = (
    "breakfast",
    "half board",
    "half-board",
    "full board",
    "full-board",
    "all inclusive",
    "all-inclusive",
    "meals included",
    "meal plan",
)


def _mentions_meal_plan(constraint: UserConstraint) -> bool:
    # normalized_text is the requirement itself (English, concise);
    # raw_text is the user's wording, where "a coffee machine for my
    # breakfast" is still a coffee machine.
    text = constraint.normalized_text.casefold()
    return any(term in text for term in _MEAL_PLAN_TERMS)