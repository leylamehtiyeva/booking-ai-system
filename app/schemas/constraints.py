from __future__ import annotations
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field as PydanticField
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