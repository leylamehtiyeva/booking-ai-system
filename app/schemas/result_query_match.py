from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field as PydanticField

from app.schemas.fields import Field as CanonicalField
from app.schemas.match import Evidence, Ternary


class ConstraintMatchResult(BaseModel):
    """
    Factual decision for one UserConstraint against one listing.

    Keyed by constraint identity (constraint_id/raw_text), not purely by
    CanonicalField, because an unresolved constraint (mapping_status ==
    unresolved, e.g. "romantic") has no Field at all and would otherwise
    have no way to appear in a structured result.
    """

    model_config = ConfigDict(extra="forbid")

    constraint_id: str | None = None
    raw_text: str
    field: CanonicalField | None = None
    value: Ternary
    evidence: list[Evidence] = PydanticField(default_factory=list)
    reason: str = ""


class ResultQueryMatch(BaseModel):
    """
    All constraint decisions for one shown candidate, keyed by result_id
    so the result_id <-> matcher output link survives regardless of how
    many constraints or candidates are involved.
    """

    model_config = ConfigDict(extra="forbid")

    result_id: str
    constraint_results: list[ConstraintMatchResult] = PydanticField(default_factory=list)
