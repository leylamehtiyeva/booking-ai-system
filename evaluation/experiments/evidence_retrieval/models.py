from __future__ import annotations

from pydantic import BaseModel


class EvidenceChunk(BaseModel):
    """
    One retrievable unit of property evidence.

    Isolated prototype model - not part of the production
    ListingRaw / ConstraintResolutionRequest schemas.
    """

    chunk_id: str
    property_id: str | None = None
    source_type: str
    path: str | None = None
    text: str

    # Always None today: guest review text is not available anywhere
    # in the current Apify pipeline (categoryReviews is omitted from
    # the actor request, and item["reviews"] is only a count, not
    # text). Kept for forward-compatibility if reviews are ever added.
    review_date: str | None = None


class EvidenceMetadata(BaseModel):
    property_id: str | None = None
    source_type: str
    path: str | None = None
    review_date: str | None = None


class EvidenceMatch(BaseModel):
    text: str
    similarity_score: float
    metadata: EvidenceMetadata
