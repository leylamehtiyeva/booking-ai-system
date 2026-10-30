from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.public_offer import PublicOffer


class Policy(BaseModel):
    title: str | None = None
    content: str | None = None


class Highlight(BaseModel):
    header: str | None = None
    contents: list[str] = Field(
        default_factory=list
    )

class Facility(BaseModel):
    """
    Provider-independent facility evidence.

    `name` is the facility itself.
    `category` preserves the provider grouping when available.
    `overview` preserves important contextual information such as
    "No parking available" or paid/off-site parking details.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    category: str | None = None
    overview: str | None = None

    requires_additional_charge: bool | None = None
    is_off_site: bool | None = None


class BedGroup(BaseModel):
    """
    Beds belonging to a bedroom / sleeping area.
    """

    model_config = ConfigDict(extra="allow")

    room: str | None = None
    beds: list[str] = Field(default_factory=list)


class RoomOption(BaseModel):
    """
    Booking option available for a room.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None

    price: float | None = None
    currency: str | None = None

    persons: int | None = None

    cancellation_type: str | None = None
    free_cancellation: bool | None = None

    choices: list[str] = Field(default_factory=list)


class Room(BaseModel):
    """
    Provider-independent room representation.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None

    available: bool | None = None
    persons: int | None = None

    bed_types: list[BedGroup] = Field(
        default_factory=list
    )

    # `str` is temporarily supported for old fixtures.
    # We can remove it after downstream migration.
    facilities: list[Facility | str] = Field(
        default_factory=list
    )

    options: list[RoomOption] = Field(
        default_factory=list
    )


class ReviewCategoryScore(BaseModel):
    name: str
    rating: float | None = None
    description: str | None = None


class ReviewSummary(BaseModel):
    """
    Provider-generated summary of guest reviews (guest opinion, not a
    listing fact). Kept with its origin for later use; nothing in the
    matching or evidence pipeline reads it yet.
    """

    source: str
    updated_at: str | None = None
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    categories: list[ReviewCategoryScore] = Field(default_factory=list)


class ListingProvenance(BaseModel):
    """Where and when a listing's data was obtained."""

    provider: str
    environment: str | None = None
    fetched_at: datetime


class ListingRaw(BaseModel):
    """
    Internal listing representation used by the retrieval and
    matching pipeline.

    Provider-specific payloads should be normalized into this model
    before they are passed further into the search pipeline.

    The raw provider payload is preserved in `raw` for debugging and
    future enrichment needs.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    city: str | None = None
    address: str | None = None

    name: str | None = None
    url: str | None = None

    price: float | None = None
    currency: str | None = None

    rating: float | None = None
    stars: int | None = None

    property_type: str | None = None

    room_type: str | None = None
    max_occupancy: int | None = None
    
    description: str | None = None
    fine_print: str | None = None

    policies: list[Policy] = Field(
        default_factory=list
    )

    highlights: list[Highlight] = Field(
        default_factory=list
    )

    facilities: list[Facility | str] = Field(
        default_factory=list
    )


    # `str` is temporarily supported for old fixtures.
    facilities: list[Facility | str] = Field(
        default_factory=list
    )

    rooms: list[Room] = Field(
        default_factory=list
    )

    latitude: float | None = None
    longitude: float | None = None

    review_count: int | None = None
    review_summary: ReviewSummary | None = None

    # Public part of the provider's offers for the requested stay; `price`
    # is the lowest public_total among them (see app.schemas.public_offer).
    public_offers: list[PublicOffer] = Field(
        default_factory=list
    )

    provenance: ListingProvenance | None = None

    raw: dict[str, Any] | None = None