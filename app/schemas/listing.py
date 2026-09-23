from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    raw: dict[str, Any] | None = None