from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RoomOption(BaseModel):
    """
    Represents a specific booking option available for a room.

    Examples include refundable rates, breakfast-included options,
    or other offer-level variations returned by the provider.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    price: float | None = None
    currency: str | None = None


class Room(BaseModel):
    """
    Represents a room-level accommodation entity returned by the provider.

    This model keeps room-specific attributes such as the room name,
    available facilities, and booking options. Room facilities are important
    for structured constraint matching because some user requirements may be
    satisfied at the room level rather than at the listing level.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    facilities: list[Any] = Field(default_factory=list)
    options: list[RoomOption] = Field(default_factory=list)


class ListingRaw(BaseModel):
    """
    Represents the raw listing contract received from the external provider.

    The model intentionally defines only the fields used by the current
    pipeline while allowing additional provider-specific fields to be preserved.
    This makes the ingestion layer more robust to schema changes and keeps
    the original payload available for debugging, enrichment, and future
    feature extraction.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    city: str | None = None

    name: str | None = None
    url: str | None = None

    price: float | None = None
    currency: str | None = None

    rating: float | None = None
    stars: int | None = None

    property_type: str | None = None

    description: str | None = None
    facilities: list[Any] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    raw: dict[str, Any] | None = None