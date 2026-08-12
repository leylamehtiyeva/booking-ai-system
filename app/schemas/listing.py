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
    available facilities, and booking options.

    Extra provider-specific fields are preserved because Full retrieval
    may contain additional room-level evidence.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    facilities: list[Any] = Field(default_factory=list)
    options: list[RoomOption] = Field(default_factory=list)


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
    facilities: list[Any] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)

    raw: dict[str, Any] | None = None