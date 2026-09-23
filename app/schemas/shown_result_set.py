from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.listing import ListingRaw


class ShownResult(BaseModel):
    """
    One listing as it was actually shown to the user in a search
    response, paired with the full ListingRaw that normalize_search_response
    does not preserve.
    """

    result_id: str
    listing: ListingRaw


class ShownResultSet(BaseModel):
    """
    Snapshot of the latest executed search's shown listings.

    Order of `items` is the source of truth for display position
    ("the second one" == items[1]) - no separate rank field.
    """

    result_set_id: str
    items: list[ShownResult] = Field(default_factory=list)
