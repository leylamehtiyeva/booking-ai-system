from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

class PriceConstraint(BaseModel):
    """
    User-side budget constraint.

    scope:
    - per_night   
    - total_stay 
    """
    min_amount: float | None = None
    max_amount: float | None = None
    currency: str | None = None
    scope: Literal["per_night", "total_stay"] | None = None


class SearchFilters(BaseModel):
    """
    Typed structured constraints extracted from the user query.

    These are structured numeric constraints.
    They must live in SearchFilters, not in semantic constraints.
    """
    bedrooms_min: int | None = None
    bedrooms_max: int | None = None
    area_sqm_min: float | None = None
    area_sqm_max: float | None = None
    bathrooms_min: float | None = None
    bathrooms_max: float | None = None

    price: PriceConstraint | None = None
    
    @model_validator(mode="after")
    def normalize_empty_price(self):
        # Drop only a price that carries nothing. A scope- or currency-only
        # price is a patch ("per night" answering the budget question) that
        # apply_intent_patch merges into the existing amount; without an
        # amount it never filters anything on its own.
        if (
            self.price is not None
            and self.price.min_amount is None
            and self.price.max_amount is None
            and self.price.currency is None
            and self.price.scope is None
        ):
            self.price = None

        return self