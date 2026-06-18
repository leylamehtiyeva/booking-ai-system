from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field as PydanticField

from app.schemas.constraints import UserConstraint
from app.schemas.filters import SearchFilters
from app.schemas.property_semantics import OccupancyType, PropertyType


class SearchIntentPatch(BaseModel):
    # --- core slots ---
    set_city: Optional[str] = None
    clear_city: bool = False

    set_check_in: Optional[str] = None
    set_check_out: Optional[str] = None
    set_nights: Optional[int] = None
    clear_dates: bool = False

    set_adults: Optional[int] = None
    set_children: Optional[int] = None
    set_rooms: Optional[int] = None


    # --- constraints (canonical source of truth patch layer) ---
    add_constraints: List[UserConstraint] = PydanticField(default_factory=list)
    remove_constraint_texts: List[str] = PydanticField(default_factory=list)

    # --- filters ---
    set_filters: Optional[SearchFilters] = None
    clear_filters: bool = False

    # --- property or occupancy ---
    add_property_types: List[PropertyType] = PydanticField(default_factory=list)
    remove_property_types: List[PropertyType] = PydanticField(default_factory=list)

    add_occupancy_types: List[OccupancyType] = PydanticField(default_factory=list)
    remove_occupancy_types: List[OccupancyType] = PydanticField(default_factory=list)

