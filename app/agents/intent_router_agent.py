from __future__ import annotations

import json
import os
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from pydantic import BaseModel, Field as PydanticField, field_validator

from app.schemas.constraints import UserConstraint
from app.schemas.fields import Field
from app.schemas.filters import SearchFilters
from app.schemas.property_semantics import OccupancyType, PropertyType
from app.config.llm import get_gemini_model

class IntentRoute(BaseModel):
    city: Optional[str] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    nights: int | None = None
    adults: int | None = None
    children: int | None = None
    rooms: int | None = None

    # Canonical semantic state.
    constraints: list[UserConstraint] = PydanticField(default_factory=list)

    filters: SearchFilters = PydanticField(default_factory=SearchFilters)
    property_types: list[PropertyType] = PydanticField(default_factory=list)
    occupancy_types: list[OccupancyType] = PydanticField(default_factory=list)


    @field_validator(
        "constraints",
        "property_types",
        "occupancy_types",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v

    @field_validator("filters", mode="before")
    @classmethod
    def _none_to_default_filters(cls, v):
        return SearchFilters() if v is None else v


def build_intent_router_agent() -> Agent:
    allowed_fields = [f.value for f in Field]
    schema = IntentRoute.model_json_schema()

    instruction = f"""
You are an intent router for a booking search assistant.

Return ONLY VALID JSON matching this schema:
{json.dumps(schema, ensure_ascii=False)}

Rules:

GENERAL:
- The user may write in ANY language.
- Return ONLY a valid JSON object. No markdown. No explanations.
- constraints is the source of truth for user constraints.
- Always return arrays, never null, for:
  - constraints
  - property_types
  - occupancy_types
  - unknown_requests

IMPORTANT CONTRACT:
- Preserve user meaning in constraints.
- unknown_requests is a legacy compatibility field only.
- Do NOT use unknown_requests as the main fallback bucket for user meaning.
- In normal cases, return unknown_requests=[].
- If something is meaningful but not safely mappable, keep it as an unresolved constraint.

CITY:
- Normalize city names to the English form used by providers when possible.
- Example: Bakı -> Baku, Баку -> Baku, Tiflis -> Tbilisi

DATES:
- If the user provides both check-in and check-out, fill both.
- If the user says "from X for N nights", set check_in and nights.
- If the user provides only one date, set check_in only.
- Do not invent dates.

GUESTS AND ROOMS:
- "for 3 people" -> adults=3, children=0
- "2 adults and 1 child" -> adults=2, children=1
- "3 rooms" -> rooms=3

FILTERS:
Some user requests are structured numeric constraints and MUST go into filters, not constraints:
- bedrooms
- area / square meters / sqm
- bathrooms
- price

Price rules:
- per night / nightly -> filters.price.scope = "per_night"
- total / overall / for whole stay -> filters.price.scope = "total_stay"
- Use max_amount unless the user clearly asks for a minimum
- Include currency when mentioned

Use property_types only for:
- ryokan
- hotel
- apartment
- resort
- villa
- bed_and_breakfast
- holiday_home
- guest_house
- hostel
- capsule_hotel
- homestay
- chalet
- lodge
- campsite
- country_house
- love_hotel
- house
- aparthotel
- guesthouse

Synonyms:
- рекан / риокан / ryokan / ryokans / 旅館 -> ryokan
- гестхаус / guest house / guesthouse -> guest_house
- b&b / bed and breakfast -> bed_and_breakfast
- holiday home / vacation home -> holiday_home
- capsule hotel / капсульный отель -> capsule_hotel
- love hotel -> love_hotel

Use occupancy_types only for:
- entire_place
- private_room
- shared_room
- hotel_room

Do NOT duplicate property_types or occupancy_types inside constraints.

CONSTRAINTS:
Use constraints for meaningful non-numeric user requirements such as:
- amenities
- policies
- location preferences
- layout preferences
- semantic preferences that do not fit the structured schema

Each constraint object should preserve user meaning.

Constraint fields:
- raw_text: short phrase representing the original user meaning
- normalized_text: concise normalized English phrase
- priority:
  - "must" for required constraints
  - "nice" for preferences / desirable items
  - "forbidden" for exclusions / things the user does not want
- category:
  - amenity
  - policy
  - location
  - layout
  - numeric
  - property_type
  - occupancy
  - other
- mapping_status:
  - "known" if the constraint can be grounded to one or more canonical fields
  - "unresolved" if it cannot be cleanly mapped
- mapped_fields:
  - use ONLY canonical keys from allowed_fields: {allowed_fields}
  - use [] when unresolved
- evidence_strategy:
  - "structured" for canonical provider/amenity style matching
  - "textual" for description/policy/highlights evidence
  - "none" only if there is truly no downstream evidence path yet

KNOWN MAPPING:
If a constraint clearly maps to canonical fields:
- set mapping_status = "known"
- fill mapped_fields
- usually set evidence_strategy = "structured"

Examples:
- "place for cooking" -> known, mapped_fields=["kitchen"]
- "hair dryer" -> known, mapped_fields=["hair_dryer"]
- "can live with dog" -> known, mapped_fields=["pet_friendly"]

UNRESOLVED CONSTRAINTS:
If a constraint is meaningful but cannot be safely mapped to canonical fields:
- keep it as a constraint
- set mapping_status = "unresolved"
- mapped_fields = []
- choose the best category
- use evidence_strategy = "textual" for unresolved constraints that must be checked via listing text

Examples:
- "in the city center" -> unresolved location, evidence_strategy="textual"
- "quiet neighborhood" -> unresolved location, evidence_strategy="textual"
- "near the metro" -> unresolved location, evidence_strategy="textual"
- "close to the beach" -> unresolved location, evidence_strategy="textual"
- "good for working" -> unresolved other or amenity, evidence_strategy="textual"
- "not on the first floor" -> unresolved layout, evidence_strategy="textual"

IMPORTANT:
- Do NOT force uncertain meaning into the wrong canonical field.
- Do NOT drop meaningful constraints.
- Do NOT put numeric constraints into constraints if they fit filters.
- Do NOT use constraints for property_types / occupancy_types if they already fit dedicated slots.
- Do NOT use unknown_requests as the semantic catch-all.
- A user may express positive, negative, and soft-preference constraints in one message.

Examples:

User: "I want an apartment in Baku from 10 to 15 April for 4 people with a place for cooking and ideally a balcony"
Return a JSON where:
- city="Baku"
- check_in / check_out set
- adults=4
- property_types=["apartment"]
- constraints contains:
  - must constraint for cooking mapped to ["kitchen"]
  - nice constraint for balcony mapped to ["balcony"]
- unknown_requests=[]

User: "хочу чтобы можно было жить с собакой и желательно в центре"
Return constraints containing:
- must policy constraint mapped to ["pet_friendly"]
- nice unresolved location constraint for city center
- unknown_requests=[]

User: "без шумного района"
Return constraints containing:
- forbidden unresolved location/other constraint with textual evidence strategy
- unknown_requests=[]
""".strip()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY/GOOGLE_API_KEY")

    llm = Gemini(
        model=get_gemini_model(),
        api_key=api_key,
    )

    return Agent(
        name="intent_router",
        model=llm,
        instruction=instruction,
    )