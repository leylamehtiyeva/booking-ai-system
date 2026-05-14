from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from app.schemas.fields import Field




@dataclass(frozen=True)
class FieldRule:
    aliases: Tuple[str, ...]
    preferred_path_prefixes: Tuple[str, ...] = ()
    negative_aliases: Tuple[str, ...] = ()


FIELD_RULES: Dict[Field, FieldRule] = {
    Field.KITCHEN: FieldRule(
        aliases=("kitchen", "private kitchen", "kitchenette", "shared kitchen"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no kitchen",),
    ),
    Field.PRIVATE_BATHROOM: FieldRule(
        aliases=("private bathroom",),
        preferred_path_prefixes=("rooms[", "listing.description"),
        negative_aliases=("shared bathroom", "shared toilet"),
    ),
    Field.WIFI: FieldRule(
        aliases=("wifi", "wi-fi", "free wifi", "wireless internet"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no wifi", "wifi not available", "internet not available"),
    ),
    Field.AIR_CONDITIONING: FieldRule(
        aliases=("air conditioning", "air-conditioned"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no air conditioning",),
    ),
    Field.WASHING_MACHINE: FieldRule(
        aliases=("washing machine", "washer", "laundry machine"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no washing machine", "washer not available"),
    ),
    Field.OVEN: FieldRule(
        aliases=("oven",),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no oven",),
    ),
    Field.MICROWAVE: FieldRule(
        aliases=("microwave",),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no microwave",),
    ),
    Field.REFRIGERATOR: FieldRule(
        aliases=("refrigerator", "fridge"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no refrigerator", "no fridge"),
    ),
    Field.BALCONY: FieldRule(
        aliases=("balcony", "private balcony", "patio", "terrace"),
        preferred_path_prefixes=("rooms[", "highlights", "listing.description"),
        negative_aliases=("no balcony",),
    ),
    Field.KETTLE: FieldRule(
        aliases=("kettle", "electric kettle"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no kettle",),
    ),
    Field.COFFEE_MACHINE: FieldRule(
        aliases=("coffee machine", "coffee maker"),
        preferred_path_prefixes=("rooms[", "listing.facilities", "listing.description"),
        negative_aliases=("no coffee machine", "no coffee maker"),
    ),
    Field.FREE_CANCELLATION: FieldRule(
        aliases=("free cancellation",),
        preferred_path_prefixes=("rooms[", "policies", "highlights"),
        negative_aliases=("non-refundable", "no free cancellation"),
    ),
    Field.PET_FRIENDLY: FieldRule(
        aliases=(
            "pets allowed",
            "pet friendly",
            "pets are allowed",
            "animals allowed",
            "dogs allowed",
            "cats allowed",
            "pets welcome",
        ),
        preferred_path_prefixes=("policies", "listing.description", "highlights"),
        negative_aliases=(
            "no pets",
            "pets not allowed",
            "pets are not allowed",
            "animals are not allowed",
            "dogs are not allowed",
            "cats are not allowed",
        ),
    ),
    Field.SMOKING_ALLOWED: FieldRule(
        aliases=(
            "smoking allowed",
            "smoking is allowed",
        ),
        preferred_path_prefixes=("policies", "listing.description"),
        negative_aliases=(
            "smoking not allowed",
            "smoking is not allowed",
            "no smoking",
        ),
    ),
    Field.PARTIES_ALLOWED: FieldRule(
        aliases=(
            "parties allowed",
            "events allowed",
            "parties are allowed",
            "events are allowed",
        ),
        preferred_path_prefixes=("policies", "listing.description"),
        negative_aliases=(
            "parties not allowed",
            "events not allowed",
            "parties/events are not allowed",
            "parties are not allowed",
            "events are not allowed",
            "no parties",
        ),
    ),
    Field.CHILDREN_ALLOWED: FieldRule(
        aliases=(
            "children are welcome",
            "children of any age are welcome",
            "child policies",
            "children allowed",
            "family friendly",
        ),
        preferred_path_prefixes=("policies", "listing.description"),
        negative_aliases=(
            "children not allowed",
            "no children",
            "adults only",
        ),
    ),
    Field.PARKING: FieldRule(
        aliases=(
            "parking",
            "free parking",
            "private parking",
            "parking available",
        ),
        preferred_path_prefixes=("highlights", "listing.facilities", "listing.description"),
        negative_aliases=(
            "no parking",
            "parking not available",
        ),
    ),
}