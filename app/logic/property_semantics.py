from __future__ import annotations

from dataclasses import dataclass
from typing import List
from app.schemas.listing import ListingRaw
from app.schemas.property_semantics import OccupancyType, PropertyType
from app.schemas.match import Ternary, Evidence, EvidenceSource


@dataclass
class SemanticMatchResult:
    attribute: str
    value: Ternary
    actual_value: str | None
    evidence: List[Evidence]
    why: str


def _texts_for_listing(
    listing: ListingRaw,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    if listing.name:
        out.append(
            (
                "listing.name",
                listing.name,
            )
        )

    if listing.description:
        out.append(
            (
                "listing.description",
                listing.description,
            )
        )

    if listing.fine_print:
        out.append(
            (
                "listing.fine_print",
                listing.fine_print,
            )
        )

    for i, room in enumerate(
        listing.rooms or []
    ):
        if room.name:
            out.append(
                (
                    f"rooms[{i}].name",
                    room.name,
                )
            )

        for j, facility in enumerate(
            room.facilities or []
        ):
            if isinstance(facility, str):
                out.append(
                    (
                        (
                            f"rooms[{i}]"
                            f".facilities[{j}]"
                        ),
                        facility,
                    )
                )
                continue

            if facility.name:
                out.append(
                    (
                        (
                            f"rooms[{i}]"
                            f".facilities[{j}].name"
                        ),
                        facility.name,
                    )
                )

            if facility.overview:
                out.append(
                    (
                        (
                            f"rooms[{i}]"
                            f".facilities[{j}]"
                            ".overview"
                        ),
                        facility.overview,
                    )
                )

    return out

def _property_type_from_canonical_value(
    value: str | None,
) -> PropertyType | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    try:
        return PropertyType(normalized)
    except ValueError:
        return None

def detect_property_type(
    listing: ListingRaw,
) -> tuple[
    PropertyType | None,
    list[Evidence],
]:
    canonical = (
        _property_type_from_canonical_value(
            listing.property_type
        )
    )

    if canonical is not None:
        return (
            canonical,
            [
                Evidence(
                    source=(
                        EvidenceSource.STRUCTURED
                    ),
                    path="listing.property_type",
                    snippet=listing.property_type,
                )
            ],
        )

    # Fallback only when structured
    # property_type is unavailable.
    texts = _texts_for_listing(listing)

    rules = [
        (
            PropertyType.CAPSULE_HOTEL,
            ["capsule hotel", "capsule"],
        ),
        (
            PropertyType.BED_AND_BREAKFAST,
            ["bed and breakfast", "b&b"],
        ),
        (
            PropertyType.HOLIDAY_HOME,
            ["holiday home", "vacation home"],
        ),
        (
            PropertyType.COUNTRY_HOUSE,
            ["country house"],
        ),
        (
            PropertyType.LOVE_HOTEL,
            ["love hotel"],
        ),
        (
            PropertyType.GUEST_HOUSE,
            ["guest house", "guesthouse"],
        ),
        (
            PropertyType.APARTHOTEL,
            ["aparthotel"],
        ),
        (
            PropertyType.RYOKAN,
            ["ryokan", "ryokans", "旅館"],
        ),
        (
            PropertyType.HOMESTAY,
            ["homestay"],
        ),
        (
            PropertyType.CAMPSITE,
            ["campsite", "camping"],
        ),
        (
            PropertyType.CHALET,
            ["chalet"],
        ),
        (
            PropertyType.LODGE,
            ["lodge"],
        ),
        (
            PropertyType.RESORT,
            ["resort"],
        ),
        (
            PropertyType.HOSTEL,
            ["hostel"],
        ),
        (
            PropertyType.VILLA,
            ["villa"],
        ),
        (
            PropertyType.HOTEL,
            ["hotel"],
        ),
        (
            PropertyType.APARTMENT,
            [
                "apartment",
                "apartments",
                "flat",
                "studio",
            ],
        ),
        (
            PropertyType.HOUSE,
            ["house", "home"],
        ),
    ]

    for path, text in texts:
        low = text.lower()

        for property_type, patterns in rules:
            for pattern in patterns:
                if pattern not in low:
                    continue

                return (
                    property_type,
                    [
                        Evidence(
                            source=(
                                EvidenceSource.STRUCTURED
                            ),
                            path=path,
                            snippet=pattern,
                        )
                    ],
                )

    return None, []

def detect_occupancy_type(listing: ListingRaw) -> tuple[OccupancyType | None, list[Evidence]]:
    texts = _texts_for_listing(listing)

    rules = [
        (OccupancyType.ENTIRE_PLACE, ["entire apartment", "entire place", "entire studio", "entire home"]),
        (OccupancyType.PRIVATE_ROOM, ["private room"]),
        (OccupancyType.SHARED_ROOM, ["shared room", "bed in dorm", "dormitory room", "shared dorm"]),
        (OccupancyType.HOTEL_ROOM, ["hotel room", "double room", "twin room"]),
    ]

    for path, text in texts:
        low = text.lower()
        for otype, patterns in rules:
            for pattern in patterns:
                if pattern in low:
                    return (
                        otype,
                        [
                            Evidence(
                                source=EvidenceSource.STRUCTURED,
                                path=path,
                                snippet=pattern,
                            )
                        ],
                    )

    return None, []


def match_property_types(
    listing: ListingRaw,
    requested: list[PropertyType] | None,
) -> SemanticMatchResult | None:
    if not requested:
        return None

    detected, evidence = detect_property_type(listing)
    if detected is None:
        return SemanticMatchResult(
            attribute="property_type",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="PROPERTY_TYPE: could not determine property type",
        )

    if detected not in requested:
        return SemanticMatchResult(
            attribute="property_type",
            value=Ternary.NO,
            actual_value=detected.value,
            evidence=evidence,
            why=f"PROPERTY_TYPE: detected {detected.value}, expected one of {[x.value for x in requested]}",
        )

    return SemanticMatchResult(
        attribute="property_type",
        value=Ternary.YES,
        actual_value=detected.value,
        evidence=evidence,
        why=f"PROPERTY_TYPE: matched {detected.value}",
    )


def match_occupancy_types(
    listing: ListingRaw,
    requested: list[OccupancyType] | None,
) -> SemanticMatchResult | None:
    if not requested:
        return None

    detected, evidence = detect_occupancy_type(listing)
    if detected is None:
        return SemanticMatchResult(
            attribute="occupancy_type",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="OCCUPANCY_TYPE: could not determine occupancy type",
        )

    if detected not in requested:
        return SemanticMatchResult(
            attribute="occupancy_type",
            value=Ternary.NO,
            actual_value=detected.value,
            evidence=evidence,
            why=f"OCCUPANCY_TYPE: detected {detected.value}, expected one of {[x.value for x in requested]}",
        )

    return SemanticMatchResult(
        attribute="occupancy_type",
        value=Ternary.YES,
        actual_value=detected.value,
        evidence=evidence,
        why=f"OCCUPANCY_TYPE: matched {detected.value}",
    )