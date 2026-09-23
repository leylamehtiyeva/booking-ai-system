from app.logic.matcher_structured import match_forbidden_fields, match_listing_structured
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw, Room, RoomOption
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest

from app.schemas.constraints import (
    UserConstraint,
    ConstraintPriority,
    ConstraintCategory,
    ConstraintMappingStatus,
    EvidenceStrategy,
)


def test_match_listing_structured_uses_signal_rules():
    listing = ListingRaw(
        id="x1",
        name="CHINAR Apartment DeLux",
        property_type="apartment",
        description="Spacious stay in Baku center",
        facilities=[{"name": "Free WiFi"}],
        rooms=[
            Room(
                name="Three-Bedroom Apartment with Balcony",
                facilities=[
                    "Private kitchen",
                    "Private bathroom",
                    "Washing machine",
                ],
                options=[
                    RoomOption(
                    name="Flexible rate",
                    choices=[
                        "Free cancellation",
                        "No prepayment needed",
                    ],
                )
                ],
            )
        ],
        policies=[
            {"title": "Pets", "content": "Pets are allowed on request."},
        ],
    )

    req = SearchRequest(
    city="Baku",
    check_in="2026-04-08",
    check_out="2026-04-15",
    adults=2,
    children=0,
    rooms=1,
    currency="USD",
    constraints=[
        # MUST
        UserConstraint(
            raw_text="private bathroom",
            normalized_text="private bathroom",
            priority=ConstraintPriority.MUST,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.PRIVATE_BATHROOM],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
        UserConstraint(
            raw_text="kitchen",
            normalized_text="kitchen",
            priority=ConstraintPriority.MUST,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.KITCHEN],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),

        # NICE
        UserConstraint(
            raw_text="wifi",
            normalized_text="wifi",
            priority=ConstraintPriority.NICE,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.WIFI],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
        UserConstraint(
            raw_text="washing machine",
            normalized_text="washing machine",
            priority=ConstraintPriority.NICE,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.WASHING_MACHINE],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
        UserConstraint(
            raw_text="free cancellation",
            normalized_text="free cancellation",
            priority=ConstraintPriority.NICE,
            category=ConstraintCategory.POLICY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.FREE_CANCELLATION],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
        UserConstraint(
            raw_text="pet friendly",
            normalized_text="pet friendly",
            priority=ConstraintPriority.NICE,
            category=ConstraintCategory.POLICY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.PET_FRIENDLY],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
        UserConstraint(
            raw_text="balcony",
            normalized_text="balcony",
            priority=ConstraintPriority.NICE,
            category=ConstraintCategory.AMENITY,
            mapping_status=ConstraintMappingStatus.KNOWN,
            mapped_fields=[Field.BALCONY],
            evidence_strategy=EvidenceStrategy.STRUCTURED,
        ),
    ],
)

    report = match_listing_structured(listing, req)

    assert report.matches[Field.PRIVATE_BATHROOM].value == Ternary.YES
    assert report.matches[Field.KITCHEN].value == Ternary.YES

    assert report.matches[Field.WIFI].value == Ternary.YES
    assert report.matches[Field.WASHING_MACHINE].value == Ternary.YES
    assert report.matches[Field.FREE_CANCELLATION].value == Ternary.YES
    assert report.matches[Field.PET_FRIENDLY].value == Ternary.YES
    assert report.matches[Field.BALCONY].value == Ternary.YES


def _forbidden_smoking_constraint() -> UserConstraint:
    return UserConstraint(
        raw_text="no smoking",
        normalized_text="no smoking",
        priority=ConstraintPriority.FORBIDDEN,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.SMOKING_ALLOWED],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def test_match_forbidden_fields_detects_violation():
    listing = ListingRaw(
        id="f1",
        name="Smokers Loft",
        policies=[
            {"title": "Smoking", "content": "Smoking allowed on the balcony."},
        ],
    )

    result = match_forbidden_fields(listing, [_forbidden_smoking_constraint()])

    assert result[Field.SMOKING_ALLOWED].value == Ternary.YES


def test_match_forbidden_fields_confirms_safe_listing():
    listing = ListingRaw(
        id="f2",
        name="Quiet Apartment",
        policies=[
            {"title": "Smoking", "content": "Smoking is not allowed anywhere on the property."},
        ],
    )

    result = match_forbidden_fields(listing, [_forbidden_smoking_constraint()])

    assert result[Field.SMOKING_ALLOWED].value == Ternary.NO


def _forbidden_non_smoking_constraint() -> UserConstraint:
    return UserConstraint(
        raw_text="no smoking",
        normalized_text="non-smoking",
        priority=ConstraintPriority.FORBIDDEN,
        category=ConstraintCategory.POLICY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.NON_SMOKING],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def test_match_forbidden_fields_non_smoking_confirms_safe_listing():
    listing = ListingRaw(
        id="ns1",
        name="Non-smoking Apartment",
        policies=[
            {"title": "Smoking", "content": "Smoking is not allowed anywhere on the property."},
        ],
    )

    result = match_forbidden_fields(listing, [_forbidden_non_smoking_constraint()])

    assert result[Field.NON_SMOKING].value == Ternary.YES


def test_match_forbidden_fields_non_smoking_detects_violation():
    listing = ListingRaw(
        id="ns2",
        name="Smokers Loft",
        policies=[
            {"title": "Smoking", "content": "Smoking allowed on the balcony."},
        ],
    )

    result = match_forbidden_fields(listing, [_forbidden_non_smoking_constraint()])

    assert result[Field.NON_SMOKING].value == Ternary.NO


def test_forbidden_fields_are_excluded_from_structured_matches():
    listing = ListingRaw(
        id="f3",
        name="Smokers Loft",
        policies=[
            {"title": "Smoking", "content": "Smoking allowed on the balcony."},
        ],
    )

    req = SearchRequest(
        city="Baku",
        check_in="2026-04-08",
        check_out="2026-04-15",
        adults=2,
        children=0,
        rooms=1,
        currency="USD",
        constraints=[_forbidden_smoking_constraint()],
    )

    report = match_listing_structured(listing, req)

    assert Field.SMOKING_ALLOWED not in report.matches
