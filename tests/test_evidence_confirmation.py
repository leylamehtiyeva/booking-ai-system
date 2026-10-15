"""
Evidence confirmation (M1, baseline case 0041): a textual-fallback YES must
rest on verifiable evidence about the requested property itself.

The model call is scripted - each case returns an analysis the model gave
or could give - so these tests exercise the real deterministic decision
(_normalize_result + _decide_from_analysis via
resolve_listing_constraints_with_fallback), including the check that every
quoted snippet exists in the listing evidence that was sent to the model.

Original failure mode, kept verbatim: the 2026-10-15 baseline response for
0041 (no evidence_type field - it did not exist) claimed strong direct
support for "nice atmosphere" from "Quiet Stay Hotel" / "quiet residential
area".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.logic import constraint_evidence_resolution
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import FieldMatch, Ternary


def _constraint(text: str, *, category=ConstraintCategory.OTHER, fields: list[Field] | None = None) -> UserConstraint:
    known = bool(fields)
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=category,
        mapping_status=ConstraintMappingStatus.KNOWN if known else ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=fields or [],
        evidence_strategy=EvidenceStrategy.STRUCTURED if known else EvidenceStrategy.TEXTUAL,
    )


def _signal(relation: str, strength: str, snippet: str, source: str = "description", **extra) -> dict:
    return {"relation": relation, "strength": strength, "snippet": snippet, "source": source, **extra}


def _analysis(*signals: dict, direct_support=False, direct_contradiction=False, reason="scripted") -> dict:
    return {
        "signals": list(signals),
        "has_direct_support": direct_support,
        "has_direct_contradiction": direct_contradiction,
        "has_only_weak_or_indirect_evidence": False,
        "has_conflicting_evidence": False,
        "evidence_missing": not signals,
        "condition_or_extra_requirement_present": False,
        "confidence": 0.9 if signals else 0.0,
        "reason": reason,
    }


@pytest.fixture
def llm_answers(monkeypatch):
    """Scripted google-genai client: returns the queued analysis as model text."""
    answers: list[dict] = []

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=json.dumps(answers.pop(0)), usage_metadata=None)

    monkeypatch.setattr(
        constraint_evidence_resolution,
        "_gemini_client",
        lambda: SimpleNamespace(models=Models()),
    )
    yield answers
    assert not answers, "scripted answer not consumed"


# baku-hotel-6 exactly as in the fixtures (the 0041 listing).
QUIET_HOTEL = ListingRaw(
    id="baku-hotel-6",
    name="Quiet Stay Hotel",
    city="Baku",
    property_type="hotel",
    description="Basic amenities, private bathroom. Small hotel about 35 sqm rooms in a quiet residential area, not city center.",
)


def _listing(description: str, facilities: list[str] = ()) -> ListingRaw:
    return ListingRaw(
        id="l1",
        name="Test Listing",
        city="Baku",
        description=description,
        facilities=[{"name": name} for name in facilities],
    )


async def _resolve(constraint: UserConstraint, listing: ListingRaw = QUIET_HOTEL):
    # As in the real pipeline: a mapped requirement reaches the textual
    # fallback only after the structured matcher found it UNCERTAIN.
    structured = {field: FieldMatch(value=Ternary.UNCERTAIN) for field in constraint.mapped_fields}
    (result,) = await constraint_evidence_resolution.resolve_listing_constraints_with_fallback(
        listing=listing,
        constraints=[constraint],
        structured_matches_by_field=structured,
        policy=FallbackPolicy(enabled=True, must_only=False),
    )
    return result


async def _decision(constraint: UserConstraint, listing: ListingRaw = QUIET_HOTEL) -> str:
    return (await _resolve(constraint, listing)).decision


# --- 0041: the original failure mode and its variants ------------------------------------


async def test_recorded_0041_baseline_response_is_no_longer_a_yes(llm_answers):
    """Verbatim shape of the baseline answer: no evidence_type, strong 'support'."""
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Quiet Stay Hotel", source="title"),
            _signal("supports", "strong", "in a quiet residential area, not city center."),
            direct_support=True,
            reason="The title and description both strongly suggest a quiet and pleasant atmosphere, "
            "directly supporting the user's constraint.",
        )
    )

    result = await _resolve(_constraint("nice atmosphere"))

    assert result.decision == "UNCERTAIN"
    assert "basis" in result.reason.lower()


async def test_associated_feature_labelled_strong_and_direct_is_not_a_yes(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Quiet Stay Hotel", source="title", evidence_type="associated_feature"),
            _signal("supports", "strong", "in a quiet residential area, not city center.", evidence_type="associated_feature"),
            direct_support=True,
            reason="The title and description strongly suggest a quiet and pleasant atmosphere.",
        )
    )

    result = await _resolve(_constraint("nice atmosphere"))

    assert result.decision == "UNCERTAIN"
    assert "related" in result.reason.lower()


async def test_wifi_does_not_confirm_a_workspace(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Free WiFi", source="facilities", evidence_type="associated_feature"),
            direct_support=True,
        )
    )

    workspace = _constraint("workspace", category=ConstraintCategory.AMENITY, fields=[Field.WORKSPACE])
    assert await _decision(workspace, _listing("Central flat.", ["Free WiFi"])) == "UNCERTAIN"


async def test_model_mislabelling_indirect_evidence_as_direct_is_not_caught(llm_answers):
    """
    KNOWN LIMITATION - pinned on purpose. If the model calls an associated
    feature a direct statement, the deterministic layer cannot tell (no
    lexical guard by design); preventing this relies on the prompt and is
    measured by targeted evaluation on the real model.
    """
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "in a quiet residential area, not city center.", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    assert await _decision(_constraint("nice atmosphere")) == "YES"


@pytest.mark.parametrize("bad_type", ["inferred", "", None])
async def test_missing_or_unknown_evidence_type_is_not_a_basis_for_yes(llm_answers, bad_type):
    signal = _signal("supports", "strong", "Free WiFi", source="facilities")
    if bad_type is not None:
        signal["evidence_type"] = bad_type
    llm_answers.append(_analysis(signal, direct_support=True))

    wifi = _constraint("wifi", category=ConstraintCategory.AMENITY, fields=[Field.WIFI])
    assert await _decision(wifi, _listing("Central flat.", ["Free WiFi"])) == "UNCERTAIN"


# --- several signals --------------------------------------------------------------------


GUESTS_PRAISE = _listing(
    "Guests consistently praise the warm, romantic atmosphere. Located in a quiet residential area."
)


async def test_direct_signal_alongside_an_associated_one_confirms(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Located in a quiet residential area.", evidence_type="associated_feature"),
            _signal("supports", "strong", "Guests consistently praise the warm, romantic atmosphere.", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    assert await _decision(_constraint("nice atmosphere"), GUESTS_PRAISE) == "YES"


async def test_only_weak_direct_signal_next_to_strong_associated_one_is_uncertain(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Located in a quiet residential area.", evidence_type="associated_feature"),
            _signal("weakly_supports", "weak", "Guests consistently praise the warm, romantic atmosphere.", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    assert await _decision(_constraint("nice atmosphere"), GUESTS_PRAISE) == "UNCERTAIN"


# --- direct statements and measurements ----------------------------------------------------


async def test_direct_description_of_the_quality_confirms_it(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Guests consistently praise the warm, romantic atmosphere.", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    assert await _decision(_constraint("nice atmosphere"), GUESTS_PRAISE) == "YES"


BIG_FLAT = _listing("Apartment about 120 square meters near the park.")


async def test_a_strong_direct_measurement_can_confirm(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Apartment about 120 square meters", evidence_type="measurement"),
            direct_support=True,
        )
    )

    assert await _decision(_constraint("spacious", category=ConstraintCategory.LAYOUT), BIG_FLAT) == "YES"


@pytest.mark.parametrize(
    ("relation", "strength", "direct"),
    [
        ("weakly_supports", "weak", False),
        ("supports", "medium", True),
        ("supports", "strong", False),  # the model itself did not call it direct support
    ],
)
async def test_a_measurement_alone_is_not_automatically_a_yes(llm_answers, relation, strength, direct):
    llm_answers.append(
        _analysis(
            _signal(relation, strength, "Apartment about 120 square meters", evidence_type="measurement"),
            direct_support=direct,
        )
    )

    assert await _decision(_constraint("spacious", category=ConstraintCategory.LAYOUT), BIG_FLAT) == "UNCERTAIN"


async def test_a_contradicting_measurement_is_no(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("contradicts", "strong", "Studio of about 18 square meters.", evidence_type="measurement"),
            direct_contradiction=True,
        )
    )

    tiny = _listing("Studio of about 18 square meters.")
    assert await _decision(_constraint("spacious", category=ConstraintCategory.LAYOUT), tiny) == "NO"


# --- snippet authenticity ----------------------------------------------------------------


async def test_a_quote_that_is_not_in_the_listing_cannot_confirm(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "Guests love the romantic atmosphere.", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    result = await _resolve(_constraint("nice atmosphere"))

    assert result.decision == "UNCERTAIN"
    assert result.evidence == []  # an unverifiable quote is not shown as evidence
    assert "not found" in result.reason.lower()


async def test_a_quote_that_is_not_in_the_listing_cannot_reject(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("contradicts", "strong", "The hotel is very noisy at night.", evidence_type="direct_statement"),
            direct_contradiction=True,
        )
    )

    assert await _decision(_constraint("quiet")) == "UNCERTAIN"


async def test_quote_matching_ignores_case_punctuation_and_whitespace(llm_answers):
    llm_answers.append(
        _analysis(
            _signal("supports", "strong", "free   WIFI!", source="facilities", evidence_type="direct_statement"),
            direct_support=True,
        )
    )

    wifi = _constraint("wifi", category=ConstraintCategory.AMENITY, fields=[Field.WIFI])
    assert await _decision(wifi, _listing("Central flat.", ["Free WiFi"])) == "YES"


# --- must keep working ---------------------------------------------------------------------


async def test_no_relevant_evidence_is_uncertain(llm_answers):
    llm_answers.append(_analysis())

    assert await _decision(_constraint("nice atmosphere")) == "UNCERTAIN"


async def test_explicit_contradiction_is_no(llm_answers):
    noisy = _listing("The lobby is noisy and the atmosphere feels unwelcoming.")
    llm_answers.append(
        _analysis(
            _signal("contradicts", "strong", "The lobby is noisy and the atmosphere feels unwelcoming.", evidence_type="direct_statement"),
            direct_contradiction=True,
        )
    )

    assert await _decision(_constraint("nice atmosphere"), noisy) == "NO"


@pytest.mark.parametrize(
    ("text", "category", "fields", "snippet", "facilities"),
    [
        ("wifi", ConstraintCategory.AMENITY, [Field.WIFI], "Free WiFi", ["Free WiFi"]),
        ("breakfast", ConstraintCategory.AMENITY, [], "Breakfast is included in the room rate.", []),
        ("parking", ConstraintCategory.AMENITY, [Field.PARKING], "Free private parking on site.", []),
        ("sea view", ConstraintCategory.LOCATION, [], "Balcony with sea view.", []),
    ],
)
async def test_factual_requirements_are_confirmed_by_a_direct_statement(llm_answers, text, category, fields, snippet, facilities):
    listing = _listing(snippet if not facilities else "Central flat.", facilities)
    llm_answers.append(
        _analysis(_signal("supports", "strong", snippet, evidence_type="direct_statement"), direct_support=True)
    )

    assert await _decision(_constraint(text, category=category, fields=fields), listing) == "YES"


def test_fallback_prompt_defines_evidence_types_with_contrasting_examples():
    prompt = constraint_evidence_resolution._build_system_prompt()

    assert '"evidence_type": "direct_statement | measurement | associated_feature"' in prompt
    assert '"quiet residential area" for "nice atmosphere"' in prompt
    assert '"Free WiFi" for "workspace"' in prompt
    assert '"about 85 square meters" for "spacious"' in prompt
    assert "An associated feature is at most weakly_supports, never supports." in prompt
    assert "Copy each snippet exactly from the listing evidence." in prompt
