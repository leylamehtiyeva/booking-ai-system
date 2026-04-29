from __future__ import annotations
from app.schemas.fallback_policy import FallbackPolicy

from app.logic import constraint_evidence_resolution as cer
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)


from app.schemas.fields import Field
from app.schemas.listing import ListingRaw, Room
from app.schemas.match import FieldMatch, Ternary



def test_unresolved_must_textual_constraint_is_fallback_eligible():
    constraint = UserConstraint(
        raw_text="satellite TV",
        normalized_text="satellite TV",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    policy = FallbackPolicy(enabled=True, must_only=True)

    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=None,
        policy=policy,
    ) is True


def test_known_constraint_with_uncertain_structured_match_is_fallback_eligible():
    constraint = UserConstraint(
        raw_text="place for cooking",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )

    policy = FallbackPolicy(enabled=True, must_only=True)

    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=Ternary.UNCERTAIN,
        policy=policy,
    ) is True


def test_known_constraint_with_positive_structured_match_is_not_fallback_eligible():
    constraint = UserConstraint(
        raw_text="place for cooking",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )

    policy=FallbackPolicy(enabled=True, must_only=True)
    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=Ternary.YES,
        policy=policy,
    ) is False


def test_decide_from_analysis_returns_no_for_strong_contradiction():
    analysis = cer.ConstraintEvidenceAnalysis(
        signals=[
            cer.EvidenceSignal(
                relation="contradicts",
                strength="strong",
                snippet="Parking costs $20 per day.",
                source="policies",
                path="policies[0].content",
            )
        ],
        has_direct_contradiction=True,
        evidence_missing=False,
    )

    assert cer._decide_from_analysis(analysis) == "NO"


def test_decide_from_analysis_returns_yes_for_strong_support():
    analysis = cer.ConstraintEvidenceAnalysis(
        signals=[
            cer.EvidenceSignal(
                relation="supports",
                strength="strong",
                snippet="Free WiFi is available in all rooms.",
                source="facilities",
                path="listing.facilities[0]",
            )
        ],
        has_direct_support=True,
        evidence_missing=False,
    )

    assert cer._decide_from_analysis(analysis) == "YES"


def test_decide_from_analysis_returns_uncertain_for_missing_evidence():
    analysis = cer.ConstraintEvidenceAnalysis(
        signals=[],
        evidence_missing=True,
    )

    assert cer._decide_from_analysis(analysis) == "UNCERTAIN"


def test_normalize_result_uses_signals_not_llm_decision():
    req = cer.ConstraintResolutionRequest(
        listing_id="listing-1",
        listing_title="Demo listing",
        constraint_id="c1",
        raw_text="free parking",
        normalized_text="free parking",
        priority="must",
        category="amenity",
        mapping_status="unresolved",
        evidence_strategy="textual",
        mapped_fields=[],
        structured_value=None,
        resolver_type="textual",
        listing_evidence=[
            {
                "source": "policies",
                "path": "policies[0].content",
                "text": "Parking costs $20 per day.",
            }
        ],
    )

    raw = {
        "signals": [
            {
                "relation": "contradicts",
                "strength": "strong",
                "snippet": "Parking costs $20 per day.",
                "source": "policies",
                "path": "policies[0].content",
                "explanation": "Paid parking contradicts the requirement for free parking.",
            }
        ],
        "has_direct_support": False,
        "has_direct_contradiction": True,
        "has_only_weak_or_indirect_evidence": False,
        "has_conflicting_evidence": False,
        "evidence_missing": False,
        "condition_or_extra_requirement_present": False,
        "confidence": 0.9,
        "reason": "The listing says parking is paid.",
    }

    result = cer._normalize_result(raw, req)

    assert result.decision == "NO"
    assert result.resolution_status == "failed"
    assert result.explicit_negative is True
    assert result.analysis is not None
    assert result.analysis.has_direct_contradiction is True
    assert result.evidence[0].snippet == "Parking costs $20 per day."


async def test_resolve_listing_constraints_with_fallback_returns_unified_results(monkeypatch):
    listing = ListingRaw(
        id="listing-123",
        name="Apartment STEL",
        description="This apartment includes a fully equipped kitchen and balcony.",
        facilities=["Kitchen", "Washing machine"],
        rooms=[
            Room(
                name="Studio",
                facilities=["Private kitchen", "Balcony"],
                options=[],
            )
        ],
    )

    unresolved_constraint = UserConstraint(
        raw_text="satellite TV",
        normalized_text="satellite TV",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    kitchen_constraint = UserConstraint(
        raw_text="place for cooking",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )

    async def _fake_resolve(req, *, model=None):
        if req.normalized_text == "satellite TV":
            return cer.ConstraintResolutionResult(
                listing_id=req.listing_id,
                listing_title=req.listing_title,
                constraint_id=req.constraint_id,
                raw_text=req.raw_text,
                normalized_text=req.normalized_text,
                resolver_type="textual",
                decision="UNCERTAIN",
                resolution_status="uncertain",
                confidence=0.35,
                reason="Satellite TV is not explicitly confirmed in the listing.",
                evidence=[],
                structured_value_before=req.structured_value,
                explicit_negative=False,
            )

        return cer.ConstraintResolutionResult(
            listing_id=req.listing_id,
            listing_title=req.listing_title,
            constraint_id=req.constraint_id,
            raw_text=req.raw_text,
            normalized_text=req.normalized_text,
            resolver_type="textual",
            decision="YES",
            resolution_status="matched",
            confidence=0.92,
            reason="Kitchen is explicitly supported by listing text.",
            evidence=[
                cer.ConstraintEvidence(
                    snippet="Private kitchen",
                    source="room_facilities",
                    path="rooms[0].facilities",
                )
            ],
            structured_value_before=req.structured_value,
            explicit_negative=False,
        )

    monkeypatch.setattr(cer, "resolve_constraint_via_textual_evidence", _fake_resolve)

    structured_matches = {
        Field.KITCHEN: FieldMatch(
            value=Ternary.UNCERTAIN,
            confidence=0.4,
            evidence=[],
        )
    }

    policy = FallbackPolicy(enabled=True)

    results = await cer.resolve_listing_constraints_with_fallback(
        listing=listing,
        constraints=[unresolved_constraint, kitchen_constraint],
        structured_matches_by_field=structured_matches,
        policy=policy,
    )

    assert len(results) == 2

    by_name = {r.normalized_text: r for r in results}

    assert by_name["satellite TV"].decision == "UNCERTAIN"
    assert by_name["satellite TV"].resolution_status == "uncertain"

    assert by_name["kitchen"].decision == "YES"
    assert by_name["kitchen"].resolution_status == "matched"
    assert by_name["kitchen"].structured_value_before == "UNCERTAIN"
    assert by_name["kitchen"].evidence[0].snippet == "Private kitchen"
    
    
def test_fallback_policy_disables_eligibility():
    constraint = UserConstraint(
        raw_text="satellite TV",
        normalized_text="satellite TV",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    policy = FallbackPolicy(enabled=False)

    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=None,
        policy=policy,
    ) is False


def test_fallback_policy_can_disable_unresolved_path():
    constraint = UserConstraint(
        raw_text="satellite TV",
        normalized_text="satellite TV",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    policy = FallbackPolicy(
        enabled=True,
        run_for_unresolved=False,
        run_for_structured_uncertain=True,
    )

    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=None,
        policy=policy,
    ) is False


def test_fallback_policy_can_disable_structured_uncertain_path():
    constraint = UserConstraint(
        raw_text="place for cooking",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )

    policy = FallbackPolicy(
        enabled=True,
        run_for_unresolved=True,
        run_for_structured_uncertain=False,
    )

    assert cer.is_constraint_fallback_eligible(
        constraint,
        structured_value=Ternary.UNCERTAIN,
        policy=policy,
    ) is False


async def test_fallback_policy_limits_constraints_per_listing(monkeypatch):
    listing = ListingRaw(
        id="listing-123",
        name="Apartment STEL",
        description="This apartment includes a fully equipped kitchen and balcony.",
        facilities=["Kitchen", "Washing machine"],
        rooms=[
            Room(
                name="Studio",
                facilities=["Private kitchen", "Balcony"],
                options=[],
            )
        ],
    )

    constraints = [
        UserConstraint(
            raw_text=f"constraint-{i}",
            normalized_text=f"constraint-{i}",
            priority=ConstraintPriority.MUST,
            category=ConstraintCategory.OTHER,
            mapping_status=ConstraintMappingStatus.UNRESOLVED,
            mapped_fields=[],
            evidence_strategy=EvidenceStrategy.TEXTUAL,
        )
        for i in range(5)
    ]

    calls: list[str] = []

    async def _fake_resolve(req, *, model=None):
        calls.append(req.normalized_text)
        return cer.ConstraintResolutionResult(
            listing_id=req.listing_id,
            listing_title=req.listing_title,
            constraint_id=req.constraint_id,
            raw_text=req.raw_text,
            normalized_text=req.normalized_text,
            resolver_type="textual",
            decision="UNCERTAIN",
            resolution_status="uncertain",
            confidence=0.3,
            reason="Not explicitly confirmed.",
            evidence=[],
            structured_value_before=req.structured_value,
            explicit_negative=False,
        )

    monkeypatch.setattr(cer, "resolve_constraint_via_textual_evidence", _fake_resolve)

    policy = FallbackPolicy(
        enabled=True,
        max_constraints_per_listing=2,
    )

    results = await cer.resolve_listing_constraints_with_fallback(
        listing=listing,
        constraints=constraints,
        structured_matches_by_field={},
        policy=policy,
    )

    assert len(results) == 2
    assert calls == ["constraint-0", "constraint-1"]