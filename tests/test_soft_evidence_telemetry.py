from __future__ import annotations

from app.logic.soft_evidence_telemetry import summarize_soft_preference_evidence
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimResolutionMethod,
    EvidenceItem,
    EvidenceRelation,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SemanticVerifierUsage,
    SoftPreferenceEvidence,
)


def _deterministic_item(relation: EvidenceRelation) -> EvidenceItem:
    return EvidenceItem(
        evidence_text="Desk",
        relation=relation,
        resolution_method=ClaimResolutionMethod.DETERMINISTIC,
        resolution_status=EvidenceResolutionStatus.RESOLVED,
        source_type="room_facilities",
        source_path="rooms[0].facilities[0].name",
    )


def _gemini_item(relation: EvidenceRelation | None, status=EvidenceResolutionStatus.RESOLVED, error=None) -> EvidenceItem:
    return EvidenceItem(
        evidence_text="Some free text evidence.",
        relation=relation,
        resolution_method=ClaimResolutionMethod.GEMINI,
        resolution_status=status,
        error=error,
        source_type="description",
    )


def test_empty_input():
    summary = summarize_soft_preference_evidence([])
    assert summary["total_hotels"] == 0
    assert summary["hotels_requiring_gemini_pct"] == 0.0
    assert summary["deterministic_evidence_item_count"] == 0
    assert summary["gemini_evidence_item_count"] == 0


def test_deterministic_only_hotel_does_not_count_as_requiring_gemini():
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="RW1",
        hypothesis="The room has a desk or dedicated workspace.",
        evidence_items=[_deterministic_item(EvidenceRelation.SUPPORT)],
    )
    evidence = SoftPreferenceEvidence(claims=[claim], semantic_verifier=None)

    summary = summarize_soft_preference_evidence([evidence])

    assert summary["total_hotels"] == 1
    assert summary["hotels_requiring_gemini"] == 0
    assert summary["hotels_requiring_gemini_pct"] == 0.0
    assert summary["deterministic_evidence_item_count"] == 1
    assert summary["gemini_evidence_item_count"] == 0
    assert summary["claim_relation_counts"]["SUPPORT"] == 1


def test_mixed_claim_relation_is_counted():
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="RW2B",
        hypothesis="Internet connection is reliable enough for remote work.",
        evidence_items=[
            _gemini_item(EvidenceRelation.SUPPORT),
            _gemini_item(EvidenceRelation.CONTRADICT),
        ],
    )
    evidence = SoftPreferenceEvidence(
        claims=[claim],
        semantic_verifier=SemanticVerifierUsage(model="gemini-2.5-flash", calls=2),
    )

    summary = summarize_soft_preference_evidence([evidence])

    assert summary["claim_relation_counts"]["MIXED"] == 1
    assert summary["hotels_requiring_gemini"] == 1
    assert summary["hotels_requiring_gemini_pct"] == 100.0
    assert summary["gemini_evidence_item_count"] == 2


def test_verification_failed_item_is_preserved_and_counted_but_not_a_relation_vote():
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="BQ1",
        hypothesis="The breakfast is good in quality.",
        evidence_items=[
            _gemini_item(None, status=EvidenceResolutionStatus.VERIFICATION_FAILED, error="boom"),
        ],
    )
    evidence = SoftPreferenceEvidence(
        claims=[claim],
        semantic_verifier=SemanticVerifierUsage(
            model="gemini-2.5-flash", calls=1, parse_failures=1
        ),
    )

    summary = summarize_soft_preference_evidence([evidence])

    assert summary["claim_relation_counts"]["NOT_ENOUGH_EVIDENCE"] == 1
    assert summary["evidence_resolution_status_counts"]["verification_failed"] == 1
    assert summary["gemini_evidence_item_count"] == 1
    assert summary["verifier"]["total_parse_failures"] == 1


def test_verifier_usage_is_summed_across_hotels():
    claim_a = AtomicClaimResult.from_evidence_items(
        claim_id="Q1", hypothesis="h", evidence_items=[_gemini_item(EvidenceRelation.SUPPORT)]
    )
    claim_b = AtomicClaimResult.from_evidence_items(
        claim_id="Q2", hypothesis="h", evidence_items=[_gemini_item(EvidenceRelation.NOT_ENOUGH_EVIDENCE)]
    )

    evidence_a = SoftPreferenceEvidence(
        claims=[claim_a],
        semantic_verifier=SemanticVerifierUsage(
            model="gemini-2.5-flash",
            calls=1,
            latency_ms=100.0,
            input_tokens=50,
            output_tokens=10,
            total_tokens=60,
            estimated_cost_usd=0.001,
        ),
    )
    evidence_b = SoftPreferenceEvidence(
        claims=[claim_b],
        semantic_verifier=SemanticVerifierUsage(
            model="gemini-2.5-flash",
            calls=1,
            latency_ms=200.0,
            input_tokens=40,
            output_tokens=5,
            total_tokens=45,
            estimated_cost_usd=0.0008,
        ),
    )

    summary = summarize_soft_preference_evidence([evidence_a, evidence_b])

    verifier = summary["verifier"]
    assert verifier["total_calls"] == 2
    assert verifier["total_latency_ms"] == 300.0
    assert verifier["total_input_tokens"] == 90
    assert verifier["total_output_tokens"] == 15
    assert verifier["total_tokens"] == 105
    assert abs(verifier["total_cost_usd"] - 0.0018) < 1e-9
    assert summary["total_hotels"] == 2
    assert summary["hotels_requiring_gemini"] == 2
    assert summary["hotels_requiring_gemini_pct"] == 100.0


def test_retrieval_status_distribution_is_counted():
    success_claim = AtomicClaimResult.from_evidence_items(
        claim_id="Q1", hypothesis="h", evidence_items=[],
        retrieval_status=RetrievalStatus.SUCCESS,
    )
    partial_claim = AtomicClaimResult.from_evidence_items(
        claim_id="Q2", hypothesis="h", evidence_items=[],
        retrieval_status=RetrievalStatus.PARTIAL, retrieval_errors=["batch 2 failed"],
    )
    failed_claim = AtomicClaimResult.from_evidence_items(
        claim_id="Q3", hypothesis="h", evidence_items=[],
        retrieval_status=RetrievalStatus.FAILED, retrieval_errors=["query failed"],
    )
    evidence = SoftPreferenceEvidence(claims=[success_claim, partial_claim, failed_claim], semantic_verifier=None)

    summary = summarize_soft_preference_evidence([evidence])

    assert summary["retrieval_status_counts"]["success"] == 1
    assert summary["retrieval_status_counts"]["partial"] == 1
    assert summary["retrieval_status_counts"]["failed"] == 1
