from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimRelation,
    ClaimResolutionMethod,
    EvidenceItem,
    EvidenceRelation,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SoftPreferenceEvidence,
    aggregate_claim_relation,
)


def _resolved(relation: EvidenceRelation, **kwargs) -> EvidenceItem:
    return EvidenceItem(
        evidence_text="some evidence",
        relation=relation,
        resolution_method=kwargs.pop("resolution_method", ClaimResolutionMethod.GEMINI),
        resolution_status=EvidenceResolutionStatus.RESOLVED,
        **kwargs,
    )


def _failed(status: EvidenceResolutionStatus, error: str = "boom") -> EvidenceItem:
    return EvidenceItem(
        evidence_text="some evidence",
        relation=None,
        resolution_method=ClaimResolutionMethod.GEMINI,
        resolution_status=status,
        error=error,
    )


# ---------------- EvidenceItem invariant ----------------


def test_evidence_item_resolved_requires_relation():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_text="x",
            relation=None,
            resolution_method=ClaimResolutionMethod.GEMINI,
            resolution_status=EvidenceResolutionStatus.RESOLVED,
        )


def test_evidence_item_unresolved_forbids_relation():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_text="x",
            relation=EvidenceRelation.SUPPORT,
            resolution_method=ClaimResolutionMethod.GEMINI,
            resolution_status=EvidenceResolutionStatus.VERIFICATION_FAILED,
        )


def test_evidence_item_verification_failed_with_no_relation_is_valid():
    item = EvidenceItem(
        evidence_text="x",
        relation=None,
        resolution_method=ClaimResolutionMethod.GEMINI,
        resolution_status=EvidenceResolutionStatus.VERIFICATION_FAILED,
        error="TimeoutError: took too long",
    )
    assert item.relation is None
    assert item.error == "TimeoutError: took too long"


def test_evidence_item_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_text="x",
            relation=EvidenceRelation.SUPPORT,
            resolution_method=ClaimResolutionMethod.DETERMINISTIC,
            resolution_status=EvidenceResolutionStatus.RESOLVED,
            unexpected_field="nope",
        )


# ---------------- aggregate_claim_relation ----------------


def test_aggregate_empty_evidence_is_not_enough_evidence():
    assert aggregate_claim_relation([]) == ClaimRelation.NOT_ENOUGH_EVIDENCE


def test_aggregate_only_support():
    items = [_resolved(EvidenceRelation.SUPPORT)]
    assert aggregate_claim_relation(items) == ClaimRelation.SUPPORT


def test_aggregate_only_contradict():
    items = [_resolved(EvidenceRelation.CONTRADICT)]
    assert aggregate_claim_relation(items) == ClaimRelation.CONTRADICT


def test_aggregate_support_and_contradict_is_mixed():
    items = [
        _resolved(EvidenceRelation.SUPPORT),
        _resolved(EvidenceRelation.CONTRADICT),
    ]
    assert aggregate_claim_relation(items) == ClaimRelation.MIXED


def test_aggregate_contradict_and_not_enough_evidence_stays_contradict():
    items = [
        _resolved(EvidenceRelation.CONTRADICT),
        _resolved(EvidenceRelation.NOT_ENOUGH_EVIDENCE),
    ]
    assert aggregate_claim_relation(items) == ClaimRelation.CONTRADICT


def test_aggregate_only_not_enough_evidence():
    items = [
        _resolved(EvidenceRelation.NOT_ENOUGH_EVIDENCE),
        _resolved(EvidenceRelation.NOT_ENOUGH_EVIDENCE),
    ]
    assert aggregate_claim_relation(items) == ClaimRelation.NOT_ENOUGH_EVIDENCE


def test_aggregate_all_failed_or_skipped_is_not_enough_evidence():
    items = [
        _failed(EvidenceResolutionStatus.VERIFICATION_FAILED),
        _failed(EvidenceResolutionStatus.SKIPPED_CALL_LIMIT),
    ]
    assert aggregate_claim_relation(items) == ClaimRelation.NOT_ENOUGH_EVIDENCE


def test_aggregate_ignores_failed_item_but_still_uses_resolved_support():
    """
    The key preservation guarantee: a failed evidence item must not be
    dropped from evidence_items, but must also not affect the aggregated
    relation - only RESOLVED items vote.
    """
    resolved_support = _resolved(EvidenceRelation.SUPPORT)
    failed = _failed(EvidenceResolutionStatus.VERIFICATION_FAILED)

    claim = AtomicClaimResult.from_evidence_items(
        claim_id="RW2B",
        hypothesis="Internet connection is reliable enough for remote work.",
        evidence_items=[resolved_support, failed],
    )

    assert claim.relation == ClaimRelation.SUPPORT
    assert len(claim.evidence_items) == 2
    assert failed in claim.evidence_items


# ---------------- AtomicClaimResult.from_evidence_items ----------------


def test_from_evidence_items_no_evidence():
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="BC1",
        hypothesis="The bed is comfortable.",
        evidence_items=[],
    )
    assert claim.evidence_items == []
    assert claim.relation == ClaimRelation.NOT_ENOUGH_EVIDENCE


# ---------------- SoftPreferenceEvidence ----------------


def test_soft_preference_evidence_defaults_to_null_verifier_usage():
    evidence = SoftPreferenceEvidence()
    assert evidence.claims == []
    assert evidence.semantic_verifier is None


def test_soft_preference_evidence_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        SoftPreferenceEvidence(claims=[], unexpected="nope")


# ---------------- RetrievalStatus (Phase B) ----------------


def test_atomic_claim_result_defaults_to_retrieval_success():
    claim = AtomicClaimResult.from_evidence_items(claim_id="Q1", hypothesis="h", evidence_items=[])
    assert claim.retrieval_status == RetrievalStatus.SUCCESS
    assert claim.retrieval_errors == []
    assert claim.retrieved_candidate_count == 0


def test_retrieval_success_forbids_errors():
    with pytest.raises(ValidationError):
        AtomicClaimResult(
            claim_id="Q1", hypothesis="h", relation=ClaimRelation.NOT_ENOUGH_EVIDENCE,
            evidence_items=[], retrieval_status=RetrievalStatus.SUCCESS,
            retrieval_errors=["should not be here"],
        )


def test_retrieval_partial_requires_errors():
    with pytest.raises(ValidationError):
        AtomicClaimResult(
            claim_id="Q1", hypothesis="h", relation=ClaimRelation.NOT_ENOUGH_EVIDENCE,
            evidence_items=[], retrieval_status=RetrievalStatus.PARTIAL, retrieval_errors=[],
        )


def test_retrieval_failed_requires_errors():
    with pytest.raises(ValidationError):
        AtomicClaimResult(
            claim_id="Q1", hypothesis="h", relation=ClaimRelation.NOT_ENOUGH_EVIDENCE,
            evidence_items=[], retrieval_status=RetrievalStatus.FAILED, retrieval_errors=[],
        )


def test_retrieval_partial_with_errors_and_real_evidence_items_is_valid():
    """
    A technical retrieval failure and "genuinely no evidence" both
    still aggregate to NOT_ENOUGH_EVIDENCE when there's nothing
    resolved, but retrieval_status stays distinct/visible.
    """
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="CL1", hypothesis="h", evidence_items=[],
        retrieval_status=RetrievalStatus.FAILED,
        retrieval_errors=["query embedding failed: TimeoutError: ..."],
        retrieved_candidate_count=0,
    )
    assert claim.relation == ClaimRelation.NOT_ENOUGH_EVIDENCE
    assert claim.retrieval_status == RetrievalStatus.FAILED
    assert claim.retrieval_errors == ["query embedding failed: TimeoutError: ..."]


def test_retrieval_partial_can_still_carry_real_resolved_evidence():
    item = _resolved(EvidenceRelation.SUPPORT)
    claim = AtomicClaimResult.from_evidence_items(
        claim_id="CL1", hypothesis="h", evidence_items=[item],
        retrieval_status=RetrievalStatus.PARTIAL,
        retrieval_errors=["pool batch 2/3 failed: TimeoutError: ..."],
        retrieved_candidate_count=3,
    )
    assert claim.relation == ClaimRelation.SUPPORT
    assert claim.retrieval_status == RetrievalStatus.PARTIAL
    assert claim.retrieved_candidate_count == 3
