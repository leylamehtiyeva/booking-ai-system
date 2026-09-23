"""
Per-request aggregates over the new soft-preference evidence pipeline.

Pure function over app.schemas.soft_evidence data - Phase A only,
nothing in the live request path calls this yet. Phase B's shadow-mode
integration is expected to call this once per request (over the
SoftPreferenceEvidence computed for each ranked hotel) and attach the
result to telemetry, alongside (not replacing) RequestTrace.summary().
"""

from __future__ import annotations

from typing import Any

from app.schemas.soft_evidence import (
    ClaimRelation,
    ClaimResolutionMethod,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SoftPreferenceEvidence,
)


def summarize_soft_preference_evidence(
    evidences: list[SoftPreferenceEvidence],
) -> dict[str, Any]:
    total_hotels = len(evidences)

    claim_relation_counts = {relation.value: 0 for relation in ClaimRelation}
    evidence_resolution_status_counts = {
        status.value: 0 for status in EvidenceResolutionStatus
    }
    retrieval_status_counts = {status.value: 0 for status in RetrievalStatus}

    # Counted per EvidenceItem, not per claim: a single claim can hold a
    # mix of deterministic and Gemini-sourced items, so "number of claims
    # resolved deterministically" would need an arbitrary tie-break for
    # mixed claims. Evidence-item-level counts avoid that ambiguity.
    deterministic_evidence_item_count = 0
    gemini_evidence_item_count = 0

    hotels_requiring_gemini = 0

    total_verifier_calls = 0
    total_verifier_latency_ms = 0.0
    total_verifier_input_tokens = 0
    total_verifier_output_tokens = 0
    total_verifier_total_tokens = 0
    total_verifier_cost_usd = 0.0
    total_verifier_parse_failures = 0

    for evidence in evidences:
        hotel_used_gemini = False

        for claim in evidence.claims:
            claim_relation_counts[claim.relation.value] += 1
            retrieval_status_counts[claim.retrieval_status.value] += 1

            for item in claim.evidence_items:
                evidence_resolution_status_counts[item.resolution_status.value] += 1

                if item.resolution_method == ClaimResolutionMethod.DETERMINISTIC:
                    deterministic_evidence_item_count += 1
                elif item.resolution_method == ClaimResolutionMethod.GEMINI:
                    gemini_evidence_item_count += 1
                    hotel_used_gemini = True

        if hotel_used_gemini:
            hotels_requiring_gemini += 1

        usage = evidence.semantic_verifier
        if usage is not None:
            total_verifier_calls += usage.calls
            total_verifier_latency_ms += usage.latency_ms
            total_verifier_input_tokens += usage.input_tokens
            total_verifier_output_tokens += usage.output_tokens
            total_verifier_total_tokens += usage.total_tokens
            total_verifier_cost_usd += usage.estimated_cost_usd or 0.0
            total_verifier_parse_failures += usage.parse_failures

    hotels_requiring_gemini_pct = (
        round(hotels_requiring_gemini / total_hotels * 100, 2)
        if total_hotels
        else 0.0
    )

    return {
        "total_hotels": total_hotels,
        "claim_relation_counts": claim_relation_counts,
        "evidence_resolution_status_counts": evidence_resolution_status_counts,
        "retrieval_status_counts": retrieval_status_counts,
        "deterministic_evidence_item_count": deterministic_evidence_item_count,
        "gemini_evidence_item_count": gemini_evidence_item_count,
        "hotels_requiring_gemini": hotels_requiring_gemini,
        "hotels_requiring_gemini_pct": hotels_requiring_gemini_pct,
        "verifier": {
            "total_calls": total_verifier_calls,
            "total_latency_ms": round(total_verifier_latency_ms, 2),
            "total_input_tokens": total_verifier_input_tokens,
            "total_output_tokens": total_verifier_output_tokens,
            "total_tokens": total_verifier_total_tokens,
            "total_cost_usd": round(total_verifier_cost_usd, 6),
            "total_parse_failures": total_verifier_parse_failures,
        },
    }
