from __future__ import annotations

import asyncio

import pytest

from app.logic import soft_evidence_orchestration as orch
from app.logic.soft_evidence_cleanup import DedupedEvidence
from app.logic.soft_evidence_retrieval import EmbedBatchOutcome
from app.logic.soft_preference_decomposition import AtomicClaimAssignment, CLAIM_HYPOTHESES, PreferenceDirection
from app.observability.trace import RequestTrace
from app.schemas.listing import ListingRaw
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence import (
    EvidenceRelation,
    EvidenceResolutionStatus,
    RetrievalStatus,
)
from app.schemas.soft_evidence_pipeline_policy import SoftEvidencePipelinePolicy
from app.logic.semantic_evidence_verification import SemanticVerificationResult


def _ok(latency_ms=10.0, batch_size=1) -> EmbedBatchOutcome:
    return EmbedBatchOutcome(success=True, batch_size=batch_size, latency_ms=latency_ms, error=None, token_count=None, estimated_cost_usd=None)


def _fail(error="boom") -> EmbedBatchOutcome:
    return EmbedBatchOutcome(success=False, batch_size=1, latency_ms=5.0, error=error, token_count=None, estimated_cost_usd=None)


def _candidate(text: str, score: float, source_type="description") -> DedupedEvidence:
    return DedupedEvidence(text=text, source_type=source_type, source_path=None, retrieval_score=score)


def _resolved(text="ok", reason="ok") -> SemanticVerificationResult:
    return SemanticVerificationResult(
        relation=EvidenceRelation.SUPPORT, reason=reason, status=EvidenceResolutionStatus.RESOLVED,
        latency_ms=10.0, prompt_tokens=50, completion_tokens=5, total_tokens=55, estimated_cost_usd=0.0001,
    )


def _assignment(claim_id: str) -> AtomicClaimAssignment:
    return AtomicClaimAssignment(
        claim_id=claim_id,
        hypothesis=CLAIM_HYPOTHESES[claim_id],
        family="quiet",
        source_constraint_id="c1",
        source_constraint_text="quiet please",
        preference_direction=PreferenceDirection.DESIRED,
    )


# ---------------- compute_pool_retrieval_status ----------------


def test_pool_status_empty_is_success():
    status, errors = orch.compute_pool_retrieval_status([])
    assert status == RetrievalStatus.SUCCESS
    assert errors == []


def test_pool_status_all_success():
    status, errors = orch.compute_pool_retrieval_status([_ok(), _ok()])
    assert status == RetrievalStatus.SUCCESS
    assert errors == []


def test_pool_status_mixed_is_partial():
    status, errors = orch.compute_pool_retrieval_status([_ok(), _fail("batch 2 timed out")])
    assert status == RetrievalStatus.PARTIAL
    assert len(errors) == 1
    assert "batch 2 timed out" in errors[0]


def test_pool_status_all_failed():
    status, errors = orch.compute_pool_retrieval_status([_fail("a"), _fail("b")])
    assert status == RetrievalStatus.FAILED
    assert len(errors) == 2


# ---------------- compute_claim_retrieval_status ----------------


def test_claim_status_query_failure_overrides_pool_status():
    status, errors = orch.compute_claim_retrieval_status(
        query_outcome=_fail("query down"), pool_status=RetrievalStatus.SUCCESS, pool_errors=[],
    )
    assert status == RetrievalStatus.FAILED
    assert "query embedding failed" in errors[0]


def test_claim_status_query_success_uses_pool_status():
    status, errors = orch.compute_claim_retrieval_status(
        query_outcome=_ok(), pool_status=RetrievalStatus.PARTIAL, pool_errors=["x"],
    )
    assert status == RetrievalStatus.PARTIAL
    assert errors == ["x"]


# ---------------- effective_per_hotel_budget ----------------


def test_effective_budget_uses_base_when_active_claims_fewer():
    assert orch.effective_per_hotel_budget(base=6, n_active_claims=2) == 6


def test_effective_budget_widens_for_more_active_claims():
    assert orch.effective_per_hotel_budget(base=6, n_active_claims=9) == 9


def test_effective_budget_capped_at_max_claims():
    assert orch.effective_per_hotel_budget(base=6, n_active_claims=999) == orch.MAX_CLAIMS


# ==================================================================
# _plan_verifier_tasks: pure, deterministic planning (no async at all)
# ==================================================================


def _plan(free_text_candidates, *, canonical_claim_order, retrieval_top_k, policy):
    return orch._plan_verifier_tasks(
        free_text_candidates=free_text_candidates,
        claim_hypotheses=CLAIM_HYPOTHESES,
        canonical_claim_order=canonical_claim_order,
        retrieval_top_k=retrieval_top_k,
        verifier_policy=policy,
    )


def test_plan_rank0_covered_across_all_claims_and_hotels_before_rank1():
    free_text_candidates = {
        0: {"Q1": [_candidate("h0-q1-r0", 0.9), _candidate("h0-q1-r1", 0.5)],
            "BC1": [_candidate("h0-bc1-r0", 0.9), _candidate("h0-bc1-r1", 0.5)]},
        1: {"Q1": [_candidate("h1-q1-r0", 0.9), _candidate("h1-q1-r1", 0.5)],
            "BC1": [_candidate("h1-bc1-r0", 0.9), _candidate("h1-bc1-r1", 0.5)]},
    }
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60)
    _, approved, _ = _plan(free_text_candidates, canonical_claim_order=("BC1", "Q1"), retrieval_top_k=2, policy=policy)

    texts = [t.candidate.text for t in approved]
    r0_positions = [i for i, t in enumerate(texts) if t.endswith("-r0")]
    r1_positions = [i for i, t in enumerate(texts) if t.endswith("-r1")]
    assert max(r0_positions) < min(r1_positions)
    assert len(texts) == 8


def test_plan_claim_major_order_within_a_rank():
    """Within rank 0: claim BC1 across all hotels, THEN claim Q1 across
    all hotels - not hotel-major."""
    free_text_candidates = {
        0: {"Q1": [_candidate("h0-q1", 0.9)], "BC1": [_candidate("h0-bc1", 0.9)]},
        1: {"Q1": [_candidate("h1-q1", 0.9)], "BC1": [_candidate("h1-bc1", 0.9)]},
    }
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60)
    _, approved, _ = _plan(free_text_candidates, canonical_claim_order=("BC1", "Q1"), retrieval_top_k=1, policy=policy)

    assert [t.candidate.text for t in approved] == ["h0-bc1", "h1-bc1", "h0-q1", "h1-q1"]


def test_plan_no_starvation_when_one_claim_has_many_candidates():
    free_text_candidates = {
        0: {"Q1": [_candidate("h0-q1-r0", 0.9), _candidate("h0-q1-r1", 0.8), _candidate("h0-q1-r2", 0.7)], "BC1": []},
        1: {"Q1": [], "BC1": [_candidate("h1-bc1-r0", 0.9)]},
    }
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60)
    slots, approved, positions = _plan(free_text_candidates, canonical_claim_order=("BC1", "Q1"), retrieval_top_k=3, policy=policy)

    assert ("h1-bc1-r0" in [t.candidate.text for t in approved])
    # hotel 1's BC1 rank-0 candidate got a real slot placeholder, not skipped
    assert slots[1]["BC1"][0] is None  # placeholder awaiting execution - proves it was approved, not SKIPPED_CALL_LIMIT


def test_plan_per_hotel_cap_marks_excess_as_skipped():
    free_text_candidates = {0: {"Q1": [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=2, max_calls_per_request=60)
    slots, approved, positions = _plan(free_text_candidates, canonical_claim_order=("Q1",), retrieval_top_k=3, policy=policy)

    assert len(approved) == 2
    assert [t.candidate.text for t in approved] == ["a", "b"]
    assert slots[0]["Q1"][2].resolution_status == EvidenceResolutionStatus.SKIPPED_CALL_LIMIT
    assert slots[0]["Q1"][2].evidence_text == "c"
    assert slots[0]["Q1"][2].relation is None


def test_plan_request_cap_applies_across_hotels():
    free_text_candidates = {
        0: {"Q1": [_candidate("h0", 0.9)]},
        1: {"Q1": [_candidate("h1", 0.9)]},
        2: {"Q1": [_candidate("h2", 0.9)]},
    }
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=2)
    _, approved, _ = _plan(free_text_candidates, canonical_claim_order=("Q1",), retrieval_top_k=1, policy=policy)
    assert len(approved) == 2


# ---------------- A: planner equivalence regardless of concurrency ----------------


def test_planner_output_identical_across_repeated_calls_concurrency_is_not_even_a_parameter():
    """
    _plan_verifier_tasks has NO concurrency parameter at all - it
    cannot possibly be affected by concurrency=1 vs concurrency=3,
    since concurrency only exists at the execution step, strictly
    after planning has already fully decided every task. This test
    proves the planning function is a pure, repeatable function of its
    (candidates, order, budgets) inputs alone.
    """
    free_text_candidates = {
        0: {"Q1": [_candidate("a", 0.9), _candidate("b", 0.8)], "BC1": [_candidate("c", 0.7)]},
        1: {"Q1": [_candidate("d", 0.9)]},
    }
    policy = SemanticVerifierPolicy(max_calls_per_hotel=2, max_calls_per_request=3)

    _, approved1, positions1 = _plan(free_text_candidates, canonical_claim_order=("BC1", "Q1"), retrieval_top_k=2, policy=policy)
    _, approved2, positions2 = _plan(free_text_candidates, canonical_claim_order=("BC1", "Q1"), retrieval_top_k=2, policy=policy)

    assert [t.candidate.text for t in approved1] == [t.candidate.text for t in approved2]
    assert positions1 == positions2


async def test_end_to_end_same_tasks_selected_at_concurrency_1_and_3(monkeypatch):
    """
    Full schedule_gemini_verification run twice, differing only in
    semantic_verifier_max_concurrency - the set of RESOLVED vs
    SKIPPED_CALL_LIMIT candidates must be identical.
    """
    monkeypatch.setattr(orch, "verify_evidence_relation", lambda *a, **k: _resolved_coro())

    free_text_candidates = {0: {"Q1": [_candidate(f"c{i}", 0.9 - i * 0.01) for i in range(5)]}}

    outcomes = {}
    for concurrency in (1, 3):
        policy = SemanticVerifierPolicy(max_calls_per_hotel=3, max_calls_per_request=3, semantic_verifier_max_concurrency=concurrency)
        gemini_items, _ = await orch.schedule_gemini_verification(
            free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
            retrieval_top_k=5, verifier_policy=policy, trace=None,
        )
        outcomes[concurrency] = [(item.evidence_text, item.resolution_status.value) for item in gemini_items[0]["Q1"]]

    assert outcomes[1] == outcomes[3]
    assert [s for _, s in outcomes[1] if s == "skipped_call_limit"] == ["skipped_call_limit"] * 2


async def _resolved_coro():
    return _resolved()


# ---------------- B: result equivalence under out-of-order completion ----------------


async def test_out_of_order_completion_still_produces_deterministic_final_order(monkeypatch):
    """Later-submitted candidates finish FIRST (reversed delay) - final
    evidence_items order must still match the planned (rank) order,
    not completion order."""
    delays = {"first": 0.03, "second": 0.015, "third": 0.0}

    async def _fake_verify(evidence_text, hypothesis, *, policy, trace):
        await asyncio.sleep(delays[evidence_text])
        return SemanticVerificationResult(
            relation=EvidenceRelation.SUPPORT, reason=f"ok-{evidence_text}", status=EvidenceResolutionStatus.RESOLVED,
            latency_ms=1.0,
        )

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    free_text_candidates = {0: {"Q1": [_candidate("first", 0.9), _candidate("second", 0.8), _candidate("third", 0.7)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60, semantic_verifier_max_concurrency=3)

    gemini_items, _ = await orch.schedule_gemini_verification(
        free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
        retrieval_top_k=3, verifier_policy=policy, trace=None,
    )

    items = gemini_items[0]["Q1"]
    assert [i.evidence_text for i in items] == ["first", "second", "third"]
    assert [i.verifier_reason for i in items] == ["ok-first", "ok-second", "ok-third"]


# ---------------- C: bounded concurrency (both executors) ----------------


async def test_verifier_executor_never_exceeds_configured_concurrency(monkeypatch):
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def _fake_verify(evidence_text, hypothesis, *, policy, trace):
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return _resolved()

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    free_text_candidates = {0: {"Q1": [_candidate(f"c{i}", 0.9 - i * 0.001) for i in range(9)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=9, max_calls_per_request=9, semantic_verifier_max_concurrency=3)

    await orch.schedule_gemini_verification(
        free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
        retrieval_top_k=9, verifier_policy=policy, trace=None,
    )
    assert peak <= 3
    assert peak == 3


async def test_embedding_executor_never_exceeds_configured_concurrency(monkeypatch):
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def _tracked_embed_batch_async(texts, *, model, trace, step):
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return [[1.0] for _ in texts], _ok()

    monkeypatch.setattr(orch, "_embed_batch_async", _tracked_embed_batch_async)

    # 3 hotels, each with a pool large enough to require multiple batches
    listings = [
        ListingRaw(id=f"h{i}", description=" ".join(f"Sentence number {j}." for j in range(250)))
        for i in range(3)
    ]
    await orch._run_embedding_phase(
        listings=listings,
        claim_ids=["Q1"],
        pipeline_policy=SoftEvidencePipelinePolicy(embedding_max_concurrency=2, retrieval_top_k=1),
        trace=None,
    )
    assert peak <= 2
    assert peak == 2


# ---------------- D: failure isolation ----------------


async def test_one_concurrent_verifier_failure_does_not_prevent_siblings(monkeypatch):
    async def _fake_verify(evidence_text, hypothesis, *, policy, trace):
        if evidence_text == "bad":
            return SemanticVerificationResult(
                relation=None, reason=None, status=EvidenceResolutionStatus.VERIFICATION_FAILED,
                error="TimeoutError: boom", latency_ms=2.0,
            )
        return _resolved()

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    free_text_candidates = {0: {"Q1": [_candidate("good1", 0.9), _candidate("bad", 0.8), _candidate("good2", 0.7)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60, semantic_verifier_max_concurrency=3)

    gemini_items, usage = await orch.schedule_gemini_verification(
        free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
        retrieval_top_k=3, verifier_policy=policy, trace=None,
    )
    items = gemini_items[0]["Q1"]
    assert items[0].resolution_status == EvidenceResolutionStatus.RESOLVED
    assert items[1].resolution_status == EvidenceResolutionStatus.VERIFICATION_FAILED
    assert items[1].error == "TimeoutError: boom"
    assert items[2].resolution_status == EvidenceResolutionStatus.RESOLVED
    assert usage[0]["calls"] == 3  # all three still attempted/counted


async def test_unexpected_exception_in_one_task_still_lets_siblings_finish_then_propagates(monkeypatch):
    """
    A genuine programming bug (not an expected operational failure -
    those never raise, see verify_evidence_relation) in one task must
    not silently vanish, but must also not orphan sibling tasks.
    """
    completed = []

    async def _fake_verify(evidence_text, hypothesis, *, policy, trace):
        if evidence_text == "buggy":
            raise RuntimeError("unexpected bug")
        await asyncio.sleep(0.01)
        completed.append(evidence_text)
        return _resolved()

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    free_text_candidates = {0: {"Q1": [_candidate("ok1", 0.9), _candidate("buggy", 0.8), _candidate("ok2", 0.7)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60, semantic_verifier_max_concurrency=3)

    with pytest.raises(RuntimeError, match="unexpected bug"):
        await orch.schedule_gemini_verification(
            free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
            retrieval_top_k=3, verifier_policy=policy, trace=None,
        )

    assert "ok1" in completed
    assert "ok2" in completed


# ---------------- E: budget preservation under concurrency ----------------


async def test_skipped_candidates_identical_at_concurrency_1_and_3(monkeypatch):
    monkeypatch.setattr(orch, "verify_evidence_relation", lambda *a, **k: _resolved_coro())

    free_text_candidates = {0: {"Q1": [_candidate(f"c{i}", 0.9 - i * 0.01) for i in range(5)]}}

    outcomes = {}
    for concurrency in (1, 3):
        policy = SemanticVerifierPolicy(max_calls_per_hotel=3, max_calls_per_request=3, semantic_verifier_max_concurrency=concurrency)
        gemini_items, _ = await orch.schedule_gemini_verification(
            free_text_candidates=free_text_candidates, canonical_claim_order=("Q1",),
            retrieval_top_k=5, verifier_policy=policy, trace=None,
        )
        outcomes[concurrency] = [(i.evidence_text, i.resolution_status.value) for i in gemini_items[0]["Q1"]]

    assert outcomes[1] == outcomes[3]
    skipped = [t for t, s in outcomes[1] if s == "skipped_call_limit"]
    assert skipped == ["c3", "c4"]


# ---------------- F: retrieval status under parallel embedding failures ----------------


async def test_partial_pool_failure_under_concurrency_produces_partial_status(monkeypatch):
    call_count = {"n": 0}

    async def _mock_embed_batch_async(texts, *, model, trace, step):
        call_count["n"] += 1
        if step == orch.EMBEDDING_STEP_QUERY:
            return [[1.0] for _ in texts], _ok()
        # fail every other pool batch
        if call_count["n"] % 2 == 0:
            return None, _fail("simulated batch failure")
        return [[1.0] for _ in texts], _ok()

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)

    listing = ListingRaw(id="h1", description=" ".join(f"Sentence number {i} here." for i in range(250)))
    _, _, _, _, pool_outcomes = await orch._run_embedding_phase(
        listings=[listing], claim_ids=["Q1"],
        pipeline_policy=SoftEvidencePipelinePolicy(embedding_max_concurrency=3),
        trace=None,
    )
    status, errors = orch.compute_pool_retrieval_status(pool_outcomes[0])
    assert status == RetrievalStatus.PARTIAL
    assert errors


async def test_all_pool_batches_failed_under_concurrency_produces_failed_status(monkeypatch):
    async def _mock_embed_batch_async(texts, *, model, trace, step):
        if step == orch.EMBEDDING_STEP_QUERY:
            return [[1.0] for _ in texts], _ok()
        return None, _fail("simulated batch failure")

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)

    listing = ListingRaw(id="h1", description=" ".join(f"Sentence number {i} here." for i in range(150)))
    _, _, _, _, pool_outcomes = await orch._run_embedding_phase(
        listings=[listing], claim_ids=["Q1"],
        pipeline_policy=SoftEvidencePipelinePolicy(embedding_max_concurrency=3),
        trace=None,
    )
    status, errors = orch.compute_pool_retrieval_status(pool_outcomes[0])
    assert status == RetrievalStatus.FAILED
    assert errors


async def test_verifier_usage_accumulated_per_hotel(monkeypatch):
    async def _fake_verify(evidence_text, hypothesis, *, policy, trace):
        return SemanticVerificationResult(
            relation=EvidenceRelation.SUPPORT, reason="ok", status=EvidenceResolutionStatus.RESOLVED,
            latency_ms=50.0, prompt_tokens=100, completion_tokens=10, total_tokens=110, estimated_cost_usd=0.001,
        )

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    trace = RequestTrace()
    free_text_candidates = {0: {"Q1": [_candidate("a", 0.9)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60)
    _, usage = await orch.schedule_gemini_verification(
        free_text_candidates=free_text_candidates,
        canonical_claim_order=("Q1",),
        retrieval_top_k=1,
        verifier_policy=policy,
        trace=trace,
    )
    assert usage[0]["calls"] == 1
    assert usage[0]["latency_ms"] == 50.0
    assert usage[0]["total_tokens"] == 110
    assert usage[0]["estimated_cost_usd"] == pytest.approx(0.001)


async def test_verifier_usage_cost_unknown_when_no_call_ever_had_a_response(monkeypatch):
    """
    If every Gemini call for a hotel fails before any response object
    exists, SemanticVerificationResult.estimated_cost_usd is None
    (genuinely unknown) - the accumulator's cost_known flag must stay
    False so the final SemanticVerifierUsage.estimated_cost_usd is
    None, never a fabricated 0.0.
    """
    async def _fake_verify_no_response(evidence_text, hypothesis, *, policy, trace):
        return SemanticVerificationResult(
            relation=None, reason=None, status=EvidenceResolutionStatus.VERIFICATION_FAILED,
            error="TimeoutError: no response", latency_ms=30.0,
            prompt_tokens=None, completion_tokens=None, total_tokens=None, estimated_cost_usd=None,
        )

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify_no_response)

    trace = RequestTrace()
    free_text_candidates = {0: {"Q1": [_candidate("a", 0.9), _candidate("b", 0.8)]}}
    policy = SemanticVerifierPolicy(max_calls_per_hotel=6, max_calls_per_request=60)
    _, usage = await orch.schedule_gemini_verification(
        free_text_candidates=free_text_candidates,
        canonical_claim_order=("Q1",),
        retrieval_top_k=2,
        verifier_policy=policy,
        trace=trace,
    )
    assert usage[0]["calls"] == 2
    assert usage[0]["cost_known"] is False
    assert usage[0]["estimated_cost_usd"] == 0.0  # raw accumulator, not yet gated


async def test_build_shadow_evidence_reports_none_cost_when_all_calls_failed_without_response(monkeypatch):
    async def _mock_embed_batch_async(texts, *, model, trace, step):
        return [[1.0] for _ in texts], _ok()

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)
    monkeypatch.setattr(
        orch, "retrieve_top_k",
        lambda query_vector, pool, vectors, k: [_candidate("A very quiet street with little traffic noise.", 0.9)],
    )

    async def _fake_verify_no_response(evidence_text, hypothesis, *, policy, trace):
        return SemanticVerificationResult(
            relation=None, reason=None, status=EvidenceResolutionStatus.VERIFICATION_FAILED,
            error="TimeoutError: no response", latency_ms=30.0,
            prompt_tokens=None, completion_tokens=None, total_tokens=None, estimated_cost_usd=None,
        )

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify_no_response)

    results = await orch.build_shadow_soft_preference_evidence(
        listings=[ListingRaw(id="h1", description="A very quiet street with little traffic noise.")],
        claim_assignments=[_assignment("Q1")],
        pipeline_policy=SoftEvidencePipelinePolicy(),
        verifier_policy=SemanticVerifierPolicy(),
        trace=RequestTrace(),
    )

    evidence = results[0]
    assert evidence.semantic_verifier is not None
    assert evidence.semantic_verifier.calls == 1
    assert evidence.semantic_verifier.estimated_cost_usd is None


# ---------------- build_shadow_soft_preference_evidence: end-to-end ----------------


async def test_query_embedding_failure_makes_every_claim_failed_with_no_evidence(monkeypatch):
    async def _mock_embed_batch_async(texts, *, model, trace, step):
        assert step == orch.EMBEDDING_STEP_QUERY  # empty pools -> only the query batch is ever planned
        return None, _fail("query api down")

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)

    listings = [ListingRaw(id="h1"), ListingRaw(id="h2")]
    assignments = [_assignment("Q1")]

    results = await orch.build_shadow_soft_preference_evidence(
        listings=listings,
        claim_assignments=assignments,
        pipeline_policy=SoftEvidencePipelinePolicy(),
        verifier_policy=SemanticVerifierPolicy(),
        trace=None,
    )

    assert len(results) == 2
    for evidence in results:
        assert len(evidence.claims) == 1
        claim = evidence.claims[0]
        assert claim.claim_id == "Q1"
        assert claim.retrieval_status == RetrievalStatus.FAILED
        assert claim.retrieval_errors
        assert claim.evidence_items == []
        assert claim.relation.value == "NOT_ENOUGH_EVIDENCE"
        assert evidence.semantic_verifier is None  # no Gemini calls were ever attempted


async def test_deterministic_only_resolution_leaves_semantic_verifier_none(monkeypatch):
    async def _mock_embed_batch_async(texts, *, model, trace, step):
        assert step == orch.EMBEDDING_STEP_QUERY  # empty pool (no name/description/etc set)
        return [[1.0] for _ in texts], _ok()

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)
    monkeypatch.setattr(orch, "retrieve_top_k", lambda query_vector, pool, vectors, k: _fixed_retrieval(pool))

    listings = [ListingRaw(id="h1", rooms=[])]
    assignments = [_assignment("RW1")]

    async def _fake_verify(*args, **kwargs):
        raise AssertionError("Gemini should never be called for a fully deterministic candidate set")

    monkeypatch.setattr(orch, "verify_evidence_relation", _fake_verify)

    results = await orch.build_shadow_soft_preference_evidence(
        listings=listings,
        claim_assignments=assignments,
        pipeline_policy=SoftEvidencePipelinePolicy(),
        verifier_policy=SemanticVerifierPolicy(),
        trace=None,
    )

    evidence = results[0]
    claim = evidence.claims[0]
    assert claim.relation.value == "SUPPORT"
    assert claim.evidence_items[0].resolution_method.value == "deterministic"
    assert evidence.semantic_verifier is None


def _fixed_retrieval(pool):
    from app.logic.soft_evidence_retrieval import RetrievedEvidence
    return [
        RetrievedEvidence(text="Desk", source_type="room_facilities", source_path="rooms[0].facilities[0].name", retrieval_score=0.99)
    ]


async def test_empty_claim_assignments_produces_empty_claims_no_calls(monkeypatch):
    called = {"n": 0}

    async def _mock_embed_batch_async(texts, *, model, trace, step):
        called["n"] += 1
        return [[1.0] for _ in texts], _ok()

    monkeypatch.setattr(orch, "_embed_batch_async", _mock_embed_batch_async)

    results = await orch.build_shadow_soft_preference_evidence(
        listings=[ListingRaw(id="h1")],
        claim_assignments=[],
        pipeline_policy=SoftEvidencePipelinePolicy(),
        verifier_policy=SemanticVerifierPolicy(),
        trace=None,
    )
    assert called["n"] == 0  # no queries, no pool - nothing to embed at all
    assert results[0].claims == []
    assert results[0].semantic_verifier is None
