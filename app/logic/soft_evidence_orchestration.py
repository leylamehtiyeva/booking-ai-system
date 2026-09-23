"""
Full soft-evidence pipeline orchestration for one request's shadow-mode
scope: retrieval -> cleanup/routing -> fair Gemini scheduling -> claim
aggregation -> SoftPreferenceEvidence, one object per hotel.

Not wired into evaluate_listings() yet - see
app.logic.listing_evaluation for the (separate, later) integration
point.

Concurrency design (performance-only - does not change any decision):
planning (which embedding batches exist; which candidates get an
approved Gemini call vs. SKIPPED_CALL_LIMIT) is always fully decided
in plain, deterministic, synchronous Python, in a fixed order, BEFORE
any concurrent execution begins. Only the already-approved, already-
ordered external API calls are then executed concurrently (bounded by
a semaphore, via app.logic.soft_evidence_concurrency.run_bounded), and
results are restored into their pre-planned positions - never by
completion order. See _plan_verifier_tasks and _plan_embedding_tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.logic.semantic_evidence_verification import verify_evidence_relation
from app.logic.soft_evidence_cleanup import DedupedEvidence, clean_and_route_candidates
from app.logic.soft_evidence_collection import SoftEvidenceCandidate, collect_soft_evidence_pool
from app.logic.soft_evidence_concurrency import run_bounded
from app.logic.soft_evidence_retrieval import (
    EMBEDDING_STEP_POOL,
    EMBEDDING_STEP_QUERY,
    MAX_BATCH_SIZE,
    EmbedBatchOutcome,
    _embed_batch_async,
    retrieve_top_k,
)
from app.logic.soft_preference_decomposition import (
    CANONICAL_CLAIM_ORDER,
    CLAIM_HYPOTHESES,
    AtomicClaimAssignment,
    get_retrieval_query,
)
from app.observability.trace import RequestTrace
from app.schemas.listing import ListingRaw
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimResolutionMethod,
    EvidenceItem,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SemanticVerifierUsage,
    SoftPreferenceEvidence,
)
from app.schemas.soft_evidence_pipeline_policy import SoftEvidencePipelinePolicy

# Total number of validated Phase B claims - the upper bound used both
# by effective_per_hotel_budget() and by the request-budget floor.
MAX_CLAIMS = len(CLAIM_HYPOTHESES)


def compute_pool_retrieval_status(outcomes: list[EmbedBatchOutcome]) -> tuple[RetrievalStatus, list[str]]:
    """
    An empty pool (no batches at all) is SUCCESS with no errors - a
    hotel genuinely having no evidence text is not a technical failure.

    Depends only on the SET of outcomes (success/failure counts), not
    their order, so it is unaffected by concurrent completion order.
    The order of `outcomes` itself is still deterministic regardless
    (see _plan_embedding_tasks: results are restored in batch-start
    order, never completion order), so retrieval_errors' order is also
    unaffected by concurrency.
    """
    if not outcomes:
        return RetrievalStatus.SUCCESS, []

    failures = [o for o in outcomes if not o.success]
    if not failures:
        return RetrievalStatus.SUCCESS, []

    errors = [f"evidence pool batch failed: {o.error}" for o in failures]
    successes = [o for o in outcomes if o.success]
    if successes:
        return RetrievalStatus.PARTIAL, errors
    return RetrievalStatus.FAILED, errors


def compute_claim_retrieval_status(
    *,
    query_outcome: EmbedBatchOutcome,
    pool_status: RetrievalStatus,
    pool_errors: list[str],
) -> tuple[RetrievalStatus, list[str]]:
    """
    Query embedding is shared/request-level and all-or-nothing: if it
    failed, no claim on any hotel has a query vector to compare
    against anything, regardless of that hotel's own pool status.
    """
    if not query_outcome.success:
        return RetrievalStatus.FAILED, [f"query embedding failed: {query_outcome.error}"]
    return pool_status, pool_errors


def effective_per_hotel_budget(*, base: int, n_active_claims: int) -> int:
    """
    Do not let a fixed base budget silently starve active claims: a
    hotel with more genuinely active claims (needing Gemini) than the
    base allows gets a larger effective budget, capped at MAX_CLAIMS
    (there are only MAX_CLAIMS claims total, so more would be
    meaningless).
    """
    return min(MAX_CLAIMS, max(base, n_active_claims))


class _HotelClaimContext:
    """Internal, per-(hotel, claim) working state during orchestration."""

    __slots__ = ("deterministic_items", "free_text_candidates", "retrieval_status", "retrieval_errors", "retrieved_candidate_count")

    def __init__(self) -> None:
        self.deterministic_items: list[EvidenceItem] = []
        self.free_text_candidates: list[DedupedEvidence] = []
        self.retrieval_status: RetrievalStatus = RetrievalStatus.SUCCESS
        self.retrieval_errors: list[str] = []
        self.retrieved_candidate_count: int = 0


# ==================================================================
# Embedding phase: deterministic planning + concurrent execution
# ==================================================================


@dataclass(frozen=True)
class _EmbeddingTaskSpec:
    """One planned embedding batch call. kind is "query" or "pool";
    hotel_idx/batch_start are meaningless (-1/0) for the query task."""

    kind: str
    hotel_idx: int
    batch_start: int
    texts: list[str]
    step: str


def _plan_embedding_tasks(
    *,
    query_texts: list[str],
    pools: dict[int, list[SoftEvidenceCandidate]],
) -> list[_EmbeddingTaskSpec]:
    """
    Deterministic, synchronous, no I/O: decides exactly which
    embedding batch calls are needed - the shared query batch (if any
    queries exist) plus every hotel's evidence pool split into
    <=MAX_BATCH_SIZE batches, in a fixed order (query first, then
    hotel 0's batches in ascending start-index order, then hotel 1's,
    ...). This plan - which calls exist, their content, their count -
    is completely independent of concurrency; only how the resulting
    task list is later executed (sequential or bounded-concurrent) can
    vary.
    """
    specs: list[_EmbeddingTaskSpec] = []

    if query_texts:
        specs.append(_EmbeddingTaskSpec(kind="query", hotel_idx=-1, batch_start=0, texts=query_texts, step=EMBEDDING_STEP_QUERY))

    for hotel_idx in sorted(pools.keys()):
        pool = pools[hotel_idx]
        for start in range(0, len(pool), MAX_BATCH_SIZE):
            batch = pool[start : start + MAX_BATCH_SIZE]
            specs.append(
                _EmbeddingTaskSpec(
                    kind="pool", hotel_idx=hotel_idx, batch_start=start,
                    texts=[c.text for c in batch], step=EMBEDDING_STEP_POOL,
                )
            )

    return specs


async def _run_embedding_phase(
    *,
    listings: list[ListingRaw],
    claim_ids: list[str],
    pipeline_policy: SoftEvidencePipelinePolicy,
    trace: RequestTrace | None,
) -> tuple[
    dict[str, list[float]],
    EmbedBatchOutcome,
    dict[int, list[SoftEvidenceCandidate]],
    dict[int, list[list[float] | None]],
    dict[int, list[EmbedBatchOutcome]],
]:
    """
    Collects every hotel's evidence pool (pure, no I/O), plans the full
    flat list of embedding batch calls (_plan_embedding_tasks - the
    shared query batch and every hotel's pool batches, unchanged call
    count/content from the sequential design), then executes that
    whole plan through run_bounded(max_concurrency=
    pipeline_policy.embedding_max_concurrency) - one bounded pool
    shared across hotels, not a separate pool per hotel, so an idle
    concurrency slot from a finished batch is immediately available to
    any other hotel's pending batch.

    Results are restored into query_vectors_by_claim / per-hotel
    pool_vectors / pool_outcomes strictly by each task's planned
    (hotel_idx, batch_start) position - run_bounded already guarantees
    its return list is aligned with the input task order regardless of
    completion order, so this restoration is safe under any
    concurrency level.
    """
    pools: dict[int, list[SoftEvidenceCandidate]] = {
        hotel_idx: collect_soft_evidence_pool(listing) for hotel_idx, listing in enumerate(listings)
    }

    queries = [(claim_id, get_retrieval_query(claim_id)) for claim_id in claim_ids]
    query_texts = [q for _, q in queries]

    specs = _plan_embedding_tasks(query_texts=query_texts, pools=pools)

    query_vectors_by_claim: dict[str, list[float]] = {}
    query_outcome = EmbedBatchOutcome(
        success=True, batch_size=0, latency_ms=0.0, error=None, token_count=None, estimated_cost_usd=None
    )
    pool_vectors: dict[int, list[list[float] | None]] = {hotel_idx: [None] * len(pools[hotel_idx]) for hotel_idx in pools}
    pool_outcomes: dict[int, list[EmbedBatchOutcome]] = {hotel_idx: [] for hotel_idx in pools}

    if not specs:
        return query_vectors_by_claim, query_outcome, pools, pool_vectors, pool_outcomes

    coros = [
        _embed_batch_async(spec.texts, model=pipeline_policy.embedding_model, trace=trace, step=spec.step)
        for spec in specs
    ]
    raw_results = await run_bounded(coros, max_concurrency=pipeline_policy.normalized_embedding_max_concurrency())

    for spec, (vectors, outcome) in zip(specs, raw_results):
        if spec.kind == "query":
            query_outcome = outcome
            if vectors is not None:
                query_vectors_by_claim = {claim_id: vector for (claim_id, _), vector in zip(queries, vectors)}
        else:
            pool_outcomes[spec.hotel_idx].append(outcome)
            if vectors is not None:
                for offset, vector in enumerate(vectors):
                    pool_vectors[spec.hotel_idx][spec.batch_start + offset] = vector

    return query_vectors_by_claim, query_outcome, pools, pool_vectors, pool_outcomes


# ==================================================================
# Verifier phase: deterministic planning + concurrent execution
# ==================================================================


@dataclass(frozen=True)
class _PlannedVerifierTask:
    hotel_idx: int
    claim_id: str
    candidate: DedupedEvidence
    hypothesis: str


def _plan_verifier_tasks(
    *,
    free_text_candidates: dict[int, dict[str, list[DedupedEvidence]]],
    claim_hypotheses: dict[str, str],
    canonical_claim_order: tuple[str, ...],
    retrieval_top_k: int,
    verifier_policy: SemanticVerifierPolicy,
) -> tuple[dict[int, dict[str, list[EvidenceItem | None]]], list[_PlannedVerifierTask], list[tuple[int, str, int]]]:
    """
    Fair, deterministic round-based PLANNING ONLY - no async, no I/O,
    identical output for any concurrency level (concurrency is not
    even a parameter here):
        candidate_rank -> claim (canonical order) -> hotel (rank order)
    i.e. every active claim on every hotel gets its rank-0 (best)
    candidate PLANNED before ANY claim's rank-1 candidate is planned.
    Per-hotel budget is "effective" (widened past the base if a hotel
    genuinely has more active claims than the base would cover); the
    request budget is a safety ceiling exactly as configured, never
    silently widened.

    The traversal does NOT stop once a budget is exhausted - every
    remaining (rank, claim, hotel) combination is still visited and
    still gets an explicit SKIPPED_CALL_LIMIT EvidenceItem placed
    directly into `slots`. A claim with zero free-text candidates for
    a hotel simply never enters this loop for that hotel.

    Returns:
      - slots: hotel_idx -> claim_id -> ordered list of EvidenceItem
        (SKIPPED_CALL_LIMIT items already filled in) or None
        (placeholder for an approved task, filled in after execution)
      - approved: ordered list of _PlannedVerifierTask, in the exact
        rank -> claim -> hotel plan order
      - approved_positions: parallel to `approved` - (hotel_idx,
        claim_id, slot_index) telling the executor exactly where each
        approved task's eventual result belongs in `slots`
    """
    hotel_indices = sorted(free_text_candidates.keys())

    base = verifier_policy.normalized_max_calls_per_hotel()
    hotel_budgets: dict[int, int] = {}
    for hotel_idx in hotel_indices:
        n_active_claims = sum(1 for claim_id in canonical_claim_order if free_text_candidates[hotel_idx].get(claim_id))
        hotel_budgets[hotel_idx] = effective_per_hotel_budget(base=base, n_active_claims=n_active_claims)

    # A safety ceiling exactly as configured - never silently widened.
    # "Consistent with the effective hotel budgets" is achieved by
    # SemanticVerifierPolicy's own default (60 = 5 hotels x MAX_CLAIMS),
    # not by overriding whatever a caller explicitly configures.
    request_budget = verifier_policy.normalized_max_calls_per_request()

    slots: dict[int, dict[str, list[EvidenceItem | None]]] = {
        hotel_idx: {claim_id: [] for claim_id in canonical_claim_order} for hotel_idx in hotel_indices
    }
    approved: list[_PlannedVerifierTask] = []
    approved_positions: list[tuple[int, str, int]] = []

    for rank in range(retrieval_top_k):
        for claim_id in canonical_claim_order:
            hypothesis = claim_hypotheses[claim_id]
            for hotel_idx in hotel_indices:
                candidates = free_text_candidates[hotel_idx].get(claim_id, [])
                if rank >= len(candidates):
                    continue
                candidate = candidates[rank]

                if request_budget <= 0 or hotel_budgets[hotel_idx] <= 0:
                    slots[hotel_idx][claim_id].append(
                        EvidenceItem(
                            evidence_text=candidate.text,
                            relation=None,
                            resolution_method=ClaimResolutionMethod.GEMINI,
                            resolution_status=EvidenceResolutionStatus.SKIPPED_CALL_LIMIT,
                            error="skipped: verifier call limit reached",
                            source_type=candidate.source_type,
                            source_path=candidate.source_path,
                            retrieval_score=candidate.retrieval_score,
                        )
                    )
                    continue

                request_budget -= 1
                hotel_budgets[hotel_idx] -= 1

                slot_index = len(slots[hotel_idx][claim_id])
                slots[hotel_idx][claim_id].append(None)  # placeholder, filled in after execution
                approved.append(_PlannedVerifierTask(hotel_idx=hotel_idx, claim_id=claim_id, candidate=candidate, hypothesis=hypothesis))
                approved_positions.append((hotel_idx, claim_id, slot_index))

    return slots, approved, approved_positions


def _new_verifier_usage_accum() -> dict:
    return {
        "calls": 0, "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "estimated_cost_usd": 0.0, "cost_known": False,
        "parse_failures": 0, "errors": [],
    }


async def schedule_gemini_verification(
    *,
    free_text_candidates: dict[int, dict[str, list[DedupedEvidence]]],
    claim_hypotheses: dict[str, str] = CLAIM_HYPOTHESES,
    canonical_claim_order: tuple[str, ...] = CANONICAL_CLAIM_ORDER,
    retrieval_top_k: int,
    verifier_policy: SemanticVerifierPolicy,
    trace: RequestTrace | None,
) -> tuple[dict[int, dict[str, list[EvidenceItem]]], dict[int, dict]]:
    """
    Plans (see _plan_verifier_tasks - deterministic, unaffected by
    concurrency) which candidates get an approved Gemini call vs.
    SKIPPED_CALL_LIMIT, THEN executes only the approved tasks
    concurrently, bounded by verifier_policy.
    normalized_semantic_verifier_max_concurrency(). Each executed
    result is restored into its pre-planned slot by position (never by
    completion order), so the final gemini_items structure is
    identical for any concurrency level.

    Returns (gemini_items, verifier_usage_accum) - same shape as
    before: gemini_items maps hotel_idx -> claim_id ->
    list[EvidenceItem]; verifier_usage_accum maps hotel_idx -> a
    running usage dict, now built directly from each
    SemanticVerificationResult's own telemetry fields (not by peeking
    at trace.llm_calls[-1], which is unsafe once calls run
    concurrently and may complete in a different order than they were
    issued).
    """
    slots, approved, approved_positions = _plan_verifier_tasks(
        free_text_candidates=free_text_candidates,
        claim_hypotheses=claim_hypotheses,
        canonical_claim_order=canonical_claim_order,
        retrieval_top_k=retrieval_top_k,
        verifier_policy=verifier_policy,
    )

    verifier_usage_accum: dict[int, dict] = {hotel_idx: _new_verifier_usage_accum() for hotel_idx in free_text_candidates}

    if approved:
        coros = [
            verify_evidence_relation(task.candidate.text, task.hypothesis, policy=verifier_policy, trace=trace)
            for task in approved
        ]
        results = await run_bounded(coros, max_concurrency=verifier_policy.normalized_semantic_verifier_max_concurrency())

        for task, result, (hotel_idx, claim_id, slot_index) in zip(approved, results, approved_positions):
            item = EvidenceItem(
                evidence_text=task.candidate.text,
                relation=result.relation,
                resolution_method=ClaimResolutionMethod.GEMINI,
                resolution_status=result.status,
                error=result.error,
                source_type=task.candidate.source_type,
                source_path=task.candidate.source_path,
                retrieval_score=task.candidate.retrieval_score,
                verifier_reason=result.reason,
            )
            slots[hotel_idx][claim_id][slot_index] = item

            accum = verifier_usage_accum[hotel_idx]
            accum["calls"] += 1
            accum["latency_ms"] += result.latency_ms
            accum["input_tokens"] += result.prompt_tokens or 0
            accum["output_tokens"] += result.completion_tokens or 0
            accum["total_tokens"] += result.total_tokens or 0
            if result.estimated_cost_usd is not None:
                accum["estimated_cost_usd"] += result.estimated_cost_usd
                accum["cost_known"] = True
            if result.parse_failure:
                accum["parse_failures"] += 1
            if result.error:
                accum["errors"].append(result.error)

    # slots no longer contains any None placeholder at this point -
    # every approved position was filled in by the loop above.
    gemini_items: dict[int, dict[str, list[EvidenceItem]]] = slots  # type: ignore[assignment]

    return gemini_items, verifier_usage_accum


async def build_shadow_soft_preference_evidence(
    *,
    listings: list[ListingRaw],
    claim_assignments: list[AtomicClaimAssignment],
    pipeline_policy: SoftEvidencePipelinePolicy,
    verifier_policy: SemanticVerifierPolicy,
    trace: RequestTrace | None = None,
) -> list[SoftPreferenceEvidence]:
    """
    Builds one SoftPreferenceEvidence per listing (positionally
    aligned with `listings`). Callers decide whether to call this at
    all (e.g. skip entirely when claim_assignments is empty or a
    listing is outside shadow scope) - this function always returns a
    populated result for every listing it's given; None-vs-populated
    is an integration-layer decision, not made here.
    """
    claim_ids = [claim_id for claim_id in CANONICAL_CLAIM_ORDER if any(a.claim_id == claim_id for a in claim_assignments)]

    query_vectors_by_claim, query_outcome, pools, pool_vectors, pool_outcomes = await _run_embedding_phase(
        listings=listings, claim_ids=claim_ids, pipeline_policy=pipeline_policy, trace=trace,
    )

    contexts: dict[int, dict[str, _HotelClaimContext]] = {}

    for hotel_idx, listing in enumerate(listings):
        contexts[hotel_idx] = {claim_id: _HotelClaimContext() for claim_id in claim_ids}

        pool = pools[hotel_idx]
        pool_status, pool_errors = compute_pool_retrieval_status(pool_outcomes[hotel_idx])

        for claim_id in claim_ids:
            ctx = contexts[hotel_idx][claim_id]
            ctx.retrieval_status, ctx.retrieval_errors = compute_claim_retrieval_status(
                query_outcome=query_outcome, pool_status=pool_status, pool_errors=pool_errors,
            )

            if not query_outcome.success:
                continue  # no query vector at all - nothing to retrieve for this claim

            retrieved = retrieve_top_k(
                query_vectors_by_claim[claim_id], pool, pool_vectors[hotel_idx], k=pipeline_policy.retrieval_top_k,
            )
            ctx.retrieved_candidate_count = len(retrieved)

            routed = clean_and_route_candidates(claim_id, retrieved)
            ctx.deterministic_items = routed.deterministic_items
            ctx.free_text_candidates = routed.free_text_candidates

    free_text_candidates_by_hotel = {
        hotel_idx: {claim_id: contexts[hotel_idx][claim_id].free_text_candidates for claim_id in claim_ids}
        for hotel_idx in contexts
    }

    gemini_items, verifier_usage_accum = await schedule_gemini_verification(
        free_text_candidates=free_text_candidates_by_hotel,
        retrieval_top_k=pipeline_policy.retrieval_top_k,
        verifier_policy=verifier_policy,
        trace=trace,
    )

    results: list[SoftPreferenceEvidence] = []
    for hotel_idx in range(len(listings)):
        claims: list[AtomicClaimResult] = []
        for claim_id in claim_ids:
            ctx = contexts[hotel_idx][claim_id]
            evidence_items = ctx.deterministic_items + gemini_items.get(hotel_idx, {}).get(claim_id, [])
            claims.append(
                AtomicClaimResult.from_evidence_items(
                    claim_id=claim_id,
                    hypothesis=CLAIM_HYPOTHESES[claim_id],
                    evidence_items=evidence_items,
                    retrieval_status=ctx.retrieval_status,
                    retrieval_errors=ctx.retrieval_errors,
                    retrieved_candidate_count=ctx.retrieved_candidate_count,
                )
            )

        accum = verifier_usage_accum.get(hotel_idx, {"calls": 0})
        semantic_verifier = (
            SemanticVerifierUsage(
                model=verifier_policy.model,
                calls=accum["calls"],
                latency_ms=round(accum["latency_ms"], 2),
                input_tokens=accum["input_tokens"],
                output_tokens=accum["output_tokens"],
                total_tokens=accum["total_tokens"],
                estimated_cost_usd=round(accum["estimated_cost_usd"], 6) if accum.get("cost_known") else None,
                parse_failures=accum["parse_failures"],
                errors=accum["errors"],
            )
            if accum["calls"] > 0
            else None
        )

        results.append(SoftPreferenceEvidence(claims=claims, semantic_verifier=semantic_verifier))

    return results
