"""
Embedding-based evidence retrieval for the new soft-evidence pipeline.

Ported/adapted from the experimentally validated
evaluation/experiments/evidence_retrieval/{embeddings,retriever,similarity}.py
- NOT imported from evaluation/experiments/ (that tree stays isolated
from app/). Cosine similarity is implemented in pure Python here rather
than with numpy: numpy is only a transitive dependency of this project
today (not declared in pyproject.toml), and per-hotel evidence pools
are small enough (hundreds of items, not millions) for pure Python to
be fine.

Model is fixed to gemini-embedding-001 - the model validated by the
experiments. Not configurable to a different model in this migration.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass

from app.logic.soft_evidence_collection import SoftEvidenceCandidate
from app.observability.pricing import estimate_llm_cost_usd
from app.observability.trace import ExternalCallTrace, RequestTrace

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# The API rejects more than 100 texts in a single embed_content call
# (same limit already validated in the experiment's embeddings.py).
MAX_BATCH_SIZE = 100

EMBEDDING_STEP_QUERY = "soft_evidence_query_embedding"
EMBEDDING_STEP_POOL = "soft_evidence_pool_embedding"


@dataclass(frozen=True)
class EmbedBatchOutcome:
    """
    Result of one embed_content call (one batch, <=MAX_BATCH_SIZE texts).

    token_count is None today: a live compatibility check
    (tests/test_embedding_google_genai_compat.py) confirmed the Gemini
    Developer API's embed_content does not populate usage/token
    statistics at all (EmbedContentResponse.metadata and
    ContentEmbedding.statistics both come back None). Token usage is
    therefore genuinely unknown, not estimated from string length -
    estimated_cost_usd is None whenever token_count is None, matching
    estimate_llm_cost_usd's existing None-input behavior.
    """

    success: bool
    batch_size: int
    latency_ms: float
    error: str | None
    token_count: int | None
    estimated_cost_usd: float | None


@dataclass(frozen=True)
class RetrievedEvidence:
    text: str
    source_type: str
    source_path: str | None
    retrieval_score: float


def _gemini_client():
    try:
        from google.genai import Client
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    return Client(api_key=api_key)


def _sum_token_count(response) -> int | None:
    """
    Sums ContentEmbedding.statistics.token_count across a response's
    embeddings, if present. Returns None (not 0) if unavailable for any
    embedding, so "unknown" is never confused with "zero tokens used".
    """
    embeddings = getattr(response, "embeddings", None) or []
    counts: list[int] = []
    for embedding in embeddings:
        statistics = getattr(embedding, "statistics", None)
        token_count = getattr(statistics, "token_count", None) if statistics else None
        if token_count is None:
            return None
        counts.append(token_count)
    return sum(counts) if counts else None


def _embed_batch(
    texts: list[str],
    *,
    model: str,
    trace: RequestTrace | None,
    step: str,
) -> tuple[list[list[float]] | None, EmbedBatchOutcome]:
    """
    One embed_content call for <=MAX_BATCH_SIZE texts. Returns
    (vectors, outcome) on success, (None, outcome) on failure - the
    caller decides what a failed batch means for retrieval_status.
    """
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"batch of {len(texts)} texts exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}")

    started = time.perf_counter()
    try:
        client = _gemini_client()
        response = client.models.embed_content(model=model, contents=texts)
    except Exception as e:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        error = f"{type(e).__name__}: {e}"

        if trace is not None:
            trace.add_external_call(
                ExternalCallTrace(
                    step=step,
                    provider="gemini_embedding",
                    latency_ms=latency_ms,
                    estimated_cost_usd=None,
                    success=False,
                    error=error,
                    metadata={"batch_size": len(texts), "model": model, "token_count": None},
                )
            )

        return None, EmbedBatchOutcome(
            success=False,
            batch_size=len(texts),
            latency_ms=latency_ms,
            error=error,
            token_count=None,
            estimated_cost_usd=None,
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    token_count = _sum_token_count(response)
    estimated_cost_usd = estimate_llm_cost_usd(model=model, prompt_tokens=token_count, completion_tokens=0)

    if trace is not None:
        trace.add_external_call(
            ExternalCallTrace(
                step=step,
                provider="gemini_embedding",
                latency_ms=latency_ms,
                estimated_cost_usd=estimated_cost_usd,
                success=True,
                metadata={"batch_size": len(texts), "model": model, "token_count": token_count},
            )
        )

    vectors = [list(embedding.values) for embedding in response.embeddings]
    return vectors, EmbedBatchOutcome(
        success=True,
        batch_size=len(texts),
        latency_ms=latency_ms,
        error=None,
        token_count=token_count,
        estimated_cost_usd=estimated_cost_usd,
    )


async def _embed_batch_async(
    texts: list[str],
    *,
    model: str,
    trace: RequestTrace | None,
    step: str,
) -> tuple[list[list[float]] | None, EmbedBatchOutcome]:
    """
    Async wrapper around _embed_batch, via asyncio.to_thread - the same
    pattern already used by semantic_evidence_verification.
    verify_evidence_relation for its (also synchronous) google-genai
    SDK call. _embed_batch itself is completely unchanged: same call,
    same trace recording, same outcome shape - this only lets it run
    on a worker thread so multiple batches can be in flight at once.
    """
    return await asyncio.to_thread(_embed_batch, texts, model=model, trace=trace, step=step)


def embed_query_texts(
    queries: list[tuple[str, str]],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    trace: RequestTrace | None = None,
) -> tuple[dict[str, list[float]], EmbedBatchOutcome]:
    """
    queries: list of (claim_id, query_text) pairs - the unique set of
    retrieval queries needed for one request. Embedded in exactly ONE
    batched call (request-level, well under MAX_BATCH_SIZE for the 12
    Phase B claim IDs) and reused across every shadow hotel - callers
    must not call this per hotel.

    Returns {} on failure (caller inspects the returned outcome for the
    error/status - see soft_evidence_orchestration.py).
    """
    if not queries:
        return {}, EmbedBatchOutcome(
            success=True, batch_size=0, latency_ms=0.0, error=None, token_count=None, estimated_cost_usd=None
        )

    texts = [query_text for _, query_text in queries]
    vectors, outcome = _embed_batch(texts, model=model, trace=trace, step=EMBEDDING_STEP_QUERY)

    if vectors is None:
        return {}, outcome

    return {claim_id: vector for (claim_id, _), vector in zip(queries, vectors)}, outcome


def embed_evidence_pool(
    candidates: list[SoftEvidenceCandidate],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    trace: RequestTrace | None = None,
) -> tuple[list[list[float] | None], list[EmbedBatchOutcome]]:
    """
    Embeds one hotel's full evidence pool (no truncation), split into
    batches of <=MAX_BATCH_SIZE. Returns one vector-or-None per input
    candidate (None for candidates whose batch failed - positionally
    aligned with `candidates`) plus the per-batch outcomes, so the
    caller can derive SUCCESS/PARTIAL/FAILED.
    """
    if not candidates:
        return [], []

    vectors: list[list[float] | None] = [None] * len(candidates)
    outcomes: list[EmbedBatchOutcome] = []

    for start in range(0, len(candidates), MAX_BATCH_SIZE):
        batch = candidates[start : start + MAX_BATCH_SIZE]
        batch_vectors, outcome = _embed_batch(
            [c.text for c in batch], model=model, trace=trace, step=EMBEDDING_STEP_POOL
        )
        outcomes.append(outcome)

        if batch_vectors is not None:
            for offset, vector in enumerate(batch_vectors):
                vectors[start + offset] = vector

    return vectors, outcomes


def _cosine_similarity(query: list[float], candidate: list[float]) -> float:
    dot = sum(q * c for q, c in zip(query, candidate))
    query_norm = math.sqrt(sum(q * q for q in query))
    candidate_norm = math.sqrt(sum(c * c for c in candidate))
    denom = query_norm * candidate_norm
    if denom == 0:
        return 0.0
    return dot / denom


def retrieve_top_k(
    query_vector: list[float],
    candidates: list[SoftEvidenceCandidate],
    candidate_vectors: list[list[float] | None],
    k: int,
) -> list[RetrievedEvidence]:
    """
    candidate_vectors must be positionally aligned with candidates
    (i.e. come from embed_evidence_pool(candidates) for the same list).
    A candidate whose vector is None (its batch failed) is excluded
    from retrieval - it never becomes a false top-K hit.
    """
    if len(candidates) != len(candidate_vectors):
        raise ValueError(
            "candidates and candidate_vectors must have the same length "
            f"(got {len(candidates)} and {len(candidate_vectors)})"
        )

    scored: list[tuple[float, SoftEvidenceCandidate]] = []
    for candidate, vector in zip(candidates, candidate_vectors):
        if vector is None:
            continue
        scored.append((_cosine_similarity(query_vector, vector), candidate))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        RetrievedEvidence(
            text=candidate.text,
            source_type=candidate.source_type,
            source_path=candidate.source_path,
            retrieval_score=round(float(score), 6),
        )
        for score, candidate in scored[: max(0, k)]
    ]
