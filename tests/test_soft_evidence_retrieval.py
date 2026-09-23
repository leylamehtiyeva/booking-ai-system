from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.logic import soft_evidence_retrieval as ser
from app.logic.soft_evidence_collection import SoftEvidenceCandidate
from app.observability.trace import RequestTrace


def _embedding(values: list[float], token_count: int | None = None) -> SimpleNamespace:
    statistics = SimpleNamespace(token_count=token_count) if token_count is not None else None
    return SimpleNamespace(values=values, statistics=statistics)


def _response(vectors: list[list[float]], token_counts: list[int | None] | None = None, metadata=None) -> SimpleNamespace:
    token_counts = token_counts or [None] * len(vectors)
    return SimpleNamespace(
        embeddings=[_embedding(v, t) for v, t in zip(vectors, token_counts)],
        metadata=metadata,
    )


def _client_returning(responses: list[SimpleNamespace] | SimpleNamespace):
    """Mocked client.models.embed_content - a list means one call per element (in order)."""
    if not isinstance(responses, list):
        responses = [responses]
    calls = {"batches": []}

    def _embed_content(*, model, contents):
        calls["batches"].append(list(contents))
        idx = len(calls["batches"]) - 1
        resp = responses[idx]
        if isinstance(resp, Exception):
            raise resp
        return resp

    client = SimpleNamespace(models=SimpleNamespace(embed_content=_embed_content))
    return client, calls


# ---------------- query embeddings: one call, reusable ----------------


def test_embed_query_texts_makes_exactly_one_call(monkeypatch):
    queries = [("Q1", "soundproofing"), ("Q2", "quiet surroundings"), ("BQ1", "The breakfast is good in quality.")]
    client, calls = _client_returning(_response([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    trace = RequestTrace()
    vectors_by_claim, outcome = ser.embed_query_texts(queries, trace=trace)

    assert len(calls["batches"]) == 1
    assert calls["batches"][0] == ["soundproofing", "quiet surroundings", "The breakfast is good in quality."]
    assert set(vectors_by_claim) == {"Q1", "Q2", "BQ1"}
    assert vectors_by_claim["Q1"] == [0.1, 0.2]
    assert outcome.success is True
    assert len(trace.external_calls) == 1
    assert trace.external_calls[0].provider == "gemini_embedding"


def test_embed_query_texts_empty_list_makes_no_call(monkeypatch):
    client, calls = _client_returning(_response([]))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    vectors_by_claim, outcome = ser.embed_query_texts([], trace=None)
    assert vectors_by_claim == {}
    assert outcome.success is True
    assert len(calls["batches"]) == 0


def test_embed_query_texts_failure_returns_empty_dict_and_records_trace(monkeypatch):
    client, _ = _client_returning(TimeoutError("upstream timed out"))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    trace = RequestTrace()
    vectors_by_claim, outcome = ser.embed_query_texts([("Q1", "soundproofing")], trace=trace)

    assert vectors_by_claim == {}
    assert outcome.success is False
    assert "TimeoutError" in outcome.error
    assert trace.external_calls[0].success is False


# ---------------- pool batching, no truncation ----------------


def test_pool_batching_splits_into_batches_of_100(monkeypatch):
    n = 250
    candidates = [SoftEvidenceCandidate(text=f"t{i}", source_type="description", source_path=None) for i in range(n)]

    responses = [
        _response([[1.0] for _ in range(100)]),
        _response([[1.0] for _ in range(100)]),
        _response([[1.0] for _ in range(50)]),
    ]
    client, calls = _client_returning(responses)
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    vectors, outcomes = ser.embed_evidence_pool(candidates, trace=None)

    assert len(calls["batches"]) == 3
    assert [len(b) for b in calls["batches"]] == [100, 100, 50]
    assert len(vectors) == n
    assert all(v is not None for v in vectors)
    assert all(o.success for o in outcomes)


def test_full_pool_embedded_no_200_item_truncation(monkeypatch):
    n = 350  # above the previously-considered-and-rejected 200 cap
    candidates = [SoftEvidenceCandidate(text=f"t{i}", source_type="description", source_path=None) for i in range(n)]

    responses = [
        _response([[1.0] for _ in range(100)]),
        _response([[1.0] for _ in range(100)]),
        _response([[1.0] for _ in range(100)]),
        _response([[1.0] for _ in range(50)]),
    ]
    client, calls = _client_returning(responses)
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    vectors, outcomes = ser.embed_evidence_pool(candidates, trace=None)
    assert len(vectors) == 350
    assert sum(len(b) for b in calls["batches"]) == 350


def test_empty_pool_makes_no_call(monkeypatch):
    client, calls = _client_returning(_response([]))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)
    vectors, outcomes = ser.embed_evidence_pool([], trace=None)
    assert vectors == []
    assert outcomes == []
    assert len(calls["batches"]) == 0


def test_partial_batch_failure_leaves_some_vectors_none(monkeypatch):
    candidates = [SoftEvidenceCandidate(text=f"t{i}", source_type="description", source_path=None) for i in range(150)]
    responses = [_response([[1.0] for _ in range(100)]), TimeoutError("batch 2 failed")]
    client, calls = _client_returning(responses)
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    trace = RequestTrace()
    vectors, outcomes = ser.embed_evidence_pool(candidates, trace=trace)

    assert len(vectors) == 150
    assert all(v is not None for v in vectors[:100])
    assert all(v is None for v in vectors[100:])
    assert outcomes[0].success is True
    assert outcomes[1].success is False
    assert len(trace.external_calls) == 2


# ---------------- cosine top-K correctness ----------------


def test_retrieve_top_k_orders_by_similarity_descending():
    candidates = [
        SoftEvidenceCandidate(text="exact match", source_type="description", source_path="listing.description"),
        SoftEvidenceCandidate(text="orthogonal", source_type="description", source_path="listing.description"),
        SoftEvidenceCandidate(text="close match", source_type="description", source_path="listing.description"),
    ]
    query_vector = [1.0, 0.0]
    candidate_vectors = [
        [1.0, 0.0],   # cosine similarity 1.0
        [0.0, 1.0],   # cosine similarity 0.0
        [0.9, 0.1],   # cosine similarity < 1.0 but > 0
    ]

    results = ser.retrieve_top_k(query_vector, candidates, candidate_vectors, k=3)

    assert [r.text for r in results] == ["exact match", "close match", "orthogonal"]
    assert results[0].retrieval_score == 1.0
    assert results[2].retrieval_score == 0.0


def test_retrieve_top_k_respects_k():
    candidates = [SoftEvidenceCandidate(text=f"t{i}", source_type="description", source_path=None) for i in range(5)]
    vectors = [[float(i), 0.0] for i in range(5)]
    results = ser.retrieve_top_k([1.0, 0.0], candidates, vectors, k=2)
    assert len(results) == 2


def test_retrieve_top_k_excludes_none_vectors():
    candidates = [
        SoftEvidenceCandidate(text="has vector", source_type="description", source_path=None),
        SoftEvidenceCandidate(text="failed embedding", source_type="description", source_path=None),
    ]
    vectors = [[1.0, 0.0], None]
    results = ser.retrieve_top_k([1.0, 0.0], candidates, vectors, k=5)
    assert len(results) == 1
    assert results[0].text == "has vector"


def test_retrieve_top_k_preserves_source_type_and_path():
    candidates = [SoftEvidenceCandidate(text="wifi", source_type="facilities", source_path="listing.facilities[0].name")]
    results = ser.retrieve_top_k([1.0], candidates, [[1.0]], k=1)
    assert results[0].source_type == "facilities"
    assert results[0].source_path == "listing.facilities[0].name"


def test_retrieve_top_k_mismatched_lengths_raises():
    candidates = [SoftEvidenceCandidate(text="a", source_type="description", source_path=None)]
    try:
        ser.retrieve_top_k([1.0], candidates, [[1.0], [2.0]], k=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------- token/cost accounting: real metadata only, never estimated ----------------


def test_cost_computed_when_token_count_is_available(monkeypatch):
    client, _ = _client_returning(_response([[1.0, 2.0]], token_counts=[1000]))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    trace = RequestTrace()
    _, outcome = ser.embed_query_texts([("Q1", "soundproofing")], trace=trace)

    assert outcome.token_count == 1000
    assert outcome.estimated_cost_usd == pytest.approx(1000 / 1_000_000 * 0.15)
    assert trace.external_calls[0].estimated_cost_usd == outcome.estimated_cost_usd


def test_cost_is_none_when_token_count_unavailable_not_estimated_from_length(monkeypatch):
    """
    Confirmed real-API behavior: embed_content does not populate
    statistics.token_count. Cost must be None (unknown), never
    estimated from string length.
    """
    client, _ = _client_returning(_response([[1.0, 2.0]], token_counts=[None]))
    monkeypatch.setattr(ser, "_gemini_client", lambda: client)

    trace = RequestTrace()
    _, outcome = ser.embed_query_texts([("Q1", "a very long soundproofing related query text")], trace=trace)

    assert outcome.token_count is None
    assert outcome.estimated_cost_usd is None
    assert trace.external_calls[0].estimated_cost_usd is None
    assert trace.external_calls[0].metadata["token_count"] is None
