from __future__ import annotations

from evaluation.experiments.evidence_retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_texts,
)
from evaluation.experiments.evidence_retrieval.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceMetadata,
)
from evaluation.experiments.evidence_retrieval.similarity import top_k


def embed_chunks(
    chunks: list[EvidenceChunk],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """
    Embed all chunks for one property in a single batched call.

    Call this once per property and reuse the returned vectors across
    multiple constraint queries (retrieve_evidence takes them in) -
    do not re-embed the same chunks per query.
    """
    return embed_texts([chunk.text for chunk in chunks], model=model)


def retrieve_evidence(
    chunks: list[EvidenceChunk],
    chunk_vectors: list[list[float]],
    constraint_text: str,
    *,
    k: int = 5,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[EvidenceMatch]:
    """
    Find the top-k evidence chunks most semantically similar to a
    soft constraint.

    chunk_vectors must line up positionally with chunks (i.e. come
    from embed_chunks(chunks) for the same chunk list).
    """
    if len(chunks) != len(chunk_vectors):
        raise ValueError(
            "chunks and chunk_vectors must have the same length "
            f"(got {len(chunks)} and {len(chunk_vectors)})"
        )

    if not chunks:
        return []

    query_vector = embed_texts([constraint_text], model=model)[0]
    ranked = top_k(query_vector, chunk_vectors, k)

    results: list[EvidenceMatch] = []
    for index, score in ranked:
        chunk = chunks[index]
        results.append(
            EvidenceMatch(
                text=chunk.text,
                similarity_score=round(float(score), 6),
                metadata=EvidenceMetadata(
                    property_id=chunk.property_id,
                    source_type=chunk.source_type,
                    path=chunk.path,
                    review_date=chunk.review_date,
                ),
            )
        )

    return results
