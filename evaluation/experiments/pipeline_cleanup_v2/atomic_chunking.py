"""
Atomic evidence pool construction: takes the existing, unmodified
build_evidence_chunks() output and expands any chunk whose text splits
into more than one sentence (via the existing production
split_into_sentences() helper) into per-sentence atomic chunks.

In practice this only visibly affects `policies` content (description
is already sentence-split by collect_listing_signals(); facilities/
room_facilities/highlights entries are already atomic tags/phrases -
splitting them is a no-op).

Also defines the heading-detection rule (path suffix .title/.header),
derived from the schema's own field semantics, not from text content.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.logic.listing_signals import split_into_sentences
from app.schemas.listing import ListingRaw
from evaluation.experiments.evidence_retrieval.chunking import build_evidence_chunks
from evaluation.experiments.pipeline_cleanup_v2.heading_rules import is_heading


@dataclass(frozen=True)
class AtomicChunk:
    chunk_id: str
    source_chunk_id: str  # original chunk_id before any sentence-splitting
    property_id: str | None
    source_type: str
    path: str | None
    text: str
    is_heading: bool
    # required for structural compatibility with the unmodified
    # evidence_retrieval.retriever.retrieve_evidence(), which reads
    # chunk.review_date when building EvidenceMetadata - always None,
    # same as the underlying EvidenceChunk (no review data exists).
    review_date: str | None = None


def build_atomic_pool(listing: ListingRaw) -> list[AtomicChunk]:
    base_chunks = build_evidence_chunks(listing)
    atomic: list[AtomicChunk] = []

    for chunk in base_chunks:
        sentences = split_into_sentences(chunk.text)

        if len(sentences) <= 1:
            atomic.append(
                AtomicChunk(
                    chunk_id=chunk.chunk_id,
                    source_chunk_id=chunk.chunk_id,
                    property_id=chunk.property_id,
                    source_type=chunk.source_type,
                    path=chunk.path,
                    text=chunk.text,
                    is_heading=is_heading(chunk.path),
                )
            )
            continue

        for i, sentence in enumerate(sentences):
            atomic.append(
                AtomicChunk(
                    chunk_id=f"{chunk.chunk_id}#s{i}",
                    source_chunk_id=chunk.chunk_id,
                    property_id=chunk.property_id,
                    source_type=chunk.source_type,
                    path=f"{chunk.path}#s{i}" if chunk.path else None,
                    text=sentence,
                    is_heading=is_heading(chunk.path),  # heading-ness inherited from the field, not the sentence
                )
            )

    return atomic
