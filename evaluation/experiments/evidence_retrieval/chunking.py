from __future__ import annotations

from app.logic.listing_signals import collect_listing_signals
from app.schemas.listing import ListingRaw
from evaluation.experiments.evidence_retrieval.models import EvidenceChunk


def _source_type_from_path(path: str) -> str:
    """
    Mirrors app.logic.constraint_evidence_resolution._source_from_path.

    Duplicated on purpose (not imported): that function is a private
    helper of the production LLM fallback module, and this prototype
    is meant to stay isolated from it. Keeping the same mapping here
    makes source_type comparable to what approach A already reports.
    """
    if path.startswith("listing.facilities"):
        return "facilities"
    if path.startswith("rooms["):
        return "room_facilities"
    if path.startswith("policies["):
        return "policies"
    if path.startswith("highlights["):
        return "highlights"
    if path.startswith("listing.description"):
        return "description"
    if path.startswith("listing.name"):
        return "title"
    if path.startswith("listing.property_type"):
        return "property_type"
    return "other"


def build_evidence_chunks(listing: ListingRaw) -> list[EvidenceChunk]:
    """
    Build retrievable evidence chunks for one property.

    Reuses collect_listing_signals() (the same evidence extraction
    the production LLM fallback uses) so this prototype is evaluated
    against an evidence pool that is directly comparable to approach A,
    rather than inventing a separate extraction path.
    """
    property_id = listing.id
    chunks: list[EvidenceChunk] = []

    for index, signal in enumerate(collect_listing_signals(listing)):
        text = (signal.raw_text or signal.text or "").strip()
        if not text:
            continue

        chunks.append(
            EvidenceChunk(
                chunk_id=f"{property_id or 'unknown'}::{index}",
                property_id=property_id,
                source_type=_source_type_from_path(signal.path),
                path=signal.path,
                text=text,
                review_date=None,
            )
        )

    return chunks
