"""
Manual smoke test for the evidence retrieval prototype.

Not wired into any eval runner or CI. Makes real, billed calls to the
Gemini embeddings API (GOOGLE_API_KEY must be set) - run it deliberately,
not as part of an automated pipeline.

Usage:
    python -m evaluation.experiments.evidence_retrieval.demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.retrieval.apify import normalize_apify_listing
from app.retrieval.fixtures import FIXTURES_PATH
from app.schemas.listing import ListingRaw
from evaluation.experiments.evidence_retrieval.chunking import build_evidence_chunks
from evaluation.experiments.evidence_retrieval.retriever import (
    embed_chunks,
    retrieve_evidence,
)

SOFT_CONSTRAINTS = [
    "quiet",
    "good for remote work",
    "good breakfast",
    "family-friendly",
]

REAL_APIFY_LOG = (
    PROJECT_ROOT
    / "logs/apify_raw/apify_raw_20260812_172615.json"
)


def load_fixture_listings(limit: int = 2) -> list[ListingRaw]:
    data = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    listings = [ListingRaw.model_validate(item) for item in data]
    return [l for l in listings if l.description][:limit]


def load_real_logged_listings(limit: int = 2) -> list[ListingRaw]:
    if not REAL_APIFY_LOG.exists():
        return []

    payload = json.loads(REAL_APIFY_LOG.read_text(encoding="utf-8"))
    items = payload.get("items") or []

    listings: list[ListingRaw] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            listing = normalize_apify_listing(item)
        except Exception:
            continue
        if listing.description:
            listings.append(listing)
        if len(listings) >= limit:
            break

    return listings


def run_demo_for_listing(listing: ListingRaw) -> None:
    print("=" * 80)
    print(f"Property: {listing.name!r} (id={listing.id})")

    chunks = build_evidence_chunks(listing)
    print(f"Built {len(chunks)} evidence chunks")

    if not chunks:
        print("No evidence chunks - skipping.")
        return

    chunk_vectors = embed_chunks(chunks)

    for constraint_text in SOFT_CONSTRAINTS:
        print(f"\n--- constraint: {constraint_text!r} ---")

        matches = retrieve_evidence(
            chunks,
            chunk_vectors,
            constraint_text,
            k=5,
        )

        for rank, match in enumerate(matches, start=1):
            print(
                f"[{rank}] score={match.similarity_score:.4f} "
                f"source={match.metadata.source_type} "
                f"path={match.metadata.path}"
            )
            print(f"    text: {match.text}")


def main() -> None:
    listings = load_fixture_listings(limit=2) + load_real_logged_listings(limit=2)

    if not listings:
        print("No listings with a description found to demo against.")
        return

    for listing in listings:
        run_demo_for_listing(listing)


if __name__ == "__main__":
    main()
