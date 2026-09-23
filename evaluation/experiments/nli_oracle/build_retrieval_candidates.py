"""
End-to-end SUPPORT verification, phase 1/2: retrieval.

Runs the EXISTING, UNCHANGED retrieval implementation
(evaluation.experiments.evidence_retrieval.retriever / chunking /
embeddings) over the FULL evidence chunk pool per property (not the
gold-only evidence lists), for every one of the 32 atomic subclaims in
retrieval_checkpoint_v2.jsonl - the original checkpoint, not the
NLI-only Q1/Q2/Q3 or FC1-atomic redesign from the earlier oracle
experiments (that redesign was intentionally never merged back into
the retrieval checkpoint).

Produces top-5 retrieval candidates per subclaim, with similarity
scores and source_type, plus gold status (SUPPORTED/NOT_SUPPORTED)
and the gold SUPPORT chunk_ids for later retrieval-failure vs
verification-failure attribution.

Run with the MAIN project .venv (needs pydantic + google-genai for the
embeddings call), NOT the isolated nli_oracle venv - this phase does
no NLI inference at all.

Embeddings are cache-backed via the SAME cache file already used by
evaluation/experiments/evidence_retrieval/run_retrieval_checkpoint_v2.py
(golden/embedding_cache.json) - the chunk texts and retrieval_query
strings used here are identical to that earlier run, so this should
be a near-total cache hit, no new billed embedding calls expected for
chunks (the query embedding call inside retrieve_evidence() is not
cached by that function itself - see note in run(); this is accepted
as the unmodified behavior of the existing retrieval implementation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.schemas.listing import ListingRaw
from evaluation.core.io import load_jsonl
from evaluation.experiments.evidence_retrieval.chunking import build_evidence_chunks
from evaluation.experiments.evidence_retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from evaluation.experiments.evidence_retrieval.retriever import retrieve_evidence

CHECKPOINT_PATH = (
    PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl"
)
PROPERTIES_DIR = PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/properties"
CACHE_PATH = PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/embedding_cache.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "retrieval_candidates.jsonl"

TOP_K = 5


def load_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def embed_with_cache(texts: list[str], cache: dict[str, list[float]]) -> list[list[float]]:
    missing = [t for t in texts if t not in cache]
    if missing:
        vectors = embed_texts(missing, model=DEFAULT_EMBEDDING_MODEL)
        for t, v in zip(missing, vectors):
            cache[t] = v
    return [cache[t] for t in texts]


def main() -> None:
    pairs = load_jsonl(CHECKPOINT_PATH)
    cache = load_cache()

    property_slugs = sorted({p["property_slug"] for p in pairs})
    property_data: dict[str, dict[str, Any]] = {}
    for slug in property_slugs:
        listing = ListingRaw.model_validate(
            json.loads((PROPERTIES_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        )
        chunks = build_evidence_chunks(listing)
        vectors = embed_with_cache([c.text for c in chunks], cache)
        # path alone is not a unique key: collect_listing_signals() reuses the
        # same path (e.g. "listing.description") across every sentence split
        # from one field. (path, text) together is verified unique per
        # property (checked empirically for all 4 properties here).
        keyed_chunk_id = {(c.path, c.text): c.chunk_id for c in chunks}
        if len(keyed_chunk_id) != len(chunks):
            raise RuntimeError(f"Non-unique (path, text) chunk keys for {slug} - lookup would be ambiguous")
        property_data[slug] = {"chunks": chunks, "vectors": vectors, "keyed_chunk_id": keyed_chunk_id}

    save_cache(cache)  # persist any chunk-embedding cache additions before the (uncached) query calls

    rows: list[dict[str, Any]] = []
    n_subclaims = 0

    for pair in pairs:
        slug = pair["property_slug"]
        data = property_data[slug]

        for subclaim_key, subclaim in pair["subclaims"].items():
            n_subclaims += 1
            gold_support_chunk_ids = sorted(
                e["chunk_id"] for e in subclaim["evidence"] if e["relation"] == "SUPPORT"
            )
            gold_status = "SUPPORTED" if gold_support_chunk_ids else "NOT_SUPPORTED"

            matches = retrieve_evidence(
                data["chunks"],
                data["vectors"],
                subclaim["retrieval_query"],
                k=TOP_K,
            )

            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "property_slug": slug,
                    "constraint_text": pair["constraint_text"],
                    "subclaim": subclaim_key,
                    "hypothesis": subclaim["hypothesis"],
                    "retrieval_query": subclaim["retrieval_query"],
                    "gold_status": gold_status,
                    "gold_support_chunk_ids": gold_support_chunk_ids,
                    "n_gold_evidence_total": len(subclaim["evidence"]),
                    "top_k": [
                        {
                            "rank": i + 1,
                            "chunk_id": data["keyed_chunk_id"].get((m.metadata.path, m.text)),
                            "text": m.text,
                            "similarity_score": m.similarity_score,
                            "source_type": m.metadata.source_type,
                            "path": m.metadata.path,
                        }
                        for i, m in enumerate(matches)
                    ],
                }
            )

    n_supported = sum(1 for r in rows if r["gold_status"] == "SUPPORTED")
    print(f"n_subclaims total: {n_subclaims} (expected 32)")
    print(f"SUPPORTED: {n_supported}, NOT_SUPPORTED: {n_subclaims - n_supported}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} subclaim rows with top-{TOP_K} candidates -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
