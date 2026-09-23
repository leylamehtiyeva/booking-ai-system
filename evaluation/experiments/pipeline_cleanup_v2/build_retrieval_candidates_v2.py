"""
Architecture-cleanup end-to-end experiment, phase 1/2: retrieval over
the ATOMIC evidence pool (sentence-split policies, Q1/Q2/Q3 quiet
schema), using the unmodified embedding model/retriever.

Run with the MAIN project .venv (needs pydantic + google-genai), NOT
the isolated nli_oracle venv - no NLI inference in this phase.

Reuses the shared embedding cache
(evidence_retrieval/golden/embedding_cache.json) - only genuinely new
atomic sentence texts (mostly from policies content) and the 3 new
Q1/Q2/Q3 retrieval queries require new (small, confirmed) embedding
calls.

Also computes the retrieval-only Recall@1/Recall@3 diagnostic
requested before verification - isolates retrieval quality on the new
atomic schema from verification quality.
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
from evaluation.core.io import load_jsonl, save_json
from evaluation.experiments.evidence_retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from evaluation.experiments.evidence_retrieval.retriever import retrieve_evidence
from evaluation.experiments.nli_oracle.build_input_v2 import FC1_CHUNKS
from evaluation.experiments.pipeline_cleanup_v2.atomic_chunking import build_atomic_pool
from evaluation.experiments.pipeline_cleanup_v2.schema_v3 import (
    FC1_HYPOTHESIS,
    PROPERTY_SLUGS,
    QUIET_HYPOTHESES,
    QUIET_RETRIEVAL_QUERIES,
    fc1_gold_support_sentences,
    quiet_gold_support_chunk_ids,
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl"
)
PROPERTIES_DIR = PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/properties"
CACHE_PATH = PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/embedding_cache.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "retrieval_candidates_v2.jsonl"
RETRIEVAL_ONLY_METRICS_PATH = Path(__file__).resolve().parent / "retrieval_only_metrics_v2.json"

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


def recall_at(subclaim_rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    supported = [r for r in subclaim_rows if r["gold_status"] == "SUPPORTED"]
    hits = 0
    for r in supported:
        gold = set(r["gold_support_chunk_ids"])
        top_ids = {c["chunk_id"] for c in r["top_k"][:k]}
        if gold & top_ids:
            hits += 1
    return {
        "k": k,
        "n_supported": len(supported),
        "hits": hits,
        "recall": round(hits / len(supported), 4) if supported else None,
    }


def main() -> None:
    checkpoint_pairs = load_jsonl(CHECKPOINT_PATH)
    cache = load_cache()

    # ---- build atomic pools + embeddings per property ----
    property_data: dict[str, dict[str, Any]] = {}
    for slug in PROPERTY_SLUGS:
        listing = ListingRaw.model_validate(
            json.loads((PROPERTIES_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        )
        atomic_chunks = build_atomic_pool(listing)
        vectors = embed_with_cache([c.text for c in atomic_chunks], cache)
        keyed_chunk_id = {(c.path, c.text): c.chunk_id for c in atomic_chunks}
        if len(keyed_chunk_id) != len(atomic_chunks):
            raise RuntimeError(f"Non-unique (path, text) atomic chunk keys for {slug}")
        property_data[slug] = {
            "atomic_chunks": atomic_chunks,
            "vectors": vectors,
            "keyed_chunk_id": keyed_chunk_id,
        }

    save_cache(cache)

    n_new_embeddings = len(cache)  # informational only; see printed diff below

    rows: list[dict[str, Any]] = []

    # ---- helper: run retrieval for one subclaim on one property ----
    def retrieve_for(slug: str, retrieval_query: str) -> list[dict[str, Any]]:
        data = property_data[slug]
        # retriever.retrieve_evidence expects EvidenceChunk-like objects
        # with a .text attribute; our AtomicChunk dataclass qualifies.
        matches = retrieve_evidence(data["atomic_chunks"], data["vectors"], retrieval_query, k=TOP_K)
        out = []
        for i, m in enumerate(matches):
            chunk_id = data["keyed_chunk_id"].get((m.metadata.path, m.text))
            out.append(
                {
                    "rank": i + 1,
                    "chunk_id": chunk_id,
                    "text": m.text,
                    "similarity_score": m.similarity_score,
                    "source_type": m.metadata.source_type,
                    "path": m.metadata.path,
                }
            )
        return out

    # ---- 1. quiet -> Q1/Q2/Q3 ----
    gold_quiet = quiet_gold_support_chunk_ids()
    for slug in PROPERTY_SLUGS:
        for q_key, hypothesis in QUIET_HYPOTHESES.items():
            query = QUIET_RETRIEVAL_QUERIES[q_key]
            gold_ids = gold_quiet[(slug, q_key)]
            rows.append(
                {
                    "pair_id": f"{slug}__quiet",
                    "property_slug": slug,
                    "constraint_text": "quiet",
                    "subclaim": q_key,
                    "hypothesis": hypothesis,
                    "retrieval_query": query,
                    "gold_status": "SUPPORTED" if gold_ids else "NOT_SUPPORTED",
                    "gold_support_chunk_ids": gold_ids,
                    "top_k": retrieve_for(slug, query),
                }
            )

    # ---- 2. FC1 -> atomic sentence pool ----
    fc1_source_chunk_by_slug = {slug: chunk_id for slug, chunk_id, _ in FC1_CHUNKS}
    fc1_gold_sentences = fc1_gold_support_sentences()
    for slug in PROPERTY_SLUGS:
        source_chunk_id = fc1_source_chunk_by_slug[slug]
        gold_sentence_texts = set(fc1_gold_sentences[source_chunk_id])
        gold_ids = sorted(
            c.chunk_id
            for c in property_data[slug]["atomic_chunks"]
            if c.source_chunk_id == source_chunk_id and c.text in gold_sentence_texts
        )
        # FC1 keeps the same retrieval_query used in v1 (from the checkpoint)
        v1_fc1 = next(
            s
            for p in checkpoint_pairs
            if p["property_slug"] == slug and p["constraint_text"] == "family-friendly"
            for sk, s in p["subclaims"].items()
            if sk == "FC1"
        )
        rows.append(
            {
                "pair_id": f"{slug}__family_friendly",
                "property_slug": slug,
                "constraint_text": "family-friendly",
                "subclaim": "FC1",
                "hypothesis": FC1_HYPOTHESIS,
                "retrieval_query": v1_fc1["retrieval_query"],
                "gold_status": "SUPPORTED" if gold_ids else "NOT_SUPPORTED",
                "gold_support_chunk_ids": gold_ids,
                "top_k": retrieve_for(slug, v1_fc1["retrieval_query"]),
            }
        )

    # ---- 3. everything else carried over unchanged (hypothesis/query/gold),
    #         but retrieval now runs over the ATOMIC pool ----
    for pair in checkpoint_pairs:
        slug = pair["property_slug"]
        constraint = pair["constraint_text"]
        if constraint == "quiet":
            continue  # replaced above
        for subclaim_key, subclaim in pair["subclaims"].items():
            if constraint == "family-friendly" and subclaim_key == "FC1":
                continue  # replaced above
            gold_ids = sorted(e["chunk_id"] for e in subclaim["evidence"] if e["relation"] == "SUPPORT")
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "property_slug": slug,
                    "constraint_text": constraint,
                    "subclaim": subclaim_key,
                    "hypothesis": subclaim["hypothesis"],
                    "retrieval_query": subclaim["retrieval_query"],
                    "gold_status": "SUPPORTED" if gold_ids else "NOT_SUPPORTED",
                    "gold_support_chunk_ids": gold_ids,
                    "top_k": retrieve_for(slug, subclaim["retrieval_query"]),
                }
            )

    n_supported = sum(1 for r in rows if r["gold_status"] == "SUPPORTED")
    print(f"n_subclaims total (v2/atomic schema): {len(rows)}")
    print(f"SUPPORTED: {n_supported}, NOT_SUPPORTED: {len(rows) - n_supported}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} subclaim rows -> {OUTPUT_PATH}")

    # ---- retrieval-only Recall@1 / Recall@3 diagnostic (pre-verification) ----
    recall_1 = recall_at(rows, 1)
    recall_3 = recall_at(rows, 3)
    retrieval_only = {
        "schema": "v2 (atomic: Q1/Q2/Q3 quiet + FC1 sentence-split)",
        "n_subclaims_total": len(rows),
        "n_subclaims_supported": n_supported,
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
    }
    save_json(RETRIEVAL_ONLY_METRICS_PATH, retrieval_only)
    print()
    print("=== RETRIEVAL-ONLY (pre-verification) DIAGNOSTIC ===")
    print(json.dumps(retrieval_only, indent=2))


if __name__ == "__main__":
    main()
