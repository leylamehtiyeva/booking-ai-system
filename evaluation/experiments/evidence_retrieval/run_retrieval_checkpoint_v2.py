"""
Subclaim-level quantitative checkpoint for the evidence retrieval
prototype (v2 of the checkpoint - see run_retrieval_checkpoint.py for
the earlier, coarser property-level version).

Gold labels here are per atomic subclaim (evidence_by_subclaim schema),
each chunk annotated with an explicit relation (SUPPORT / CONTRADICT /
RELEVANT_NEUTRAL) assigned purely from the meaning of the text, not
from any retrieval output. Recall@K is computed per (property,
subclaim) pair, not per (property, constraint) - a compound constraint
like "good for remote work" is scored as several independent recall
targets (RW1, RW2a, RW2b), not one.

Not wired into any production pipeline or CI. Makes real, billed calls
to the Gemini embeddings API on first run; the shared cache file
(golden/embedding_cache.json) avoids re-billing on subsequent runs.

Usage:
    python -m evaluation.experiments.evidence_retrieval.run_retrieval_checkpoint_v2
"""

from __future__ import annotations

import json
import statistics
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
from evaluation.experiments.evidence_retrieval.chunking import build_evidence_chunks
from evaluation.experiments.evidence_retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_texts,
)
from evaluation.experiments.evidence_retrieval.similarity import cosine_similarity

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
PROPERTIES_DIR = GOLDEN_DIR / "properties"
CHECKPOINT_PATH = GOLDEN_DIR / "retrieval_checkpoint_v2.jsonl"
CACHE_PATH = GOLDEN_DIR / "embedding_cache.json"

OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/outputs/evidence_retrieval_checkpoint_v2_report.json"
)

K_VALUES = (1, 3, 5)
TOP_K_FOR_REPORT = 5


def load_embedding_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_embedding_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def embed_with_cache(texts: list[str], cache: dict[str, list[float]]) -> list[list[float]]:
    missing = [t for t in texts if t not in cache]
    if missing:
        vectors = embed_texts(missing, model=DEFAULT_EMBEDDING_MODEL)
        for text, vector in zip(missing, vectors):
            cache[text] = vector
    return [cache[t] for t in texts]


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "sorted_scores": [round(s, 4) for s in sorted(scores)],
    }


def run() -> dict[str, Any]:
    pairs = load_jsonl(CHECKPOINT_PATH)
    cache = load_embedding_cache()

    property_slugs = sorted({p["property_slug"] for p in pairs})
    property_data: dict[str, dict[str, Any]] = {}

    for slug in property_slugs:
        listing_json = json.loads((PROPERTIES_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        listing = ListingRaw.model_validate(listing_json)
        chunks = build_evidence_chunks(listing)
        chunk_vectors = embed_with_cache([c.text for c in chunks], cache)
        property_data[slug] = {
            "chunks": chunks,
            "chunk_by_id": {c.chunk_id: c for c in chunks},
            "chunk_vectors": chunk_vectors,
        }

    # Every distinct retrieval_query across all subclaims, embedded once.
    unique_queries = sorted(
        {
            subclaim["retrieval_query"]
            for pair in pairs
            for subclaim in pair["subclaims"].values()
        }
    )
    query_vectors = dict(zip(unique_queries, embed_with_cache(unique_queries, cache)))

    save_embedding_cache(cache)

    recall_hits = {k: 0 for k in K_VALUES}
    n_supported = 0

    gold_scores: list[float] = []
    irrelevant_topk_with_evidence_scores: list[float] = []
    topk_no_evidence_scores: list[float] = []

    subclaim_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        slug = pair["property_slug"]
        data = property_data[slug]
        chunks = data["chunks"]
        chunk_vectors = data["chunk_vectors"]

        for subclaim_key, subclaim in pair["subclaims"].items():
            status = subclaim["status"]
            support_ids = {
                e["chunk_id"] for e in subclaim["evidence"] if e["relation"] == "SUPPORT"
            }
            all_annotated_ids = {e["chunk_id"] for e in subclaim["evidence"]}

            query_vector = query_vectors[subclaim["retrieval_query"]]
            similarities = cosine_similarity(query_vector, chunk_vectors)

            ranked = sorted(zip(chunks, similarities), key=lambda p: p[1], reverse=True)
            top_k = ranked[:TOP_K_FOR_REPORT]
            top_k_ids = [chunk.chunk_id for chunk, _ in top_k]

            row = {
                "pair_id": pair["pair_id"],
                "property_slug": slug,
                "constraint_text": pair["constraint_text"],
                "subclaim": subclaim_key,
                "hypothesis": subclaim["hypothesis"],
                "retrieval_query": subclaim["retrieval_query"],
                "status": status,
                "support_chunk_ids": sorted(support_ids),
                "top_5": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_type": chunk.source_type,
                        "score": round(score, 4),
                        "is_support": chunk.chunk_id in support_ids,
                        "text": chunk.text,
                    }
                    for chunk, score in top_k
                ],
            }

            if status == "SUPPORTED":
                n_supported += 1

                for k in K_VALUES:
                    if any(cid in support_ids for cid in top_k_ids[:k]):
                        recall_hits[k] += 1

                for chunk, score in zip(chunks, similarities):
                    if chunk.chunk_id in support_ids:
                        gold_scores.append(score)

                for chunk, score in top_k:
                    if chunk.chunk_id not in support_ids:
                        irrelevant_topk_with_evidence_scores.append(score)

                recall_at_5_hit = any(cid in support_ids for cid in top_k_ids)
                row["recall_at_5_hit"] = recall_at_5_hit

                if not recall_at_5_hit:
                    errors.append(
                        {
                            "pair_id": pair["pair_id"],
                            "subclaim": subclaim_key,
                            "hypothesis": subclaim["hypothesis"],
                            "support_chunk_ids": sorted(support_ids),
                            "support_texts": [
                                data["chunk_by_id"][cid].text
                                for cid in support_ids
                                if cid in data["chunk_by_id"]
                            ],
                            "top_5_retrieved": row["top_5"],
                        }
                    )
            else:
                # NO_EVIDENCE (or, in the future, CONTRADICTED with no
                # SUPPORT chunks): every top-k result here is noise
                # relative to this specific atomic hypothesis.
                for chunk, score in top_k:
                    topk_no_evidence_scores.append(score)

            subclaim_results.append(row)

    recall_at_k = {
        f"recall@{k}": round(recall_hits[k] / n_supported, 4) if n_supported else None
        for k in K_VALUES
    }

    report = {
        "n_subclaims_total": len(subclaim_results),
        "n_subclaims_supported": n_supported,
        "n_subclaims_no_evidence": len(subclaim_results) - n_supported,
        "recall_at_k": recall_at_k,
        "score_distributions": {
            "gold_evidence": summarize_scores(gold_scores),
            "irrelevant_topk_with_evidence": summarize_scores(irrelevant_topk_with_evidence_scores),
            "topk_no_evidence": summarize_scores(topk_no_evidence_scores),
        },
        "errors": errors,
        "n_errors": len(errors),
        "subclaims": subclaim_results,
    }

    save_json(OUTPUT_PATH, report)

    print("=== EVIDENCE RETRIEVAL CHECKPOINT v2 (per-subclaim) ===")
    print(f"subclaims total: {report['n_subclaims_total']}")
    print(f"subclaims SUPPORTED: {report['n_subclaims_supported']}")
    print(f"subclaims NO_EVIDENCE: {report['n_subclaims_no_evidence']}")
    print()
    print("Recall@K (over SUPPORTED subclaims only):")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall_at_k[f'recall@{k}']}")
    print()
    print("Score distributions:")
    for name, dist in report["score_distributions"].items():
        if dist["n"] == 0:
            print(f"  {name}: n=0")
            continue
        print(
            f"  {name}: n={dist['n']} min={dist['min']} max={dist['max']} "
            f"mean={dist['mean']} median={dist['median']} stdev={dist['stdev']}"
        )
    print()
    print(f"Errors (recall@5 misses): {len(errors)}")
    for err in errors:
        print(f"  - {err['pair_id']} / {err['subclaim']}")
    print()
    print(f"Full report saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()
