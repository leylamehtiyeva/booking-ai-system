"""
Small, hand-labeled quantitative checkpoint for the evidence retrieval
prototype - answers "how often does retrieval put the right evidence
in top-k, and how separable are the similarity score distributions?"

Not wired into any production pipeline or CI. Makes real, billed calls
to the Gemini embeddings API on first run; a local cache file avoids
re-billing on subsequent runs of the same texts.

This does NOT tune a production threshold - it only reports Recall@K
and raw score distributions for manual inspection.

Usage:
    python -m evaluation.experiments.evidence_retrieval.run_retrieval_checkpoint
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
CHECKPOINT_PATH = GOLDEN_DIR / "retrieval_checkpoint_v1.jsonl"
CACHE_PATH = GOLDEN_DIR / "embedding_cache.json"

OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/outputs/evidence_retrieval_checkpoint_report.json"
)

K_VALUES = (1, 3, 5)
TOP_K_FOR_REPORT = 5


def load_embedding_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_embedding_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def embed_with_cache(
    texts: list[str],
    cache: dict[str, list[float]],
) -> list[list[float]]:
    """
    Embed texts, reusing cached vectors for texts already seen (keyed
    on the raw text itself, so identical evidence strings that repeat
    across properties or chunks are only billed once).
    """
    missing = [t for t in texts if t not in cache]

    if missing:
        vectors = embed_texts(missing, model=DEFAULT_EMBEDDING_MODEL)
        for text, vector in zip(missing, vectors):
            cache[text] = vector

    return [cache[t] for t in texts]


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"n": 0}

    sorted_scores = sorted(scores)
    return {
        "n": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "sorted_scores": [round(s, 4) for s in sorted_scores],
    }


def run() -> dict[str, Any]:
    pairs = load_jsonl(CHECKPOINT_PATH)
    cache = load_embedding_cache()

    # Load + chunk + embed every property once, reused across all
    # constraint queries for that property.
    property_slugs = sorted({p["property_slug"] for p in pairs})
    property_data: dict[str, dict[str, Any]] = {}

    for slug in property_slugs:
        listing_json = json.loads(
            (PROPERTIES_DIR / f"{slug}.json").read_text(encoding="utf-8")
        )
        listing = ListingRaw.model_validate(listing_json)
        chunks = build_evidence_chunks(listing)
        chunk_vectors = embed_with_cache([c.text for c in chunks], cache)

        property_data[slug] = {
            "chunks": chunks,
            "chunk_by_id": {c.chunk_id: c for c in chunks},
            "chunk_vectors": chunk_vectors,
        }

    # Embed each unique constraint text once.
    unique_constraints = sorted({p["constraint_text"] for p in pairs})
    query_vectors = dict(
        zip(unique_constraints, embed_with_cache(unique_constraints, cache))
    )

    save_embedding_cache(cache)

    recall_hits = {k: 0 for k in K_VALUES}
    evidence_pair_count = 0

    gold_scores: list[float] = []
    irrelevant_topk_with_evidence_scores: list[float] = []
    topk_no_evidence_scores: list[float] = []

    pair_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        slug = pair["property_slug"]
        constraint_text = pair["constraint_text"]
        has_evidence = pair["has_evidence"]
        gold_ids = set(pair["gold_chunk_ids"])

        data = property_data[slug]
        chunks = data["chunks"]
        chunk_vectors = data["chunk_vectors"]

        query_vector = query_vectors[constraint_text]
        similarities = cosine_similarity(query_vector, chunk_vectors)

        ranked = sorted(
            zip(chunks, similarities),
            key=lambda pair_: pair_[1],
            reverse=True,
        )
        top_k = ranked[:TOP_K_FOR_REPORT]

        top_k_ids = [chunk.chunk_id for chunk, _ in top_k]

        if has_evidence:
            evidence_pair_count += 1

            for k in K_VALUES:
                hit = any(cid in gold_ids for cid in top_k_ids[:k])
                if hit:
                    recall_hits[k] += 1

            # Gold-evidence score: computed directly for every gold
            # chunk, independent of whether it made top-k.
            for chunk, score in zip(chunks, similarities):
                if chunk.chunk_id in gold_ids:
                    gold_scores.append(score)

            # Irrelevant top-k: retrieved but not gold, for a pair
            # where real evidence does exist somewhere in the pool.
            for chunk, score in top_k:
                if chunk.chunk_id not in gold_ids:
                    irrelevant_topk_with_evidence_scores.append(score)

            recall_at_5_hit = any(cid in gold_ids for cid in top_k_ids)
            if not recall_at_5_hit:
                errors.append(
                    {
                        "pair_id": pair["pair_id"],
                        "constraint_text": constraint_text,
                        "gold_chunk_ids": sorted(gold_ids),
                        "gold_texts": [
                            data["chunk_by_id"][cid].text
                            for cid in gold_ids
                            if cid in data["chunk_by_id"]
                        ],
                        "top_5_retrieved": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "score": round(score, 4),
                                "text": chunk.text,
                            }
                            for chunk, score in top_k
                        ],
                    }
                )
        else:
            # No genuine evidence exists - every top-k result is noise
            # by construction.
            for chunk, score in top_k:
                topk_no_evidence_scores.append(score)

        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "property_slug": slug,
                "constraint_text": constraint_text,
                "has_evidence": has_evidence,
                "gold_chunk_ids": sorted(gold_ids),
                "top_5": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_type": chunk.source_type,
                        "score": round(score, 4),
                        "is_gold": chunk.chunk_id in gold_ids,
                        "text": chunk.text,
                    }
                    for chunk, score in top_k
                ],
            }
        )

    recall_at_k = {
        f"recall@{k}": round(recall_hits[k] / evidence_pair_count, 4)
        if evidence_pair_count
        else None
        for k in K_VALUES
    }

    report = {
        "n_pairs_total": len(pairs),
        "n_pairs_with_evidence": evidence_pair_count,
        "n_pairs_no_evidence": len(pairs) - evidence_pair_count,
        "recall_at_k": recall_at_k,
        "score_distributions": {
            "gold_evidence": summarize_scores(gold_scores),
            "irrelevant_topk_with_evidence": summarize_scores(
                irrelevant_topk_with_evidence_scores
            ),
            "topk_no_evidence": summarize_scores(topk_no_evidence_scores),
        },
        "errors": errors,
        "n_errors": len(errors),
        "pairs": pair_results,
    }

    save_json(OUTPUT_PATH, report)

    print("=== EVIDENCE RETRIEVAL CHECKPOINT ===")
    print(f"pairs total: {report['n_pairs_total']}")
    print(f"pairs with evidence: {report['n_pairs_with_evidence']}")
    print(f"pairs with no evidence: {report['n_pairs_no_evidence']}")
    print()
    print("Recall@K (over pairs with evidence only):")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall_at_k[f'recall@{k}']}")
    print()
    print("Score distributions:")
    for name, dist in report["score_distributions"].items():
        if dist["n"] == 0:
            print(f"  {name}: n=0")
            continue
        print(
            f"  {name}: n={dist['n']} "
            f"min={dist['min']} max={dist['max']} "
            f"mean={dist['mean']} median={dist['median']} stdev={dist['stdev']}"
        )
    print()
    print(f"Errors (recall@5 misses): {len(errors)}")
    for err in errors:
        print(f"  - {err['pair_id']}: gold not in top-5")
    print()
    print(f"Full report saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()
