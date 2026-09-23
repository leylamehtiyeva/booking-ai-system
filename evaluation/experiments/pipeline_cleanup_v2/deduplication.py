"""
Deduplicates retrieval candidates by exact evidence text (post heading-
filtering, pre-verification). Keeps the best-ranked (lowest rank /
highest similarity) representative as canonical, and records every
original chunk_id that shared the text so traceability is preserved.
"""

from __future__ import annotations

from typing import Any


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    candidates: list of dicts with at least "text", "rank", "chunk_id".
    Assumes candidates are already ordered by rank ascending (best first).
    """
    by_text: dict[str, dict[str, Any]] = {}

    for cand in candidates:
        text = cand["text"]
        if text not in by_text:
            canonical = dict(cand)
            canonical["duplicate_chunk_ids"] = [cand["chunk_id"]]
            by_text[text] = canonical
        else:
            by_text[text]["duplicate_chunk_ids"].append(cand["chunk_id"])
            # candidates are rank-ascending, so the first occurrence
            # already holds the best rank/score - nothing to update.

    # preserve original best-rank ordering
    return sorted(by_text.values(), key=lambda c: c["rank"])
