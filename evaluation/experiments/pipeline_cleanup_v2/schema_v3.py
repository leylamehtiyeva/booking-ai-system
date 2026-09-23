"""
v3 subclaim schema for the architecture-cleanup end-to-end experiment.

Reuses (imports, does not retype) the gold relation decisions already
made and validated in the NLI-only oracle experiment:
- QUIET_RELABEL / QUIET_HYPOTHESES / QUIET_CHUNKS
- FC1_HYPOTHESIS / FC1_CHUNKS / FC1_SENTENCE_RELATIONS

Only new addition here: retrieval_query strings for Q1/Q2/Q3 (these
never existed before because the earlier oracle experiment never ran
retrieval - it fed gold evidence directly to NLI).

Does not modify retrieval_checkpoint_v2.jsonl or build_input_v2.py.
"""

from __future__ import annotations

from evaluation.experiments.nli_oracle.build_input_v2 import (
    FC1_CHUNKS,
    FC1_HYPOTHESIS,
    FC1_SENTENCE_RELATIONS,
    QUIET_CHUNKS,
    QUIET_HYPOTHESES,
    QUIET_RELABEL,
)

# New: short retrieval queries for the atomic quiet hypotheses -
# mirrors the style of existing retrieval_query strings in
# retrieval_checkpoint_v2.jsonl (short noun-phrase-like, not the full
# formal hypothesis sentence).
QUIET_RETRIEVAL_QUERIES = {
    "Q1": "soundproofing",
    "Q2": "quiet surroundings",
    "Q3": "no noise disturbance",
}

PROPERTY_SLUGS = ["apt_city_center", "nizami_studio", "fermaart_hotel", "modern_apt_nizami"]

# gold_support_chunk_ids per (property_slug, "Q1"/"Q2"/"Q3"), derived
# from QUIET_RELABEL + QUIET_CHUNKS. Properties with zero original
# quiet evidence (nizami_studio) are INCLUDED here with an empty gold
# set - deliberately not excluded, per the false-positive-testing
# principle established in the previous end-to-end experiment.
def quiet_gold_support_chunk_ids() -> dict[tuple[str, str], list[str]]:
    property_by_chunk = {chunk_id: slug for slug, chunk_id, _ in QUIET_CHUNKS}
    result: dict[tuple[str, str], list[str]] = {
        (slug, q): [] for slug in PROPERTY_SLUGS for q in ("Q1", "Q2", "Q3")
    }
    for chunk_id, slug in property_by_chunk.items():
        for q in ("Q1", "Q2", "Q3"):
            if QUIET_RELABEL[(chunk_id, q)] == "SUPPORT":
                result[(slug, q)].append(chunk_id)
    return {k: sorted(v) for k, v in result.items()}


# FC1 gold SUPPORT sentence texts per property (exact text as produced
# by split_into_sentences() - used to label the atomic evidence pool
# built in atomic_chunking.py by exact-text match against the same
# source_chunk_id).
def fc1_gold_support_sentences() -> dict[str, list[str]]:
    """source_chunk_id -> list of exact sentence texts that are SUPPORT."""
    from app.logic.listing_signals import split_into_sentences

    result: dict[str, list[str]] = {}
    for _, chunk_id, text in FC1_CHUNKS:
        sentences = split_into_sentences(text)
        relations = FC1_SENTENCE_RELATIONS[chunk_id]
        result[chunk_id] = [s for s, r in zip(sentences, relations) if r == "SUPPORT"]
    return result


__all__ = [
    "FC1_HYPOTHESIS",
    "QUIET_HYPOTHESES",
    "QUIET_RETRIEVAL_QUERIES",
    "PROPERTY_SLUGS",
    "quiet_gold_support_chunk_ids",
    "fc1_gold_support_sentences",
]
