"""
Dedicated evidence collector for the new soft-evidence pipeline.

Deliberately a NEW module, not a modification of
app.logic.listing_signals.collect_listing_signals (the existing
production fallback's collector) - reusing it read-only avoids any risk
of changing the old fallback's behavior, per the migration principle
that the old fallback must not be touched.

Adds one new evidence family collect_listing_signals does not cover at
all: listing.raw["reviewSummary"].pros / .cons, tagged with a distinct
source_type="review_summary" (never folded into "description").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.logic.listing_signals import collect_listing_signals
from app.schemas.listing import ListingRaw

# Mirrors app.logic.constraint_evidence_resolution._source_from_path /
# evaluation/experiments/evidence_retrieval/chunking.py's
# _source_type_from_path. Duplicated on purpose (not imported): both of
# those are private helpers of other modules this pipeline is meant to
# stay isolated from - same small-duplication precedent already
# established twice in this codebase.
_SOURCE_TYPE_PATH_PREFIXES: tuple[tuple[str, str], ...] = (
    ("listing.facilities", "facilities"),
    ("rooms[", "room_facilities"),
    ("policies[", "policies"),
    ("highlights[", "highlights"),
    ("listing.description", "description"),
    ("listing.fine_print", "other"),
    ("listing.name", "title"),
    ("listing.property_type", "property_type"),
)

_HTML_CATEGORY_PREFIX = re.compile(r"^\s*<b>.*?</b>\s*:\s*", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SoftEvidenceCandidate:
    """One retrievable unit of evidence for the new pipeline."""

    text: str
    source_type: str
    source_path: str | None


def _source_type_from_path(path: str) -> str:
    for prefix, source_type in _SOURCE_TYPE_PATH_PREFIXES:
        if path.startswith(prefix):
            return source_type
    return "other"


def _strip_review_summary_markup(text: str) -> str:
    """
    Real reviewSummary descriptions look like:
        "<b>Location</b>: Central setting near Nizami Street..."
    Strips the leading "<b>Category</b>: " prefix, then any remaining
    stray HTML tags as a defensive fallback (none observed in real
    payloads beyond the leading category tag, but cheap to guard).
    """
    text = _HTML_CATEGORY_PREFIX.sub("", text)
    text = _HTML_TAG.sub("", text)
    return text.strip()


def _collect_review_summary_candidates(listing: ListingRaw) -> list[SoftEvidenceCandidate]:
    """
    Reads listing.raw["reviewSummary"] defensively - listing.raw can be
    None, reviewSummary may be absent, and pros/cons may be missing or
    empty. Real payloads (evaluation/experiments/verifier_comparison/
    holdout_properties/*.json) confirm this exact shape:
        {"pros": [{"description": "...", "numberOfMentions": int,
                    "sentiment": "POSITIVE"|"MIXED"|"NEGATIVE"}, ...],
         "cons": [...same shape...]}

    sentiment / numberOfMentions / pros-vs-cons bucket are NOT used to
    infer SUPPORT/CONTRADICT here - they remain unused metadata; the
    semantic relation comes only from routing/verifier downstream.
    """
    raw = listing.raw
    if not isinstance(raw, dict):
        return []

    review_summary = raw.get("reviewSummary")
    if not isinstance(review_summary, dict):
        return []

    candidates: list[SoftEvidenceCandidate] = []

    for bucket in ("pros", "cons"):
        entries = review_summary.get(bucket)
        if not isinstance(entries, list):
            continue

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            raw_description = entry.get("description")
            if not isinstance(raw_description, str):
                continue

            text = _strip_review_summary_markup(raw_description)
            if not text:
                continue

            candidates.append(
                SoftEvidenceCandidate(
                    text=text,
                    source_type="review_summary",
                    source_path=f"raw.reviewSummary.{bucket}[{index}].description",
                )
            )

    return candidates


def collect_soft_evidence_pool(listing: ListingRaw) -> list[SoftEvidenceCandidate]:
    """
    Full evidence pool for one listing: everything
    collect_listing_signals already extracts (description, facilities,
    room facilities, policies, highlights, room/options text, bed
    types, fine print via "other") plus reviewSummary.pros/cons.

    No item limit and no truncation - the full pool is embedded
    downstream in batches (app.logic.soft_evidence_retrieval).
    """
    candidates: list[SoftEvidenceCandidate] = []

    for signal in collect_listing_signals(listing):
        text = (signal.raw_text or signal.text or "").strip()
        if not text:
            continue
        candidates.append(
            SoftEvidenceCandidate(
                text=text,
                source_type=_source_type_from_path(signal.path),
                source_path=signal.path,
            )
        )

    candidates.extend(_collect_review_summary_candidates(listing))

    return candidates
