from __future__ import annotations

from app.logic.soft_evidence_cleanup import (
    clean_and_route_candidates,
    deduplicate_by_text,
    filter_headings,
    is_heading,
)
from app.logic.soft_evidence_retrieval import RetrievedEvidence
from app.schemas.soft_evidence import ClaimResolutionMethod, EvidenceRelation


def _candidate(text: str, source_type: str, source_path: str | None, score: float) -> RetrievedEvidence:
    return RetrievedEvidence(text=text, source_type=source_type, source_path=source_path, retrieval_score=score)


# ---------------- heading filter ----------------


def test_is_heading_matches_title_and_header_suffix():
    assert is_heading("policies[0].title") is True
    assert is_heading("highlights[0].header") is True
    assert is_heading("listing.description") is False
    assert is_heading(None) is False


def test_filter_headings_removes_heading_paths():
    candidates = [
        _candidate("Children and beds", "policies", "policies[0].title", 0.9),
        _candidate("Genuine content", "description", "listing.description", 0.8),
    ]
    result = filter_headings(candidates)
    assert [c.text for c in result] == ["Genuine content"]


# ---------------- dedup ----------------


def test_dedup_keeps_best_scored_representative():
    candidates = [
        _candidate("Non-smoking throughout", "facilities", "listing.facilities[0].name", 0.9),
        _candidate("Non-smoking throughout", "facilities", "rooms[0].facilities[2].name", 0.7),
    ]
    result = deduplicate_by_text(candidates)
    assert len(result) == 1
    assert result[0].retrieval_score == 0.9
    assert result[0].source_path == "listing.facilities[0].name"
    assert result[0].duplicate_source_paths == ("rooms[0].facilities[2].name",)


def test_dedup_preserves_distinct_texts():
    candidates = [
        _candidate("text a", "description", "listing.description", 0.9),
        _candidate("text b", "description", "listing.description", 0.8),
    ]
    result = deduplicate_by_text(candidates)
    assert len(result) == 2


def test_dedup_empty_list():
    assert deduplicate_by_text([]) == []


# ---------------- ordering: retrieval -> heading filter -> dedup ----------------


def test_processing_order_heading_filter_before_dedup():
    """
    A heading-path duplicate must not survive into the dedup step at
    all - filter_headings runs first.
    """
    candidates = [
        _candidate("Family friendly", "highlights", "highlights[0].header", 0.95),  # heading, dropped
        _candidate("Family friendly", "description", "listing.description", 0.5),   # genuine, kept
    ]
    result = clean_and_route_candidates("FAM1", candidates)
    # only the non-heading occurrence should ever reach routing
    all_texts = [d.text for d in result.deterministic_items] + [c.text for c in result.free_text_candidates]
    assert all_texts == ["Family friendly"]
    if result.free_text_candidates:
        assert result.free_text_candidates[0].source_path == "listing.description"


# ---------------- strict routing integration ----------------


def test_deterministic_support_for_known_alias():
    candidates = [_candidate("Desk", "room_facilities", "rooms[0].facilities[1].name", 0.99)]
    result = clean_and_route_candidates("RW1", candidates)
    assert len(result.deterministic_items) == 1
    assert result.free_text_candidates == []
    item = result.deterministic_items[0]
    assert item.relation == EvidenceRelation.SUPPORT
    assert item.resolution_method == ClaimResolutionMethod.DETERMINISTIC
    assert item.retrieval_score == 0.99  # real retrieval score preserved


def test_deterministic_not_enough_evidence_for_controlled_tag_without_mapping():
    candidates = [_candidate("Air conditioning", "room_facilities", "rooms[0].facilities[1].name", 0.8)]
    result = clean_and_route_candidates("RW1", candidates)
    assert len(result.deterministic_items) == 1
    assert result.deterministic_items[0].relation == EvidenceRelation.NOT_ENOUGH_EVIDENCE
    assert result.free_text_candidates == []


def test_free_text_candidate_routed_to_gemini_bucket():
    candidates = [_candidate("A very quiet street with little traffic noise.", "description", "listing.description", 0.85)]
    result = clean_and_route_candidates("Q2", candidates)
    assert result.deterministic_items == []
    assert len(result.free_text_candidates) == 1
    assert result.free_text_candidates[0].text == "A very quiet street with little traffic noise."


def test_deterministic_candidate_does_not_suppress_separate_free_text_candidate():
    candidates = [
        _candidate("Desk", "room_facilities", "rooms[0].facilities[1].name", 0.99),
        _candidate("There is a small desk-like surface mentioned in reviews.", "review_summary", None, 0.4),
    ]
    result = clean_and_route_candidates("RW1", candidates)
    assert len(result.deterministic_items) == 1
    assert len(result.free_text_candidates) == 1
    assert result.free_text_candidates[0].text == "There is a small desk-like surface mentioned in reviews."


def test_mixed_deterministic_and_free_text_preserve_rank_order_within_free_text_bucket():
    candidates = [
        _candidate("best free text", "description", "listing.description", 0.9),
        _candidate("Desk", "room_facilities", "rooms[0].facilities[1].name", 0.8),
        _candidate("second free text", "highlights", "highlights[0].contents", 0.5),
    ]
    result = clean_and_route_candidates("RW1", candidates)
    assert [c.text for c in result.free_text_candidates] == ["best free text", "second free text"]
