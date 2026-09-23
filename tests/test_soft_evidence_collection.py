from __future__ import annotations

from app.logic.soft_evidence_collection import collect_soft_evidence_pool
from app.schemas.listing import Facility, Highlight, ListingRaw, Policy, Room


def _base_listing(**overrides) -> ListingRaw:
    defaults = dict(
        id="hotel-1",
        name="Test Hotel",
        description="A quiet apartment near the center. Great for families.",
        fine_print="No parties allowed.",
        facilities=[Facility(name="Desk"), Facility(name="Free WiFi", overview="Available in all rooms.")],
        policies=[Policy(title="Child policies", content="Children of any age are welcome.")],
        highlights=[Highlight(header="Great location", contents=["Near the old city"])],
        rooms=[Room(name="Deluxe Room", facilities=[Facility(name="Air conditioning")])],
    )
    defaults.update(overrides)
    return ListingRaw(**defaults)


def test_description_facilities_rooms_policies_highlights_present():
    listing = _base_listing()
    pool = collect_soft_evidence_pool(listing)

    source_types = {c.source_type for c in pool}
    assert "description" in source_types
    assert "facilities" in source_types
    assert "room_facilities" in source_types
    assert "policies" in source_types
    assert "highlights" in source_types

    texts = {c.text for c in pool}
    assert "Desk" in texts
    assert "Children of any age are welcome." in texts
    assert "No parties allowed." in texts  # fine_print -> "other"


def test_fine_print_maps_to_other_source_type():
    listing = _base_listing()
    pool = collect_soft_evidence_pool(listing)
    fine_print_candidates = [c for c in pool if c.text == "No parties allowed."]
    assert len(fine_print_candidates) == 1
    assert fine_print_candidates[0].source_type == "other"
    assert fine_print_candidates[0].source_path == "listing.fine_print"


def test_no_review_summary_when_raw_is_none():
    listing = _base_listing(raw=None)
    pool = collect_soft_evidence_pool(listing)
    assert not any(c.source_type == "review_summary" for c in pool)


def test_no_review_summary_when_raw_has_no_review_summary_key():
    listing = _base_listing(raw={"some_other_field": 1})
    pool = collect_soft_evidence_pool(listing)
    assert not any(c.source_type == "review_summary" for c in pool)


def test_review_summary_pros_and_cons_extracted_with_distinct_source_type():
    listing = _base_listing(
        raw={
            "reviewSummary": {
                "pros": [{"description": "<b>Location</b>: Central setting near Nizami Street.", "numberOfMentions": 18, "sentiment": "POSITIVE"}],
                "cons": [{"description": "<b>Breakfast</b>: Selection is often limited.", "numberOfMentions": 38, "sentiment": "MIXED"}],
            }
        }
    )
    pool = collect_soft_evidence_pool(listing)
    review_candidates = [c for c in pool if c.source_type == "review_summary"]
    assert len(review_candidates) == 2

    texts = {c.text for c in review_candidates}
    assert "Central setting near Nizami Street." in texts
    assert "Selection is often limited." in texts

    paths = {c.source_path for c in review_candidates}
    assert "raw.reviewSummary.pros[0].description" in paths
    assert "raw.reviewSummary.cons[0].description" in paths


def test_html_category_prefix_is_stripped():
    listing = _base_listing(
        raw={"reviewSummary": {"pros": [{"description": "<b>Cleanliness</b>: Spotless and well-maintained.", "sentiment": "POSITIVE"}], "cons": []}}
    )
    pool = collect_soft_evidence_pool(listing)
    review_candidates = [c for c in pool if c.source_type == "review_summary"]
    assert review_candidates[0].text == "Spotless and well-maintained."
    assert "<b>" not in review_candidates[0].text
    assert "</b>" not in review_candidates[0].text


def test_review_summary_missing_pros_or_cons_key_handled_safely():
    listing = _base_listing(raw={"reviewSummary": {"pros": [{"description": "<b>Staff</b>: Friendly team."}]}})
    pool = collect_soft_evidence_pool(listing)
    review_candidates = [c for c in pool if c.source_type == "review_summary"]
    assert len(review_candidates) == 1
    assert review_candidates[0].text == "Friendly team."


def test_review_summary_malformed_entries_are_skipped_not_raised():
    listing = _base_listing(
        raw={
            "reviewSummary": {
                "pros": ["not a dict", {"description": None}, {"no_description_key": True}, {"description": "<b>X</b>: OK text."}],
                "cons": None,
            }
        }
    )
    pool = collect_soft_evidence_pool(listing)
    review_candidates = [c for c in pool if c.source_type == "review_summary"]
    assert len(review_candidates) == 1
    assert review_candidates[0].text == "OK text."


def test_sentiment_and_number_of_mentions_are_not_used_for_relation():
    """
    collect_soft_evidence_pool only ever returns (text, source_type,
    source_path) - sentiment/numberOfMentions/pros-vs-cons bucket never
    leak into anything relation-like; SoftEvidenceCandidate simply has
    no field for them.
    """
    listing = _base_listing(
        raw={"reviewSummary": {"pros": [{"description": "<b>X</b>: text", "sentiment": "POSITIVE", "numberOfMentions": 99}], "cons": []}}
    )
    pool = collect_soft_evidence_pool(listing)
    candidate = next(c for c in pool if c.source_type == "review_summary")
    assert not hasattr(candidate, "sentiment")
    assert not hasattr(candidate, "relation")


def test_empty_listing_produces_empty_pool():
    listing = ListingRaw(id="empty")
    pool = collect_soft_evidence_pool(listing)
    assert pool == []
