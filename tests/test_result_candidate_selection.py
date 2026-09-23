from __future__ import annotations

from app.logic.result_candidate_selection import select_shown_candidates
from app.schemas.conversation_route import ResultScope
from app.schemas.listing import ListingRaw
from app.schemas.shown_result_set import ShownResult, ShownResultSet


def make_shown_result_set(*listings: tuple[str, str]) -> ShownResultSet:
    return ShownResultSet(
        result_set_id="rs-test",
        items=[
            ShownResult(
                result_id=result_id,
                listing=ListingRaw(
                    id=result_id,
                    name=title,
                    city="Baku",
                    description=f"Description for {title}.",
                    facilities=[{"name": "Free parking"}],
                    policies=[{"title": "Pets", "content": "Pets allowed."}],
                ),
            )
            for result_id, title in listings
        ],
    )


def test_current_results_returns_all_shown_items_in_order():
    shown = make_shown_result_set(("H1", "Hilton Baku"), ("H2", "Marriott Baku"))

    result = select_shown_candidates(
        result_scope=ResultScope.CURRENT_RESULTS,
        target_result_id=None,
        shown_result_set=shown,
    )

    assert result.status == "ok"
    assert [c.result_id for c in result.candidates] == ["H1", "H2"]
    assert [c.listing.name for c in result.candidates] == [
        "Hilton Baku",
        "Marriott Baku",
    ]


def test_specific_result_returns_exactly_the_validated_target():
    shown = make_shown_result_set(("H1", "Hilton Baku"), ("H2", "Marriott Baku"))

    result = select_shown_candidates(
        result_scope=ResultScope.SPECIFIC_RESULT,
        target_result_id="H2",
        shown_result_set=shown,
    )

    assert result.status == "ok"
    assert len(result.candidates) == 1
    assert result.candidates[0].result_id == "H2"
    assert result.candidates[0].listing.name == "Marriott Baku"


def test_current_results_with_empty_shown_set_does_not_execute():
    shown = ShownResultSet(result_set_id="rs-empty", items=[])

    result = select_shown_candidates(
        result_scope=ResultScope.CURRENT_RESULTS,
        target_result_id=None,
        shown_result_set=shown,
    )

    assert result.status == "no_shown_results"
    assert result.candidates == []


def test_current_results_with_no_shown_result_set_at_all():
    result = select_shown_candidates(
        result_scope=ResultScope.CURRENT_RESULTS,
        target_result_id=None,
        shown_result_set=None,
    )

    assert result.status == "no_shown_results"
    assert result.candidates == []


def test_specific_result_with_missing_target_does_not_pick_first():
    shown = make_shown_result_set(("H1", "Hilton Baku"), ("H2", "Marriott Baku"))

    result = select_shown_candidates(
        result_scope=ResultScope.SPECIFIC_RESULT,
        target_result_id=None,
        shown_result_set=shown,
    )

    assert result.status == "invalid_target"
    assert result.candidates == []


def test_specific_result_with_unknown_target_does_not_pick_nearest():
    shown = make_shown_result_set(("H1", "Hilton Baku"), ("H2", "Marriott Baku"))

    result = select_shown_candidates(
        result_scope=ResultScope.SPECIFIC_RESULT,
        target_result_id="H999",
        shown_result_set=shown,
    )

    assert result.status == "invalid_target"
    assert result.candidates == []


def test_selected_candidates_preserve_full_listing_raw():
    shown = make_shown_result_set(("H1", "Hilton Baku"))

    result = select_shown_candidates(
        result_scope=ResultScope.CURRENT_RESULTS,
        target_result_id=None,
        shown_result_set=shown,
    )

    listing = result.candidates[0].listing
    assert listing.facilities
    assert listing.policies
    assert listing.description
