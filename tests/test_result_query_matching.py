from __future__ import annotations

import pytest

from app.logic import constraint_evidence_resolution
from app.logic.result_candidate_selection import CandidateSelectionResult
from app.logic.result_query_matching import match_result_query_against_candidates
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import EvidenceSource, Ternary
from app.schemas.result_query import ResultQuery
from app.schemas.shown_result_set import ShownResult

pytestmark = pytest.mark.asyncio

_FALLBACK_DISABLED = FallbackPolicy(enabled=False)


def _constraint(field: Field, text: str = "test") -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _unresolved_constraint(text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.NONE,
    )


def _shown_result(result_id: str, **listing_kwargs) -> ShownResult:
    return ShownResult(
        result_id=result_id,
        listing=ListingRaw(id=result_id, name="Test Listing", **listing_kwargs),
    )


def _ok_selection(*candidates: ShownResult) -> CandidateSelectionResult:
    return CandidateSelectionResult(candidates=list(candidates), status="ok")


def _fake_fallback(decision: str, *, snippet: str = "Fallback evidence.", source: str = "description", path=None):
    """
    Mocks resolve_constraint_via_textual_evidence (the actual LLM call),
    not resolve_listing_constraints_with_fallback itself - this way the
    REAL is_constraint_fallback_eligible/eligibility logic still runs, so
    tests prove the real code decides not to call the LLM for
    structured YES/NO, rather than a hand-rolled stand-in re-asserting
    the test's own assumption.
    """
    calls: list[str] = []

    async def _fake(req, *, model=None, trace=None):
        calls.append(req.constraint_id)
        return constraint_evidence_resolution.ConstraintResolutionResult(
            listing_id=req.listing_id,
            listing_title=req.listing_title,
            constraint_id=req.constraint_id,
            raw_text=req.raw_text,
            normalized_text=req.normalized_text,
            resolver_type="textual",
            priority=req.priority,
            mapped_fields=req.mapped_fields,
            decision=decision,
            resolution_status={"YES": "matched", "NO": "failed", "UNCERTAIN": "uncertain"}[decision],
            confidence=0.8,
            reason="Fallback decided it.",
            evidence=[
                constraint_evidence_resolution.ConstraintEvidence(
                    snippet=snippet, source=source, path=path
                )
            ],
        )

    return _fake, calls


# --- A. CURRENT_RESULTS / parking (fallback disabled - pure structured) --


async def test_current_results_parking_yes_no_uncertain_preserved_in_order():
    h1 = _shown_result("H1", facilities=[{"name": "Private parking"}])
    h2 = _shown_result("H2", description="No parking available on site.")
    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h1, h2, h3),
        fallback_policy=_FALLBACK_DISABLED,
    )

    assert outcome.status == "ok"
    assert [r.result_id for r in outcome.results] == ["H1", "H2", "H3"]

    values = {r.result_id: r.constraint_results[0].value for r in outcome.results}
    assert values == {
        "H1": Ternary.YES,
        "H2": Ternary.NO,
        "H3": Ternary.UNCERTAIN,
    }


# --- B. SPECIFIC_RESULT / balcony ----------------------------------------


async def test_specific_result_balcony_only_returns_that_candidate():
    h2 = _shown_result(
        "H2", description="The room has a private balcony with a nice view."
    )

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.BALCONY, "balcony")]),
        candidate_selection=_ok_selection(h2),
        fallback_policy=_FALLBACK_DISABLED,
    )

    assert outcome.status == "ok"
    assert len(outcome.results) == 1
    assert outcome.results[0].result_id == "H2"
    assert outcome.results[0].constraint_results[0].value == Ternary.YES


# --- C. multiple constraints ---------------------------------------------


async def test_multiple_constraints_kept_separate_per_candidate():
    h1 = _shown_result(
        "H1",
        facilities=[{"name": "Private parking"}, {"name": "Free WiFi"}],
    )
    h2 = _shown_result(
        "H2",
        description="No parking available on site. Free WiFi included.",
    )

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(
            constraints=[
                _constraint(Field.PARKING, "parking"),
                _constraint(Field.WIFI, "wifi"),
            ]
        ),
        candidate_selection=_ok_selection(h1, h2),
        fallback_policy=_FALLBACK_DISABLED,
    )

    assert outcome.status == "ok"

    by_id = {r.result_id: r for r in outcome.results}
    h1_by_field = {cr.field: cr.value for cr in by_id["H1"].constraint_results}
    h2_by_field = {cr.field: cr.value for cr in by_id["H2"].constraint_results}

    assert h1_by_field == {Field.PARKING: Ternary.YES, Field.WIFI: Ternary.YES}
    assert h2_by_field == {Field.PARKING: Ternary.NO, Field.WIFI: Ternary.YES}


# --- D. missing evidence, fallback disabled -------------------------------


async def test_missing_evidence_is_uncertain_not_no():
    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h3),
        fallback_policy=_FALLBACK_DISABLED,
    )

    value = outcome.results[0].constraint_results[0].value
    assert value == Ternary.UNCERTAIN
    assert value != Ternary.NO


# --- E. unresolved constraint, fallback disabled --------------------------


async def test_unresolved_constraint_without_fallback_is_uncertain_without_crash():
    h1 = _shown_result("H1", description="A charming romantic hideaway.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_unresolved_constraint("romantic")]),
        candidate_selection=_ok_selection(h1),
        fallback_policy=_FALLBACK_DISABLED,
    )

    assert outcome.status == "ok"
    cr = outcome.results[0].constraint_results[0]
    assert cr.value == Ternary.UNCERTAIN
    assert cr.field is None
    assert cr.reason == "unresolved_constraint"


# --- F. zero constraints ---------------------------------------------------


async def test_zero_constraints_does_not_run_matcher():
    h1 = _shown_result("H1", facilities=[{"name": "Private parking"}])

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[]),
        candidate_selection=_ok_selection(h1),
    )

    assert outcome.status == "no_constraints"
    assert outcome.results == []


# --- guard: candidate-selection failure is never re-derived --------------


async def test_failed_candidate_selection_short_circuits_before_matching():
    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=CandidateSelectionResult(candidates=[], status="invalid_target"),
    )

    assert outcome.status == "candidate_selection_failed"
    assert outcome.results == []


# --- structured YES/NO never call the fallback LLM ------------------------


async def test_structured_yes_does_not_call_fallback(monkeypatch):
    fake, calls = _fake_fallback("NO")  # would be a dangerous flip if it were ever called
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h1 = _shown_result("H1", facilities=[{"name": "Private parking"}])

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h1),
    )

    assert outcome.results[0].constraint_results[0].value == Ternary.YES
    assert calls == []


async def test_structured_no_does_not_call_fallback(monkeypatch):
    fake, calls = _fake_fallback("YES")  # would be a dangerous flip if it were ever called
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h2 = _shown_result("H2", description="No parking available on site.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h2),
    )

    assert outcome.results[0].constraint_results[0].value == Ternary.NO
    assert calls == []


# --- structured UNCERTAIN -> textual fallback ------------------------------


async def test_structured_uncertain_fallback_yes_becomes_final_yes(monkeypatch):
    fake, calls = _fake_fallback("YES", snippet="Free parking is available on request.", path="listing.fine_print")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h3),
    )

    cr = outcome.results[0].constraint_results[0]
    assert cr.value == Ternary.YES
    assert cr.reason == "fallback_resolved"
    assert len(calls) == 1
    assert len(cr.evidence) == 1
    assert cr.evidence[0].source == EvidenceSource.LLM_FALLBACK
    assert cr.evidence[0].path == "listing.fine_print"
    assert cr.evidence[0].snippet == "Free parking is available on request."


async def test_structured_uncertain_fallback_no_becomes_final_no(monkeypatch):
    fake, calls = _fake_fallback("NO")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h3),
    )

    assert outcome.results[0].constraint_results[0].value == Ternary.NO
    assert len(calls) == 1


async def test_structured_uncertain_fallback_uncertain_stays_uncertain(monkeypatch):
    fake, calls = _fake_fallback("UNCERTAIN")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h3),
    )

    assert outcome.results[0].constraint_results[0].value == Ternary.UNCERTAIN
    assert len(calls) == 1


# --- unresolved constraint -> fallback -------------------------------------


async def test_unresolved_constraint_goes_through_fallback(monkeypatch):
    fake, calls = _fake_fallback("YES", snippet="Absolutely a romantic hideaway.")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h1 = _shown_result("H1", description="A charming romantic hideaway with candlelight dinners.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_unresolved_constraint("romantic")]),
        candidate_selection=_ok_selection(h1),
    )

    cr = outcome.results[0].constraint_results[0]
    assert cr.value == Ternary.YES
    assert cr.reason == "fallback_resolved"
    assert len(calls) == 1


# --- multiple candidates: fallback only where eligible ---------------------


async def test_multiple_candidates_fallback_only_runs_for_eligible_candidate(monkeypatch):
    fake, calls = _fake_fallback("YES")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h1 = _shown_result("H1", facilities=[{"name": "Private parking"}])  # structured YES
    h2 = _shown_result("H2", description="A cozy apartment near the center.")  # UNCERTAIN -> fallback
    h3 = _shown_result("H3", description="No parking available on site.")  # structured NO

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h1, h2, h3),
    )

    by_id = {r.result_id: r.constraint_results[0].value for r in outcome.results}
    assert by_id == {"H1": Ternary.YES, "H2": Ternary.YES, "H3": Ternary.NO}
    assert len(calls) == 1


# --- multiple constraints: independent fallback per constraint ------------


async def test_multiple_constraints_independent_fallback_resolution(monkeypatch):
    fake, calls = _fake_fallback("YES")
    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", fake)

    h1 = _shown_result(
        "H1",
        facilities=[{"name": "Private parking"}],  # parking -> structured YES
        description="A cozy apartment near the center.",  # wifi -> UNCERTAIN -> fallback
    )

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(
            constraints=[
                _constraint(Field.PARKING, "parking"),
                _constraint(Field.WIFI, "wifi"),
            ]
        ),
        candidate_selection=_ok_selection(h1),
    )

    by_field = {cr.field: cr.value for cr in outcome.results[0].constraint_results}
    assert by_field == {Field.PARKING: Ternary.YES, Field.WIFI: Ternary.YES}
    assert len(calls) == 1


# --- fallback technical failure --------------------------------------------


async def test_fallback_technical_failure_keeps_structured_uncertain(monkeypatch):
    async def _raise(req, *, model=None, trace=None):
        raise RuntimeError("Gemini API unavailable")

    monkeypatch.setattr(constraint_evidence_resolution, "resolve_constraint_via_textual_evidence", _raise)

    h3 = _shown_result("H3", description="A cozy apartment near the center.")

    outcome = await match_result_query_against_candidates(
        result_query=ResultQuery(constraints=[_constraint(Field.PARKING, "parking")]),
        candidate_selection=_ok_selection(h3),
    )

    assert outcome.status == "ok"
    cr = outcome.results[0].constraint_results[0]
    assert cr.value == Ternary.UNCERTAIN
    assert cr.value != Ternary.NO
    assert cr.value != Ternary.YES
