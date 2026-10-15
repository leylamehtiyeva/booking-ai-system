"""
M2 (baseline case 0049): an optional (NICE) requirement that nothing checks
- no canonical field, and the textual fallback runs only for must/forbidden
- used to vanish from the result. It must be shown as "Not confirmed
(optional)" without changing tier, ranking, eligibility, LLM calls or any
existing check result.

Real fixture retrieval, structured matching, ranking, selection and
normalization; the textual fallback LLM is a counting fake that must never
be called here.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from app.logic import constraint_evidence_resolution
from app.logic.build_answer_payload import build_answer_payload
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.fields import Field
from app.schemas.query import SearchRequest
from app.tools.orchestrate_search_tool import orchestrate_search_request

OPTIONAL_REASON = "Not confirmed (optional)"


def _known(text: str, field: Field, priority=ConstraintPriority.MUST, category=ConstraintCategory.AMENITY) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=category,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _textual(text: str, priority=ConstraintPriority.NICE) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


WIFI = _known("wifi", Field.WIFI)
LOCAL_FEEL = _textual("local and less touristy feel")


@pytest.fixture
def fallback_calls(monkeypatch):
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("no textual fallback call expected")

    monkeypatch.setattr(constraint_evidence_resolution, "_gemini_client", lambda: SimpleNamespace(models=Models()))
    return calls


async def _search(*constraints: UserConstraint):
    request = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 10),
        check_out=date(2026, 4, 15),
        property_types=["apartment"],
        constraints=list(constraints),
    )
    response, _ = await orchestrate_search_request(
        request,
        result_limit=5,
        candidate_pool_size=40,
        fallback_policy=FallbackPolicy(enabled=True, top_k=5),  # production policy
    )
    return response


def _statuses(result, bucket: str) -> list[tuple[str, str | None]]:
    return [(s.name, s.reason) for s in getattr(result, bucket)]


async def test_unchecked_optional_requirement_is_shown_as_not_confirmed(fallback_calls):
    response = await _search(WIFI, LOCAL_FEEL)

    assert response.results
    for result in response.results:
        assert ("local and less touristy feel", OPTIONAL_REASON) in _statuses(result, "uncertain_constraints")
        assert ("local and less touristy feel", OPTIONAL_REASON) in _statuses(result, "uncertain_requested_constraints")
    assert fallback_calls == []


async def test_tier_ranking_eligibility_and_scores_are_unchanged(fallback_calls):
    without = await _search(WIFI)
    with_optional = await _search(WIFI, LOCAL_FEEL)

    def decisions(response):
        return [(r.result_id, r.match_tier, r.eligibility_status, r.score, r.selection_reasons) for r in response.results]

    assert decisions(with_optional) == decisions(without)
    assert {r.match_tier for r in with_optional.results} == {"strong"}


async def test_checked_optional_requirement_is_not_duplicated(fallback_calls):
    """A NICE requirement with a canonical field already has a structured result."""
    balcony = _known("balcony", Field.BALCONY, priority=ConstraintPriority.NICE)

    response = await _search(WIFI, balcony)

    for result in response.results:
        all_names = [s.name for s in (*result.matched_constraints, *result.uncertain_constraints, *result.failed_constraints)]
        assert all_names.count("balcony") == 1
        assert OPTIONAL_REASON not in [s.reason for s in result.uncertain_constraints if s.name == "balcony"]


async def test_existing_check_results_are_not_overwritten(fallback_calls):
    """A failed optional requirement stays 'not satisfied'."""
    smoking = _known("smoking allowed", Field.SMOKING_ALLOWED, priority=ConstraintPriority.NICE, category=ConstraintCategory.POLICY)

    response = await _search(WIFI, smoking)

    failed_somewhere = [r for r in response.results if "smoking allowed" in [s.name for s in r.failed_constraints]]
    assert failed_somewhere
    for result in failed_somewhere:
        assert "smoking allowed" not in [s.name for s in result.uncertain_constraints]


async def test_unconfirmed_must_keeps_its_own_reason(fallback_calls, monkeypatch):
    """Only NICE requirements get the optional label; a MUST one is never relabelled."""
    no_fallback = FallbackPolicy(enabled=False)
    request = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 10),
        check_out=date(2026, 4, 15),
        constraints=[WIFI, _textual("romantic atmosphere", priority=ConstraintPriority.MUST)],
    )
    response, _ = await orchestrate_search_request(request, result_limit=5, candidate_pool_size=40, fallback_policy=no_fallback)

    for result in response.results:
        reasons = [s.reason for s in result.uncertain_constraints if s.name == "romantic atmosphere"]
        assert reasons and OPTIONAL_REASON not in reasons


async def test_card_sections_keep_confirmed_not_confirmed_and_not_satisfied_apart(fallback_calls):
    smoking = _known("smoking allowed", Field.SMOKING_ALLOWED, priority=ConstraintPriority.NICE, category=ConstraintCategory.POLICY)
    response = await _search(WIFI, LOCAL_FEEL, smoking)

    payload = build_answer_payload(response, top_k=len(response.results))
    for item in payload["top_results"]:
        sections = item["answer_explanation"]
        confirmed = {p["name"] for p in sections["confirmed"]}
        needs_confirmation = {p["name"]: p["reason"] for p in sections["needs_confirmation"]}
        not_satisfied = {p["name"] for p in sections["not_satisfied"]}

        assert "wifi" in confirmed
        assert needs_confirmation["local and less touristy feel"] == OPTIONAL_REASON
        assert not ({"local and less touristy feel"} & (confirmed | not_satisfied))
        assert not_satisfied <= {"smoking allowed"}
