from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.agents.factual_question_agent import build_factual_question_agent
from app.logic import factual_question_extraction
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
)
from app.schemas.result_query import ResultQuery


def test_result_query_is_not_search_request_or_patch():
    """
    ResultQuery must carry only constraints - no scope, no
    target_result_id, no SearchRequest-shaped fields. This is the
    structural guarantee that it can never be mistaken for (or duck-typed
    into) SearchRequest/SearchIntentPatch.
    """
    rq = ResultQuery()
    assert rq.constraints == []
    assert set(ResultQuery.model_fields.keys()) == {"constraints"}


def test_result_query_coerces_none_constraints_to_empty_list():
    rq = ResultQuery.model_validate({"constraints": None})
    assert rq.constraints == []


def test_result_query_holds_real_user_constraints():
    rq = ResultQuery.model_validate(
        {
            "constraints": [
                {
                    "raw_text": "parking",
                    "normalized_text": "parking",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["parking"],
                    "evidence_strategy": "structured",
                }
            ]
        }
    )
    assert len(rq.constraints) == 1
    assert rq.constraints[0].mapped_fields[0].value == "parking"
    assert rq.constraints[0].priority == ConstraintPriority.MUST
    assert rq.constraints[0].mapping_status == ConstraintMappingStatus.KNOWN
    assert rq.constraints[0].category == ConstraintCategory.AMENITY
    assert rq.constraints[0].evidence_strategy == EvidenceStrategy.STRUCTURED


def test_factual_question_agent_builds_and_embeds_field_vocabulary(
    monkeypatch,
):
    """
    Smoke test mirroring the sibling intent agents: build_*_agent()
    constructs without a network call, embeds the existing Field
    vocabulary (not a new one), and does not reference SearchRequest/
    SearchIntentPatch-only concepts like scope/target_result_id.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    agent = build_factual_question_agent()

    assert agent.name == "factual_question_extraction"
    assert '"parking"' in agent.instruction
    assert '"iron"' in agent.instruction
    assert "requirement" in agent.instruction.lower()

    # The schema itself (not just prose explaining why it's absent) must
    # not declare scope/target_result_id as actual output properties.
    schema = ResultQuery.model_json_schema()
    assert set(schema["properties"].keys()) == {"constraints"}


class FakeEvent:
    def __init__(self, *, text_parts: list[str], is_final: bool = True) -> None:
        self.content = SimpleNamespace(
            parts=[SimpleNamespace(text=text) for text in text_parts]
        )
        self._is_final = is_final

    def is_final_response(self) -> bool:
        return self._is_final


async def _event_stream(*events: FakeEvent) -> AsyncIterator[FakeEvent]:
    for event in events:
        yield event


class FakeSessionService:
    async def create_session(self, **kwargs):
        return None


def _make_fake_runner(response_text_parts: list[str]):
    class FakeRunner:
        def __init__(self, *, agent, app_name, session_service):
            pass

        def run_async(self, **kwargs):
            return _event_stream(FakeEvent(text_parts=response_text_parts))

    return FakeRunner


@pytest.mark.asyncio
async def test_extract_factual_constraints_parses_json_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        factual_question_extraction,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        factual_question_extraction,
        "Runner",
        _make_fake_runner(
            [
                '{"constraints":[{"raw_text":"parking","normalized_text":'
                '"parking","priority":"must","category":"amenity",'
                '"mapping_status":"known","mapped_fields":["parking"],'
                '"evidence_strategy":"structured"}]}'
            ]
        ),
    )

    result = await factual_question_extraction.extract_factual_constraints_async(
        "Which of these have parking?"
    )

    assert isinstance(result, ResultQuery)
    assert len(result.constraints) == 1
    assert result.constraints[0].normalized_text == "parking"


@pytest.mark.asyncio
async def test_extract_factual_constraints_strips_json_fence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        factual_question_extraction,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        factual_question_extraction,
        "Runner",
        _make_fake_runner(
            [
                "```json\n",
                '{"constraints":[]}',
                "\n```",
            ]
        ),
    )

    result = await factual_question_extraction.extract_factual_constraints_async(
        "Is the second one romantic?"
    )

    assert result.constraints == []


@pytest.mark.asyncio
async def test_extract_factual_constraints_raises_on_empty_response(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        factual_question_extraction,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        factual_question_extraction,
        "Runner",
        _make_fake_runner([]),
    )

    with pytest.raises(ValueError):
        await factual_question_extraction.extract_factual_constraints_async(
            "Which of these have parking?"
        )
