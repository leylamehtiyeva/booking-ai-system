from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.logic import semantic_evidence_verification as sev
from app.observability.trace import RequestTrace
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence import EvidenceRelation, EvidenceResolutionStatus


class _FakeGenaiTypes:
    """Stand-in for google.genai.types: constructors just need to not raise."""

    @staticmethod
    def Content(*, role, parts):
        return SimpleNamespace(role=role, parts=parts)

    @staticmethod
    def Part(*, text):
        return SimpleNamespace(text=text)

    @staticmethod
    def GenerateContentConfig(**kwargs):
        return SimpleNamespace(**kwargs)


def _fake_usage(prompt_tokens=100, completion_tokens=20, total_tokens=120):
    return SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=total_tokens,
    )


def _fake_response(text: str, usage=None) -> SimpleNamespace:
    return SimpleNamespace(text=text, usage_metadata=usage or _fake_usage())


def _policy() -> SemanticVerifierPolicy:
    return SemanticVerifierPolicy(model="gemini-2.5-flash")


# ---------------- 3. Valid structured response ----------------


async def test_verify_evidence_relation_success(monkeypatch):
    resp = _fake_response(json.dumps({"relation": "SUPPORT", "reason": "explicit match"}))

    monkeypatch.setattr(sev, "_gemini_client", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: resp)
    ))
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    trace = RequestTrace()
    result = await sev.verify_evidence_relation(
        "Free WiFi in every room.",
        "Wi-Fi / internet access is available.",
        policy=_policy(),
        trace=trace,
    )

    assert result.status == EvidenceResolutionStatus.RESOLVED
    assert result.relation == EvidenceRelation.SUPPORT
    assert result.reason == "explicit match"
    assert result.error is None

    # Result carries its own telemetry directly - matches the trace entry,
    # so a caller never needs to peek at trace.llm_calls to aggregate usage.
    assert result.parse_failure is False
    assert result.latency_ms >= 0
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.total_tokens == 120
    assert result.estimated_cost_usd is not None

    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.step == "semantic_evidence_verifier"
    assert call.model == "gemini-2.5-flash"
    assert call.success is True
    assert call.parse_failure is False
    assert call.latency_ms is not None and call.latency_ms >= 0
    assert call.prompt_tokens == 100
    assert call.total_tokens == 120
    assert call.estimated_cost_usd is not None
    assert call.latency_ms == result.latency_ms
    assert call.total_tokens == result.total_tokens
    assert call.estimated_cost_usd == result.estimated_cost_usd


# ---------------- 1. API / network / timeout exception (no response object) ----------------


async def test_verify_evidence_relation_api_exception_records_failed_call_without_response(monkeypatch):
    def _raise(**kwargs):
        raise TimeoutError("upstream timed out")

    monkeypatch.setattr(sev, "_gemini_client", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=_raise)
    ))
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    trace = RequestTrace()
    result = await sev.verify_evidence_relation(
        "Some evidence.",
        "Some hypothesis.",
        policy=_policy(),
        trace=trace,
    )

    assert result.status == EvidenceResolutionStatus.VERIFICATION_FAILED
    assert result.relation is None
    assert "TimeoutError" in result.error

    # No response object existed - result's own token accounting must
    # stay unknown, not zero, matching the trace entry.
    assert result.parse_failure is False
    assert result.prompt_tokens is None
    assert result.total_tokens is None
    assert result.estimated_cost_usd is None

    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.success is False
    assert call.parse_failure is False  # exception path, NOT a parse failure
    assert call.error is not None and "TimeoutError" in call.error
    assert call.latency_ms is not None and call.latency_ms >= 0
    # No response object existed - token accounting must stay unknown, not zero.
    assert call.prompt_tokens is None
    assert call.total_tokens is None
    assert call.estimated_cost_usd is None
    assert call.latency_ms == result.latency_ms


# ---------------- 2. Response received, but JSON/enum parsing fails ----------------


async def test_verify_evidence_relation_invalid_json_is_parse_failure(monkeypatch):
    resp = _fake_response("not valid json at all")

    monkeypatch.setattr(sev, "_gemini_client", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: resp)
    ))
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    trace = RequestTrace()
    result = await sev.verify_evidence_relation(
        "Some evidence.",
        "Some hypothesis.",
        policy=_policy(),
        trace=trace,
    )

    assert result.status == EvidenceResolutionStatus.VERIFICATION_FAILED
    assert result.relation is None
    assert result.error is not None

    # response WAS received; content just didn't parse - result's own
    # parse_failure/tokens must reflect that (usage_metadata still readable).
    assert result.parse_failure is True
    assert result.prompt_tokens == 100

    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.success is False
    assert call.parse_failure is True  # response WAS received; content just didn't parse
    # A response object DID exist here, so usage_metadata is still readable.
    assert call.prompt_tokens == 100
    assert call.estimated_cost_usd is not None


async def test_verify_evidence_relation_invalid_relation_enum_is_parse_failure(monkeypatch):
    resp = _fake_response(json.dumps({"relation": "MAYBE", "reason": "unsure"}))

    monkeypatch.setattr(sev, "_gemini_client", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: resp)
    ))
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    trace = RequestTrace()
    result = await sev.verify_evidence_relation(
        "Some evidence.",
        "Some hypothesis.",
        policy=_policy(),
        trace=trace,
    )

    assert result.status == EvidenceResolutionStatus.VERIFICATION_FAILED
    assert result.relation is None

    call = trace.llm_calls[0]
    assert call.parse_failure is True
    assert call.success is False


# ---------------- structural separation from retrieval metadata ----------------


def test_semantic_verification_result_has_no_retrieval_fields():
    """
    Still owns nothing about retrieval - only this call's own outcome
    and telemetry (relation/status/error/reason plus latency/tokens/
    cost/parse_failure, added for concurrency-safe usage aggregation).
    """
    fields = set(sev.SemanticVerificationResult.model_fields)
    assert fields == {
        "relation", "reason", "status", "error",
        "parse_failure", "latency_ms", "prompt_tokens",
        "completion_tokens", "total_tokens", "estimated_cost_usd",
    }
    assert "source_type" not in fields
    assert "source_path" not in fields
    assert "retrieval_score" not in fields
    assert "evidence_text" not in fields


# ---------------- setup/config errors must NOT be swallowed as VERIFICATION_FAILED ----------------


async def test_missing_api_key_propagates_instead_of_becoming_verification_failed(monkeypatch):
    """
    A missing GOOGLE_API_KEY (or an uninstalled google-genai) is a
    setup/config bug, not a transient operational failure - it must
    raise, not be silently converted into a per-call
    SemanticVerificationResult(status=VERIFICATION_FAILED).
    """

    def _raise_missing_key():
        raise ValueError("Missing GOOGLE_API_KEY")

    monkeypatch.setattr(sev, "_gemini_client", _raise_missing_key)
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    trace = RequestTrace()
    with pytest.raises(ValueError, match="Missing GOOGLE_API_KEY"):
        await sev.verify_evidence_relation(
            "Some evidence.",
            "Some hypothesis.",
            policy=_policy(),
            trace=trace,
        )

    # Nothing should have been recorded - the call never even started.
    assert trace.llm_calls == []


async def test_missing_google_genai_dependency_propagates(monkeypatch):
    def _raise_import_error():
        raise ImportError("google-genai is not installed")

    monkeypatch.setattr(sev, "_gemini_client", _raise_import_error)
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    with pytest.raises(ImportError):
        await sev.verify_evidence_relation(
            "Some evidence.",
            "Some hypothesis.",
            policy=_policy(),
            trace=None,
        )


# ---------------- trace=None must not raise ----------------


async def test_verify_evidence_relation_works_without_trace(monkeypatch):
    resp = _fake_response(json.dumps({"relation": "NOT_ENOUGH_EVIDENCE", "reason": "unrelated"}))

    monkeypatch.setattr(sev, "_gemini_client", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: resp)
    ))
    monkeypatch.setattr(sev, "_genai_types", lambda: _FakeGenaiTypes)

    result = await sev.verify_evidence_relation(
        "Some evidence.",
        "Some hypothesis.",
        policy=_policy(),
        trace=None,
    )

    assert result.status == EvidenceResolutionStatus.RESOLVED
    assert result.relation == EvidenceRelation.NOT_ENOUGH_EVIDENCE
