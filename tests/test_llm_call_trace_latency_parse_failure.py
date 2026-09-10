from __future__ import annotations

from types import SimpleNamespace

from app.observability.llm_usage import (
    record_llm_call_failed,
    record_llm_call_from_response,
)
from app.observability.trace import RequestTrace


def _fake_response(prompt_tokens=10, completion_tokens=5, total_tokens=15):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=total_tokens,
        )
    )


def test_record_llm_call_from_response_backward_compatible_without_new_kwargs():
    """Existing call sites (e.g. constraint_evidence_resolution.py) call this
    without latency_ms/parse_failure at all - must keep working unchanged."""
    trace = RequestTrace()

    record_llm_call_from_response(
        trace=trace,
        step="constraint_textual_fallback",
        model="gemini-2.5-flash",
        response=_fake_response(),
        success=True,
    )

    call = trace.llm_calls[0]
    assert call.prompt_tokens == 10
    assert call.total_tokens == 15
    assert call.success is True
    # New fields default to values that don't change old behavior.
    assert call.latency_ms is None
    assert call.parse_failure is False


def test_record_llm_call_from_response_with_new_kwargs():
    trace = RequestTrace()

    record_llm_call_from_response(
        trace=trace,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        response=_fake_response(),
        success=False,
        error="invalid relation value",
        latency_ms=123.45,
        parse_failure=True,
    )

    call = trace.llm_calls[0]
    assert call.latency_ms == 123.45
    assert call.parse_failure is True
    assert call.success is False
    assert call.error == "invalid relation value"
    # A response object existed, so token accounting is still populated.
    assert call.prompt_tokens == 10


def test_record_llm_call_from_response_with_none_trace_is_noop():
    # Must not raise.
    record_llm_call_from_response(
        trace=None,
        step="x",
        model="gemini-2.5-flash",
        response=_fake_response(),
    )


def test_record_llm_call_failed_has_no_response_and_unknown_tokens():
    trace = RequestTrace()

    record_llm_call_failed(
        trace=trace,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        error="TimeoutError: upstream timed out",
        latency_ms=987.6,
        parse_failure=False,
    )

    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.success is False
    assert call.parse_failure is False
    assert call.error == "TimeoutError: upstream timed out"
    assert call.latency_ms == 987.6
    # No response object -> tokens/cost are unknown, not zero.
    assert call.prompt_tokens is None
    assert call.completion_tokens is None
    assert call.total_tokens is None
    assert call.estimated_cost_usd is None


def test_record_llm_call_failed_with_none_trace_is_noop():
    record_llm_call_failed(
        trace=None,
        step="x",
        model="gemini-2.5-flash",
        error="boom",
    )


def test_llm_call_trace_flows_into_request_trace_summary():
    trace = RequestTrace()
    record_llm_call_failed(
        trace=trace,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        error="boom",
        latency_ms=42.0,
    )

    summary = trace.summary()
    calls = summary["llm"]["calls"]
    assert len(calls) == 1
    assert calls[0]["latency_ms"] == 42.0
    assert calls[0]["parse_failure"] is False
    assert calls[0]["error"] == "boom"


def test_record_llm_call_from_response_returns_the_constructed_call():
    """
    A caller can read this call's own latency/tokens/cost directly
    from the return value - never needs to guess which trace.llm_calls
    entry is "the one it just made" (unsafe under concurrent execution).
    """
    trace = RequestTrace()

    returned = record_llm_call_from_response(
        trace=trace,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        response=_fake_response(prompt_tokens=20, completion_tokens=7, total_tokens=27),
        success=True,
        latency_ms=99.0,
    )

    assert returned is trace.llm_calls[0]
    assert returned.prompt_tokens == 20
    assert returned.total_tokens == 27
    assert returned.latency_ms == 99.0


def test_record_llm_call_from_response_returns_call_even_with_no_trace():
    """trace=None still returns the constructed call - just doesn't append it anywhere."""
    returned = record_llm_call_from_response(
        trace=None,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        response=_fake_response(),
        success=True,
    )
    assert returned.prompt_tokens == 10


def test_record_llm_call_failed_returns_the_constructed_call():
    returned = record_llm_call_failed(
        trace=None,
        step="semantic_evidence_verifier",
        model="gemini-2.5-flash",
        error="TimeoutError: no response",
        latency_ms=55.0,
    )
    assert returned.error == "TimeoutError: no response"
    assert returned.latency_ms == 55.0
    assert returned.prompt_tokens is None
    assert returned.estimated_cost_usd is None
