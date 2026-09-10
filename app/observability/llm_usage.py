from __future__ import annotations

from typing import Any

from app.observability.pricing import (
    estimate_llm_cost_usd,
    estimate_tokens_from_text,
)
from app.observability.trace import LLMCallTrace, RequestTrace


def record_llm_call_from_response(
    *,
    trace: RequestTrace | None,
    step: str,
    model: str,
    response: Any,
    success: bool = True,
    error: str | None = None,
    latency_ms: float | None = None,
    parse_failure: bool = False,
) -> LLMCallTrace:
    """
    Records an LLM call for which a response object was actually
    received - usage_metadata (and therefore token counts/cost) is read
    from it. This covers both a fully successful call and a call whose
    response parsed but whose content failed structured-output/enum
    validation (success=False, parse_failure=True): the API call itself
    still happened and still consumed tokens, so cost should still be
    attributed.

    For a call that produced no response object at all (an API/network/
    timeout exception before any response came back), use
    record_llm_call_failed instead - there is nothing here to read
    usage_metadata from.

    Always returns the constructed LLMCallTrace (even when trace is
    None, in which case it is simply not appended anywhere) so a caller
    that needs this call's own latency/tokens/cost - e.g. to aggregate
    them itself - never has to guess which trace.llm_calls entry was
    "the one it just made" (unsafe under concurrent execution, where
    trace.llm_calls[-1] is not necessarily this call's entry).
    """
    usage = getattr(response, "usage_metadata", None)

    prompt_tokens = getattr(usage, "prompt_token_count", None)
    completion_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    call = LLMCallTrace(
        step=step,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_llm_cost_usd(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        success=success,
        error=error,
        latency_ms=latency_ms,
        parse_failure=parse_failure,
    )

    if trace is not None:
        trace.add_llm_call(call)

    return call


def record_llm_call_failed(
    *,
    trace: RequestTrace | None,
    step: str,
    model: str,
    error: str,
    latency_ms: float | None = None,
    parse_failure: bool = False,
) -> LLMCallTrace:
    """
    Records an LLM call that produced no usable response object at all
    (e.g. an API/network/timeout exception raised before any response
    came back). record_llm_call_from_response can't be reused for this
    case - there is no response to read usage_metadata from - so token
    counts and cost are left None (unknown, not zero) rather than forcing
    a response-shaped call for a call that never got a response.

    Always returns the constructed LLMCallTrace - see
    record_llm_call_from_response's docstring for why.
    """
    call = LLMCallTrace(
        step=step,
        model=model,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
        success=False,
        error=error,
        latency_ms=latency_ms,
        parse_failure=parse_failure,
    )

    if trace is not None:
        trace.add_llm_call(call)

    return call


def record_llm_call_estimated(
    *,
    trace: RequestTrace | None,
    step: str,
    model: str,
    prompt_text: str,
    response_text: str | None,
    success: bool = True,
    error: str | None = None,
) -> None:
    
    if trace is None:
        return

    prompt_tokens = estimate_tokens_from_text(prompt_text)
    completion_tokens = estimate_tokens_from_text(response_text)
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    trace.add_llm_call(
        LLMCallTrace(
            step=step,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_llm_cost_usd(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            success=success,
            error=error,
        )
    )