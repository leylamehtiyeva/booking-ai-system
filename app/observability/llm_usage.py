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
) -> None:
    if trace is None:
        return

    usage = getattr(response, "usage_metadata", None)

    prompt_tokens = getattr(usage, "prompt_token_count", None)
    completion_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

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