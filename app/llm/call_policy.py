"""
Failure policy for a single LLM stage call: timeout, retry of retryable
provider errors only, and classification into a small closed set of
error codes.

Conversation-router conventions are reused as-is - the same error codes
and the same retryable set (rate limits and provider unavailability;
timeouts, auth errors and invalid responses are not retried).

Anything that is not a recognised provider/network/configuration failure
is treated as an internal programming error and propagates unchanged -
it is never relabelled as a provider failure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import litellm
from google.genai.errors import APIError as GenaiAPIError
from google.genai.errors import UnknownApiResponseError

from app.config import settings
from app.llm.model_factory import LlmModelConfigurationError
from app.observability.llm_usage import record_llm_call_failed
from app.observability.trace import RequestTrace

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_ERROR_CODES = frozenset({"rate_limited", "provider_unavailable"})


class LlmCallError(RuntimeError):
    """
    A recognised, user-safe LLM stage failure. `code` is one of:
    timeout, rate_limited, authentication_error, provider_unavailable,
    provider_error, configuration_error, empty_response, invalid_response.
    The message never carries provider payloads or prompt text.
    """

    def __init__(
        self,
        code: str,
        *,
        stage: str,
        retryable: bool = False,
        parse_failure: bool = False,
    ) -> None:
        super().__init__(f"LLM stage {stage!r} failed: {code}")
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.parse_failure = parse_failure


def classify_api_error(exc: GenaiAPIError) -> str:
    code = getattr(exc, "code", None)

    if code == 429:
        return "rate_limited"

    if code in {401, 403}:
        return "authentication_error"

    if isinstance(code, int) and 500 <= code < 600:
        return "provider_unavailable"

    return "provider_error"


def classify_litellm_error(exc: Exception) -> str:
    if isinstance(exc, litellm.Timeout):
        return "timeout"

    if isinstance(exc, litellm.RateLimitError):
        return "rate_limited"

    if isinstance(exc, (litellm.AuthenticationError, litellm.PermissionDeniedError)):
        return "authentication_error"

    if isinstance(
        exc,
        (
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.APIConnectionError,
        ),
    ):
        return "provider_unavailable"

    return "provider_error"


_LITELLM_ERRORS = (
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.BadRequestError,
    litellm.NotFoundError,
    litellm.APIError,
)


def classify_provider_exception(exc: BaseException) -> str | None:
    """
    Error code for a provider/network/configuration failure, or None when
    the exception is not one (i.e. a programming error to re-raise).
    """
    if isinstance(exc, LlmCallError):
        return exc.code

    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "timeout"

    if isinstance(exc, _LITELLM_ERRORS):
        return classify_litellm_error(exc)

    if isinstance(exc, UnknownApiResponseError):
        return "invalid_response"

    if isinstance(exc, GenaiAPIError):
        return classify_api_error(exc)

    if isinstance(exc, httpx.TransportError):
        return "provider_unavailable"

    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp ships with google-genai
        aiohttp = None

    if aiohttp is not None and isinstance(exc, aiohttp.ClientConnectionError):
        return "provider_unavailable"

    if isinstance(exc, LlmModelConfigurationError):
        return "configuration_error"

    return None


def _retry_delay_seconds(attempt: int) -> float:
    return settings.LLM_STAGE_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))


async def call_llm_stage(
    attempt: Callable[[], Awaitable[T]],
    *,
    stage: str,
    model: str,
    trace: RequestTrace | None,
) -> T:
    """
    Run one LLM stage call under the shared failure policy.

    `attempt` performs a single provider call and returns the parsed
    result; it raises LlmCallError("empty_response" / "invalid_response")
    itself when the provider answered with unusable content, and records
    its own success telemetry. Every failed attempt is recorded here with
    the error code only.

    Raises LlmCallError for recognised failures; any other exception is
    re-raised unchanged.
    """
    max_attempts = max(1, settings.LLM_STAGE_MAX_ATTEMPTS)

    for attempt_number in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(settings.LLM_STAGE_TIMEOUT_SECONDS):
                return await attempt()

        except Exception as exc:
            code = classify_provider_exception(exc)
            if code is None:
                raise

            error = (
                exc
                if isinstance(exc, LlmCallError)
                else LlmCallError(code, stage=stage)
            )
            error.stage = stage
            error.retryable = code in RETRYABLE_ERROR_CODES

            record_llm_call_failed(
                trace=trace,
                step=stage,
                model=model,
                error=code,
                parse_failure=error.parse_failure,
            )

            if not error.retryable or attempt_number >= max_attempts:
                if error is exc:
                    raise
                raise error from exc

            logger.warning(
                "Retrying LLM stage %s after %s (attempt %s/%s)",
                stage,
                code,
                attempt_number,
                max_attempts,
            )
            await asyncio.sleep(_retry_delay_seconds(attempt_number))

    raise AssertionError("unreachable: call_llm_stage exhausted attempts")
