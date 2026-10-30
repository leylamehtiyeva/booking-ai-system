"""
Async HTTP client for LiteAPI (Nuitee).

Only the read-only endpoints the adapter needs are reachable (hotel details
and rate search); booking endpoints (prebook/book) are not. By default the
client only accepts a sandbox key ("sand_" prefix) and refuses a rates
response that does not declare itself a sandbox response.

Failures are raised as RetrievalError with the shared error-code vocabulary.
The API key travels in a header only; it never appears in an error, a log
line or telemetry, and neither does a response body. Telemetry records the
endpoint, HTTP status and latency - never payload contents or amounts.

Responses are returned as parsed JSON for the caller to normalize at once;
nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from app.observability.trace import ExternalCallTrace, RequestTrace
from app.retrieval.errors import RetrievalError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.liteapi.travel/v3.0"
SANDBOX_KEY_PREFIX = "sand_"
API_KEY_ENV = "LITEAPI_SANDBOX_KEY"

HOTEL_DETAILS_PATH = "/data/hotel"
RATES_PATH = "/hotels/rates"
ALLOWED_PATHS = frozenset({HOTEL_DETAILS_PATH, RATES_PATH})

RETRYABLE_CODES = frozenset({"rate_limited", "provider_unavailable"})


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def classify_liteapi_http_status(status: int) -> str:
    if status == 429:
        return "rate_limited"
    if status in {401, 403}:
        return "authentication_error"
    if 500 <= status < 600:
        return "provider_unavailable"
    return "provider_error"


class LiteApiClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        retry_delay_seconds: float = 0.5,
        require_sandbox: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise RetrievalError("configuration_error")
        if require_sandbox and not key.startswith(SANDBOX_KEY_PREFIX):
            # A production key must be enabled explicitly, never by accident.
            raise RetrievalError("configuration_error")

        self._require_sandbox = require_sandbox
        self._max_attempts = max(1, max_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": key, "Accept": "application/json"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> LiteApiClient:
        kwargs.setdefault("base_url", os.getenv("LITEAPI_BASE_URL", DEFAULT_BASE_URL))
        kwargs.setdefault("timeout_seconds", _env_float("LITEAPI_TIMEOUT_SECONDS", 20.0))
        kwargs.setdefault("max_attempts", _env_int("LITEAPI_MAX_ATTEMPTS", 2))
        return cls(os.getenv(API_KEY_ENV), **kwargs)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> LiteApiClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def get_hotel_details(
        self,
        hotel_id: str,
        *,
        timeout_seconds: int = 10,
        trace: RequestTrace | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            HOTEL_DETAILS_PATH,
            params={"hotelId": hotel_id, "timeout": timeout_seconds},
            trace=trace,
        )

    async def search_rates(
        self,
        body: dict[str, Any],
        *,
        trace: RequestTrace | None = None,
    ) -> dict[str, Any]:
        payload = await self._request("POST", RATES_PATH, json_body=body, trace=trace)
        if self._require_sandbox and payload.get("sandbox") is False:
            raise RetrievalError("configuration_error")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        trace: RequestTrace | None,
    ) -> dict[str, Any]:
        if path not in ALLOWED_PATHS:
            raise ValueError(f"LiteAPI path not allowed: {path}")

        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            status: int | None = None
            try:
                response = await self._http.request(method, path, params=params, json=json_body)
                status = response.status_code
                if status != 200:
                    raise RetrievalError(classify_liteapi_http_status(status))
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RetrievalError("invalid_response") from exc
                if not isinstance(payload, dict):
                    raise RetrievalError("invalid_response")
            except RetrievalError as exc:
                error = exc
            except httpx.TimeoutException as exc:
                error = RetrievalError("timeout")
                error.__cause__ = exc
            except httpx.TransportError as exc:
                error = RetrievalError("provider_unavailable")
                error.__cause__ = exc
            else:
                _record_call(trace, path, started, status=status, error=None)
                return payload

            _record_call(trace, path, started, status=status, error=error.code)
            logger.warning(
                "LiteAPI %s %s failed: %s (HTTP %s, attempt %s/%s)",
                method, path, error.code, status, attempt, self._max_attempts,
            )
            if error.code not in RETRYABLE_CODES or attempt >= self._max_attempts:
                raise error
            await asyncio.sleep(self._retry_delay_seconds * attempt)

        raise AssertionError("unreachable: LiteAPI request attempts exhausted")


def _record_call(
    trace: RequestTrace | None,
    path: str,
    started: float,
    *,
    status: int | None,
    error: str | None,
) -> None:
    if trace is None:
        return
    trace.add_external_call(
        ExternalCallTrace(
            step=f"liteapi{path.replace('/', '_')}",
            provider="liteapi",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            success=error is None,
            error=error,
            metadata={"endpoint": path, "http_status": status},
        )
    )
