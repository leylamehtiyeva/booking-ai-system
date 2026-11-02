"""
LiteApiClient: sandbox guard, path whitelist, error mapping, retries and
telemetry. Hermetic - every response comes from httpx.MockTransport.
"""

from __future__ import annotations

import httpx
import pytest

from app.observability.trace import RequestTrace
from app.retrieval.errors import RetrievalError
from app.retrieval.liteapi_client import LiteApiClient

KEY = "sand_secret-test-key-123"


def _client(handler, **kwargs) -> LiteApiClient:
    kwargs.setdefault("retry_delay_seconds", 0)
    return LiteApiClient(KEY, transport=httpx.MockTransport(handler), **kwargs)


def _responses(*responses):
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler, calls


async def test_key_is_sent_as_header_and_never_in_the_url():
    handler, calls = _responses(httpx.Response(200, json={"data": {"id": "h1"}}))

    payload = await _client(handler).get_hotel_details("h1")

    assert payload == {"data": {"id": "h1"}}
    assert calls[0].headers["X-API-Key"] == KEY
    assert KEY not in str(calls[0].url)
    assert calls[0].url.path == "/v3.0/data/hotel"
    assert calls[0].url.params["hotelId"] == "h1"


@pytest.mark.parametrize("key", [None, "", "prod_live-key", "plain-key"])
def test_only_sandbox_keys_are_accepted_by_default(key):
    with pytest.raises(RetrievalError) as exc:
        LiteApiClient(key)
    assert exc.value.code == "configuration_error"


def test_a_production_key_needs_an_explicit_opt_in():
    LiteApiClient("prod_live-key", require_sandbox=False)


async def test_a_non_sandbox_rates_response_is_refused():
    handler, _ = _responses(httpx.Response(200, json={"sandbox": False, "data": []}))

    with pytest.raises(RetrievalError) as exc:
        await _client(handler).search_rates({"hotelIds": ["h1"]})

    assert exc.value.code == "configuration_error"


async def test_paths_outside_the_whitelist_are_rejected():
    handler, calls = _responses()

    with pytest.raises(ValueError):
        await _client(handler)._request("POST", "/rates/book", trace=None)

    assert calls == []


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(401, json={"error": "denied"}), "authentication_error"),
        (httpx.Response(403), "authentication_error"),
        (httpx.Response(400, json={"error": "bad"}), "provider_error"),
        (httpx.Response(404), "provider_error"),
        (httpx.Response(200, content=b"not json"), "invalid_response"),
        (httpx.Response(200, json=["not", "an", "object"]), "invalid_response"),
    ],
)
async def test_non_retryable_failures_map_to_retrieval_codes(response, code):
    handler, calls = _responses(response)

    with pytest.raises(RetrievalError) as exc:
        await _client(handler, max_attempts=3).get_hotel_details("h1")

    assert exc.value.code == code
    assert len(calls) == 1
    assert KEY not in str(exc.value)


async def test_rate_limit_is_retried_then_succeeds():
    handler, calls = _responses(httpx.Response(429), httpx.Response(200, json={"sandbox": True, "data": []}))
    trace = RequestTrace()

    payload = await _client(handler, max_attempts=2).search_rates({"hotelIds": ["h1"]}, trace=trace)

    assert payload["data"] == []
    assert len(calls) == 2
    assert [call.error for call in trace.external_calls] == ["rate_limited", None]


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (httpx.Response(503), "provider_unavailable"),
        (httpx.ConnectError("synthetic"), "provider_unavailable"),
        (httpx.ReadTimeout("synthetic"), "timeout"),
    ],
)
async def test_retryable_failures_stop_after_max_attempts(failure, code):
    handler, calls = _responses(httpx.Response(503), failure)

    with pytest.raises(RetrievalError) as exc:
        await _client(handler, max_attempts=2).get_hotel_details("h1")

    assert exc.value.code == code
    assert len(calls) == 2


async def test_timeout_is_not_retried():
    handler, calls = _responses(httpx.ReadTimeout("synthetic"), httpx.Response(200, json={}))

    with pytest.raises(RetrievalError) as exc:
        await _client(handler, max_attempts=3).get_hotel_details("h1")

    assert exc.value.code == "timeout"
    assert len(calls) == 1


async def test_telemetry_holds_no_payload_or_key():
    secret_body = {"data": {"id": "h1", "hotelDescription": "SECRET-CONTENT"}, "commission": 12.34}
    handler, _ = _responses(httpx.Response(200, json=secret_body))
    trace = RequestTrace()

    await _client(handler).get_hotel_details("h1", trace=trace)

    call = trace.external_calls[0]
    assert call.provider == "liteapi"
    assert call.step == "liteapi_data_hotel"
    assert call.success is True
    assert call.metadata == {"endpoint": "/data/hotel", "http_status": 200}
    dumped = str(trace.summary())
    assert "SECRET-CONTENT" not in dumped and "12.34" not in dumped and KEY not in dumped
