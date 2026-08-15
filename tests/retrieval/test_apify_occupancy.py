from datetime import date

import pytest

import app.retrieval.apify as apify_module
from app.retrieval.apify import ApifyRetriever
from app.schemas.query import SearchRequest


@pytest.mark.asyncio
async def test_apify_sends_occupancy_to_actor(
    monkeypatch,
):
    captured_payload = {}

    def fake_post_json_sync(
        url,
        payload,
        timeout=180,
    ):
        captured_payload.update(payload)
        return []

    monkeypatch.setenv(
        "APIFY_TOKEN",
        "test-token",
    )

    monkeypatch.setattr(
        apify_module,
        "_post_json_sync",
        fake_post_json_sync,
    )

    monkeypatch.setattr(
        apify_module,
        "_save_apify_debug_payload",
        lambda *args, **kwargs: None,
    )

    req = SearchRequest(
        city="Baku",
        check_in=date(2026, 8, 20),
        check_out=date(2026, 8, 25),
        adults=3,
        children=2,
        rooms=2,
    )

    retriever = ApifyRetriever()

    await retriever.get_candidates(
        req,
        max_items=10,
    )

    assert captured_payload["adults"] == 3
    assert captured_payload["children"] == 2
    assert captured_payload["rooms"] == 2

    assert captured_payload["search"] == "Baku"
    assert captured_payload["checkIn"] == "2026-08-20"
    assert captured_payload["checkOut"] == "2026-08-25"