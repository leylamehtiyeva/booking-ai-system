from datetime import date

import pytest

from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.query import SearchRequest
from app.schemas.search_response import (
    NormalizedSearchResponse,
)
from app.tools.orchestrate_search_tool import (
    orchestrate_search_request,
)

from app.schemas.search_response import (
    NormalizedSearchResponse,
    SearchStatus,
)


def make_request(
    *,
    city: str = "Baku",
) -> SearchRequest:
    return SearchRequest(
        city=city,
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 25),
        adults=2,
        children=0,
        rooms=1,
        constraints=[],
    )


@pytest.mark.asyncio
async def test_search_orchestrator_returns_typed_response_for_results():
    response = await orchestrate_search_request(
        make_request(),
        source="fixtures",
        candidate_pool_size=5,
        fallback_policy=FallbackPolicy(
            enabled=False
        ),
    )

    assert isinstance(
        response,
        NormalizedSearchResponse,
    )

    assert response.status == SearchStatus.RESULTS
    assert response.results


@pytest.mark.asyncio
async def test_search_orchestrator_returns_typed_response_for_no_results():
    response = await orchestrate_search_request(
        make_request(city="NoSuchCity"),
        source="fixtures",
        candidate_pool_size=5,
        fallback_policy=FallbackPolicy(
            enabled=False
        ),
    )

    assert isinstance(
        response,
        NormalizedSearchResponse,
    )

    assert response.status == SearchStatus.NO_RESULTS
    assert response.results == []


@pytest.mark.asyncio
async def test_search_orchestrator_rejects_candidate_pool_above_hard_cap():
    with pytest.raises(
        ValueError,
        match="candidate_pool_size must be <=",
    ):
        await orchestrate_search_request(
            make_request(),
            source="fixtures",
            candidate_pool_size=999,
            fallback_policy=FallbackPolicy(
                enabled=False
            ),
        )