from __future__ import annotations

from app.config.settings import MAX_ITEMS_HARD_CAP

from app.logic.normalize_search_response import normalize_search_response

from app.logic.result_selection import select_ranked_items
from app.observability.trace import RequestTrace
from app.retrieval import Source, get_candidates
from app.schemas.fallback_policy import FallbackPolicy

from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse

from app.schemas.search_response import (
    NormalizedSearchResponse,
    SearchStatus,
)

from app.logic.listing_evaluation import (
    evaluate_listings,
)


async def orchestrate_search_request(
    req: SearchRequest,
    *,
    result_limit: int = MAX_ITEMS_HARD_CAP,
    candidate_pool_size: int = MAX_ITEMS_HARD_CAP,
    source: Source = "fixtures",
    fallback_policy: FallbackPolicy | None = None,
    trace: RequestTrace | None = None,
) -> NormalizedSearchResponse:
    """
    Execute accommodation search for an already validated
    canonical SearchRequest.

    The caller is responsible for constructing and validating
    the SearchRequest before entering the search layer.
    """

    if candidate_pool_size <= 0:
        raise ValueError(
            "candidate_pool_size must be > 0"
        )

    if result_limit <= 0:
        raise ValueError(
            "result_limit must be > 0"
        )

    if candidate_pool_size > MAX_ITEMS_HARD_CAP:
        raise ValueError(
            "candidate_pool_size must be <= "
            f"{MAX_ITEMS_HARD_CAP}"
        )

    if (
        not req.city
        or req.check_in is None
        or req.check_out is None
    ):
        raise ValueError(
            "orchestrate_search_request requires "
            "city, check_in and check_out"
        )

    if trace is None:
        trace = RequestTrace()

    # Retrieval
    with trace.step(
        "retrieval",
        source=source,
        candidate_pool_size=candidate_pool_size,
    ):
        listings = await get_candidates(
            req,
            max_items=candidate_pool_size,
            source=source,
            trace=trace,
        )

    evaluation = await evaluate_listings(
        req,
        listings,
        fallback_policy=fallback_policy,
        trace=trace,
    )

    ranked = evaluation.ranked_items

    if not ranked:
        return NormalizedSearchResponse(
            status=SearchStatus.NO_RESULTS,
            results=[],
            debug_notes=(
                evaluation.debug_notes
            ),
        )

    # Final result limit
    with trace.step(
        "final_selection",
        ranked_count=len(ranked),
        result_limit=result_limit,
    ):
        selected = select_ranked_items(
            ranked,
            top_n=result_limit,
        )

    # Normalize search output
    with trace.step(
        "normalize_response",
        selected_count=len(selected),
    ):
        normalized = normalize_search_response(
        req,
        selected,
        top_n=result_limit,
        dropped_requests=[],
    )

    return normalized