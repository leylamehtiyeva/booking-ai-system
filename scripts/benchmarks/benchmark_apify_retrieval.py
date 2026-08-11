from __future__ import annotations

import asyncio
import time
from datetime import date

from dotenv import load_dotenv

from app.observability.trace import RequestTrace
from app.retrieval.apify import ApifyRetriever
from app.schemas.query import SearchRequest


load_dotenv()


async def main() -> None:
    request = SearchRequest(
        city="Baku",
        check_in=date(2026, 9, 15),
        check_out=date(2026, 9, 18),
        adults=2,
        children=0,
        rooms=1,
    )

    retriever = ApifyRetriever()
    trace = RequestTrace()

    started = time.perf_counter()

    listings = await retriever.get_candidates(
        req=request,
        max_items=10,
        trace=trace,
    )

    total_latency_seconds = time.perf_counter() - started

    print("\n=== RETRIEVAL BENCHMARK ===")
    print(f"City: {request.city}")
    print(f"Requested listings: 10")
    print(f"Returned listings: {len(listings)}")
    print(f"Total retrieval latency: {total_latency_seconds:.2f} sec")

    if trace.external_calls:
        external_call = trace.external_calls[0]

        print(
            f"Apify call latency: "
            f"{external_call.latency_ms / 1000:.2f} sec"
        )

    print("\n=== FIELD COVERAGE ===")

    for index, listing in enumerate(listings, start=1):
        print(f"\nListing #{index}: {listing.name}")
        print(f"  price: {listing.price is not None}")
        print(f"  rating: {listing.rating is not None}")
        print(f"  property_type: {listing.property_type is not None}")
        print(f"  description: {bool(listing.description)}")
        print(f"  facilities: {bool(listing.facilities)}")
        print(f"  rooms: {bool(listing.rooms)}")


if __name__ == "__main__":
    asyncio.run(main())