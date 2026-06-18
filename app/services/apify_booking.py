from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from apify_client import ApifyClient

from app.schemas.query import SearchRequest
from app.schemas.listing import ListingRaw


class ApifyBookingService:
    """
    Apify Booking service using ApifyClient SDK.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        self.token = token or os.getenv("APIFY_TOKEN")
        self.actor_id = actor_id or os.getenv("APIFY_BOOKING_ACTOR_ID")

        if not self.token:
            raise ValueError("APIFY_TOKEN is missing (env var).")
        if not self.actor_id:
            raise ValueError("APIFY_BOOKING_ACTOR_ID is missing (env var).")

        self.client = ApifyClient(self.token)

    def _build_actor_input(self, request: SearchRequest) -> Dict[str, Any]:
        run_input: Dict[str, Any] = {
            "search": request.city,              # required
            "currency": request.currency or "USD",
            "language": "en-gb",
            "maxItems": 10,                     
            "adults": request.adults,
            "maxConcurrency": 1,
            "maxRequestsPerCrawl": 50,
        }

        if request.check_in:
            run_input["checkIn"] = request.check_in.isoformat()
        if request.check_out:
            run_input["checkOut"] = request.check_out.isoformat()

        return run_input


    def search_listings(self, request: SearchRequest, *, limit: int = 10) -> List[ListingRaw]:
        run_input = self._build_actor_input(request)
        run_input["maxItems"] = limit

        run = self.client.actor(self.actor_id).call(run_input=run_input)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise RuntimeError("Apify run returned no defaultDatasetId")

        items: List[ListingRaw] = []
        for it in self.client.dataset(dataset_id).iterate_items():
            if isinstance(it, dict):
                items.append(ListingRaw(**it))
        return items
