# scripts/smoke_apify_search.py
from __future__ import annotations

from datetime import date, timedelta

from app.schemas.query import SearchRequest
from app.services.apify_booking import ApifyBookingService
from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    req = SearchRequest(
        user_message="Хочу квартиру в Баку с возможностью готовить еду, чайником и ванной",
        city="Baku",
        check_in=date.today() + timedelta(days=14),
        check_out=date.today() + timedelta(days=17),
        adults=2,
        must_have_fields=[],  # в PR#3 это поле пока не используется сервисом
    )

    svc = ApifyBookingService()
    items = svc.search_listings(req, limit=10)

    print(f"Got {len(items)} listings")
    for i, x in enumerate(items[:5], start=1):
        print(f"{i}. {x.name} | {x.price} {x.currency} | rating={x.rating} | {x.url}")


if __name__ == "__main__":
    main()
