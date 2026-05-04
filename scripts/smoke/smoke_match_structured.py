# from __future__ import annotations

# from datetime import date, timedelta

# from dotenv import load_dotenv

# load_dotenv()

# from app.logic.matcher_structured import match_listing_structured
# from app.schemas.fields import Field
# from app.schemas.query import SearchRequest
# from app.services.apify_booking import ApifyBookingService


# def main() -> None:
#     req = SearchRequest(
#         user_message="Хочу квартиру в Баку с возможностью готовить еду, чайником и ванной",
#         city="Baku",
#         check_in=date.today() + timedelta(days=14),
#         check_out=date.today() + timedelta(days=17),
#         adults=2,
#         must_have_fields=[Field.KITCHEN, Field.KETTLE, Field.PRIVATE_BATHROOM],
#     )

#     svc = ApifyBookingService()
#     listings = svc.search_listings(req, limit=10)

#     print(f"Listings fetched: {len(listings)}")

#     for i, lst in enumerate(listings, start=1):
#         report = match_listing_structured(lst, req)

#         yes_cnt = sum(1 for m in report.matches.values() if m.value.name == "YES")
#         total = len(report.matches)

#         print(f"\n{i}. {lst.name} | rating={lst.rating} | {lst.url}")
#         print(f"   must-have matched: {yes_cnt}/{total}")

#         for f, m in report.matches.items():
#             print(f"   - {f.name}: {m.value.name} (conf={m.confidence})")
#             if m.evidence:
#                 print(f"     evidence: {m.evidence}")


# if __name__ == "__main__":
#     main()
from __future__ import annotations

import json
from pathlib import Path

from app.logic.matcher_structured import match_listing_structured
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest


def main() -> None:
    # OFFLINE: читаем тестовые листинги из fixtures
    data = json.loads(Path("fixtures/listings_sample.json").read_text(encoding="utf-8"))
    listings = [ListingRaw(**x) for x in data]

    req = SearchRequest(
        user_message="Хочу квартиру в Баку с возможностью готовить еду, чайником и ванной",
        city="Baku",
        adults=2,
        must_have_fields=[Field.KITCHEN, Field.KETTLE, Field.PRIVATE_BATHROOM],
    )

    print(f"Listings loaded: {len(listings)}")

    for i, lst in enumerate(listings, start=1):
        report = match_listing_structured(lst, req)

        yes_cnt = sum(1 for m in report.matches.values() if m.value.name == "YES")
        total = len(report.matches)

        print(f"\n{i}. {lst.name} | rating={lst.rating} | {lst.url}")
        print(f"   must-have matched: {yes_cnt}/{total}")

        for f, m in report.matches.items():
            print(f"   - {f.name}: {m.value.name} (conf={m.confidence})")
            if m.evidence:
                for ev in m.evidence:
                    print(f"     evidence: source={ev.source.value} path={ev.path} snippet='{ev.snippet}'")



if __name__ == "__main__":
    main()
