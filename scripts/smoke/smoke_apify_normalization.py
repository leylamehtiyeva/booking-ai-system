from datetime import date
import asyncio

from dotenv import load_dotenv

from app.retrieval.apify import ApifyRetriever
from app.schemas.query import SearchRequest


load_dotenv()


async def main():
    req = SearchRequest(
        city="Baku",
        check_in=date(2026, 9, 10),
        check_out=date(2026, 9, 12),
        adults=3,
        children=2,
        rooms=2,
        currency="USD",
    )

    listings = await ApifyRetriever().get_candidates(
        req,
        max_items=3,
    )

    print("\n=== NORMALIZED LISTINGS ===")
    print("Returned listings:", len(listings))

    for i, listing in enumerate(listings, start=1):
        print(f"\n--- Listing #{i} ---")
        print("id:", listing.id)
        print("name:", listing.name)
        print("city:", listing.city)
        print("address:", listing.address)
        print("property_type:", listing.property_type)
        print("price:", listing.price)
        print("currency:", listing.currency)

        print("facilities:", len(listing.facilities))
        print("rooms:", len(listing.rooms))

        if listing.facilities:
            print(
                "first facility:",
                listing.facilities[0],
            )

        if listing.rooms:
            room = listing.rooms[0]

            print("room.name:", room.name)
            print(
                "room.persons:",
                room.persons,
            )
            print(
                "room.facilities:",
                len(room.facilities),
            )
            print(
                "room.bed_types:",
                len(room.bed_types),
            )
            print(
                "room.options:",
                len(room.options),
            )

            if room.options:
                print(
                    "first option:",
                    room.options[0],
                )


if __name__ == "__main__":
    asyncio.run(main())