"""
LiteAPI (Nuitee) adapter: rates + hotel details -> ListingRaw.

    LiteApiRetriever.get_candidates(req, max_items, trace) -> list[ListingRaw]

Same contract as the other retrievers (app.retrieval.base.CandidateRetriever).
Selected with source="liteapi" in app.retrieval.get_candidates; the default
source stays "fixtures", and the public API always uses fixtures.

Flow: one POST /hotels/rates for the city and stay (availability and offers),
then GET /data/hotel for the first few hotels with offers (bounded
concurrency), then normalize_liteapi_listing for each.

Rules this module guarantees:
- prices: only the suggested selling price (SSP) is read, per offer, into
  PublicOffer.public_total; when it is missing or ambiguous the price is
  None. Retail/net totals, initial prices, commissions and fee or penalty
  amounts are never read, so they cannot reach ListingRaw, an LLM prompt,
  the frontend or telemetry. ListingRaw.raw stays None - no provider payload
  is kept, in memory or on disk, beyond the call that normalizes it;
- missing or false provider flags are never turned into a stated "no":
  only facts the provider states positively are emitted (policyAttributes
  booleans are not reliable as negatives);
- hotel-level facilities go to ListingRaw.facilities, room-level amenities
  to the matching Room. ListingRaw.rooms holds only rooms with an offer for
  the stay: a catalog room is used when exactly one has the offer's name,
  otherwise the offer gets a minimal Room of its own; catalog rooms without
  an offer are left out;
- refundability (RFN/NRFN) is kept per offer on structured fields; it is
  never stated as free cancellation and never written into listing
  evidence text, and cancellation-policy times are kept uninterpreted;
- HTML is reduced to plain sentences;
- the provider's review sentiment is kept as ReviewSummary with its origin;
  nothing in matching or evidence reads it yet.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.observability.trace import RequestTrace
from app.retrieval.errors import RetrievalError
from app.retrieval.liteapi_client import LiteApiClient
from app.schemas.listing import (
    BedGroup,
    Facility,
    Highlight,
    ListingProvenance,
    ListingRaw,
    Policy,
    ReviewCategoryScore,
    ReviewSummary,
    Room,
    RoomOption,
)
from app.schemas.public_offer import PublicOffer
from app.schemas.query import SearchRequest

logger = logging.getLogger(__name__)

PROVIDER = "liteapi"
REVIEW_SUMMARY_SOURCE = "liteapi.sentiment_analysis"
PRICE_TOLERANCE = 0.01

# Live-data scope validated in the sandbox smoke (2026-10-29): rates for 40
# hotels came back as a 372 KB response, while ranking only ever looks at
# the 5 hotels whose details are fetched - 20 keeps a choice of offers at
# about half the payload.
DEFAULT_COUNTRY_CODE = "ES"
DEFAULT_SUPPORTED_CITIES = "Barcelona"
REQUIRED_ADULTS = 2
DEFAULT_RATES_LIMIT = 20
DEFAULT_MAX_DETAIL_CALLS = 5

# LiteAPI hotelType -> canonical PropertyType value. Only types whose
# meaning is unambiguous; anything else leaves property_type unset so the
# property-type matcher falls back to the listing text.
_PROPERTY_TYPES = {
    "hotels": "hotel",
    "hotel": "hotel",
    "apartments": "apartment",
    "apartment": "apartment",
    "aparthotels": "aparthotel",
    "aparthotel": "aparthotel",
    "hostels": "hostel",
    "hostel": "hostel",
    "guest houses": "guest_house",
    "guest house": "guest_house",
    "bed and breakfasts": "bed_and_breakfast",
    "bed and breakfast": "bed_and_breakfast",
    "resorts": "resort",
    "resort": "resort",
    "villas": "villa",
    "villa": "villa",
    "holiday homes": "holiday_home",
    "holiday home": "holiday_home",
    "homestays": "homestay",
    "homestay": "homestay",
    "ryokans": "ryokan",
    "ryokan": "ryokan",
    "capsule hotels": "capsule_hotel",
    "capsule hotel": "capsule_hotel",
}

# policyAttributes / room attributes that are emitted only when True.
_PROPERTY_FACILITY_FLAGS = {
    "elevator": "Elevator",
    "fitnessCenter": "Fitness center",
    "pool": "Swimming pool",
    "hotTub": "Hot tub",
    "spaServices": "Spa services",
    "sauna": "Sauna",
    "nonSmokingProperty": "Non-smoking property",
}
_FOOD_AND_DRINK_FLAGS = {
    "restaurant": "Restaurant",
    "bar": "Bar",
    "roomService": "Room service",
    "breakfastAvailable": "Breakfast available",
}
_CURRENTLY_CLOSED_FLAGS = {"pool": "poolCurrentlyClosed", "sauna": "saunaCurrentlyClosed"}
_ROOM_ATTRIBUTE_FLAGS = {
    "television": "TV",
    "airConditioner": "Air conditioning",
    "hairDryer": "Hair dryer",
    "iron": "Iron",
    "hangers": "Hangers",
    "coffeeMaker": "Coffee maker",
    "crib": "Crib available",
    "laptopFriendlyWorkplace": "Laptop-friendly workspace",
    "kitchen": "Kitchen",
    "smokeDetector": "Smoke detector",
    "carbonMonoxideDetector": "Carbon monoxide detector",
}
_AREA_UNITS = {"m2": "m²", "m²": "m²", "sqm": "m²", "sq m": "m²", "sqft": "sq ft", "sq ft": "sq ft", "ft2": "sq ft"}


# --- small helpers -----------------------------------------------------------


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.I | re.S)
_BLOCK_TAG = re.compile(r"<\s*/?\s*(?:br|p|div|li|ul|ol|h[1-6]|tr|table)\b[^>]*>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


def clean_html_text(value: Any) -> str | None:
    """
    Plain text from provider HTML: block tags become sentence breaks, inline
    tags are dropped, entities are unescaped. A line without final
    punctuation (a heading such as "<strong>Outdoor pool</strong>") gets a
    full stop, so each line stays its own sentence for evidence quoting.
    """
    if not isinstance(value, str):
        return None
    text = _SCRIPT_OR_STYLE.sub(" ", value)
    text = _BLOCK_TAG.sub("\n", text)
    text = _ANY_TAG.sub(" ", text)
    text = html.unescape(text)
    sentences = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            sentences.append(line if line[-1] in ".!?:;" else f"{line}.")
    return " ".join(sentences) or None


def _area_label(size: Any, unit: Any) -> str | None:
    value = _number(size)
    unit_label = _AREA_UNITS.get((_text(unit) or "").casefold())
    if value is None or value <= 0 or unit_label is None:
        return None
    shown = int(value) if value.is_integer() else value
    return f"Room size: {shown} {unit_label}"


# --- offers ------------------------------------------------------------------


def _amount(value: Any) -> tuple[float, str] | None:
    """(amount, currency) of one {amount, currency} price object, else None."""
    price = _mapping(value)
    amount = _number(price.get("amount"))
    currency = _text(price.get("currency"))
    if amount is None or currency is None:
        return None
    return amount, currency.upper()


def _public_total(offer: Mapping[str, Any], requested_currency: str | None) -> tuple[float | None, str | None]:
    """
    The offer's SSP, or (None, None) when it cannot be stated unambiguously:
    missing or non-positive, in another currency than requested, or not
    matching the sum of the per-rate SSPs it is made of.
    """
    offer_ssp = _amount(offer.get("suggestedSellingPrice"))
    if offer_ssp is None or offer_ssp[0] <= 0:
        return None, None
    amount, currency = offer_ssp
    if requested_currency and currency != requested_currency.upper():
        return None, None

    rate_total = 0.0
    for rate in _items(offer.get("rates")):
        parts = [_amount(part) for part in (_mapping(rate.get("retailRate")).get("suggestedSellingPrice") or [])]
        if not parts or any(part is None or part[1] != currency for part in parts):
            return None, None
        rate_total += sum(part[0] for part in parts)

    if abs(rate_total - amount) > PRICE_TOLERANCE:
        return None, None
    return amount, currency


def _single(values: list[Any]) -> Any:
    """The common value when all entries agree, else None."""
    distinct = {value for value in values if value is not None}
    return distinct.pop() if len(distinct) == 1 and None not in values else None


def parse_public_offer(offer: Mapping[str, Any], *, requested_currency: str | None) -> PublicOffer:
    rates = _items(offer.get("rates"))
    public_total, currency = _public_total(offer, requested_currency)

    refundable_tags = [_text(_mapping(rate.get("cancellationPolicies")).get("refundableTag")) for rate in rates]
    refundable = {"RFN": True, "NRFN": False}.get(_single(refundable_tags))  # missing or mixed -> None

    # Kept as given, never interpreted (see PublicOffer.cancel_policy_times).
    cancel_policy_times = [
        " ".join(filter(None, [_text(info.get("cancelTime")), _text(info.get("timezone"))]))
        for rate in rates
        for info in _items(_mapping(rate.get("cancellationPolicies")).get("cancelPolicyInfos"))
        if _text(info.get("cancelTime"))
    ]

    excluded = [
        _text(tax.get("description"))
        for rate in rates
        for tax in _items(_mapping(rate.get("retailRate")).get("taxesAndFees"))
        if tax.get("included") is False
    ]

    return PublicOffer(
        room_name=_single([_text(rate.get("name")) for rate in rates]),
        board_type=_single([_text(rate.get("boardType")) for rate in rates]),
        board_name=_single([_text(rate.get("boardName")) for rate in rates]),
        refundable=refundable,
        cancel_policy_times=_unique(cancel_policy_times),
        max_occupancy=_integer(rates[0].get("maxOccupancy")) if len(rates) == 1 else None,
        public_total=public_total,
        currency=currency,
        excluded_taxes_and_fees=_unique([text for text in excluded if text]),
    )


def parse_rates_response(
    payload: Mapping[str, Any],
    *,
    requested_currency: str | None,
) -> dict[str, list[PublicOffer]]:
    """hotelId -> public offers, in the provider's order."""
    offers: dict[str, list[PublicOffer]] = {}
    for hotel in _items(payload.get("data")):
        hotel_id = _text(hotel.get("hotelId"))
        if hotel_id is None:
            continue
        parsed = [
            parse_public_offer(room_type, requested_currency=requested_currency)
            for room_type in _items(hotel.get("roomTypes"))
        ]
        if parsed:
            offers.setdefault(hotel_id, []).extend(parsed)
    return offers


def _offer_option(offer: PublicOffer) -> RoomOption:
    """
    Refundability is kept on the option's structured fields only, never in
    `choices`: choices are listing evidence, and matching reads them for the
    whole hotel. "Refundable rate" there could be read as free cancellation
    (a false YES), and "Non-refundable rate" would mark the hotel as having
    no free cancellation although only a few of its rates were fetched
    (maxRatesPerHotel) - a false NO. free_cancellation is never True: a
    refundable rate may still carry charges; a non-refundable rate is not
    freely cancellable, so it is False there.
    """
    return RoomOption(
        name=offer.room_name,
        price=offer.public_total,
        currency=offer.currency if offer.public_total is not None else None,
        persons=offer.max_occupancy,
        cancellation_type=offer.refundability_label,
        free_cancellation=False if offer.refundable is False else None,
        choices=[offer.board_name] if offer.board_name else [],
    )


# --- hotel content -----------------------------------------------------------


def _hotel_facilities(hotel: Mapping[str, Any]) -> list[Facility]:
    names = [_text(item.get("name")) for item in _items(hotel.get("facilities"))]
    if not any(names):
        names = [_text(name) for name in hotel.get("hotelFacilities") or []]
    facilities = [Facility(name=name) for name in _unique([name for name in names if name])]

    attributes = _mapping(hotel.get("policyAttributes"))
    known = {facility.name.casefold() for facility in facilities}
    for group, flags in (("facilities", _PROPERTY_FACILITY_FLAGS), ("foodAndDrink", _FOOD_AND_DRINK_FLAGS)):
        values = _mapping(attributes.get(group))
        for flag, name in flags.items():
            if values.get(flag) is not True:
                continue
            if flag == "pool" and _text(values.get("poolType")):
                name = f"{_text(values.get('poolType')).capitalize()} swimming pool"
            closed = _CURRENTLY_CLOSED_FLAGS.get(flag)
            overview = "Currently closed" if closed and values.get(closed) is True else None
            if name.casefold() in known and overview is None:
                continue
            known.add(name.casefold())
            facilities.append(Facility(name=name, category="property_attributes", overview=overview))
    return facilities


def _hotel_policies(hotel: Mapping[str, Any]) -> list[Policy]:
    times = _mapping(hotel.get("checkinCheckoutTimes"))
    attributes = _mapping(hotel.get("policyAttributes"))
    policies: list[Policy] = []

    def add(title: str, parts: list[str | None]) -> None:
        content = "; ".join(part for part in parts if part)
        if content:
            policies.append(Policy(title=title, content=content))

    check_in_start = _text(times.get("checkin_start")) or _text(_mapping(attributes.get("checkIn")).get("from"))
    check_in_end = _text(times.get("checkin_end")) or _text(_mapping(attributes.get("checkIn")).get("until"))
    add("Check-in", [
        f"Check-in from {check_in_start}" if check_in_start else None,
        f"until {check_in_end}" if check_in_end else None,
    ])
    check_out = _text(times.get("checkout")) or _text(_mapping(attributes.get("checkOut")).get("until"))
    add("Check-out", [f"Check-out until {check_out}" if check_out else None])

    parking = _mapping(attributes.get("parking"))
    parking_available = parking.get("selfParking") is True or parking.get("valet") is True
    add("Parking", [
        "Self parking available" if parking.get("selfParking") is True else None,
        "Valet parking available" if parking.get("valet") is True else None,
        "Free parking" if parking_available and parking.get("free") is True else None,
        "Parking charges apply"
        if parking.get("chargeMode") == "charges_apply" or _text(hotel.get("parking")) == "PAID"
        else None,
        "EV charger available" if parking.get("evCharger") is True else None,
    ])

    wifi = _mapping(attributes.get("wifi"))
    add("WiFi", [
        "WiFi in rooms" if wifi.get("inRoom") is True else None,
        "Free WiFi" if wifi.get("free") is True else None,
        "Free WiFi in public areas" if wifi.get("freeInPublicAreas") is True else None,
    ])

    pets = _mapping(attributes.get("pets"))
    add("Pets", ["Pets allowed" if pets.get("allowed") is True or hotel.get("petsAllowed") is True else None])

    children = _mapping(attributes.get("children"))
    add("Children", [
        "Children allowed" if children.get("allowed") is True or hotel.get("childAllowed") is True else None,
        "Cribs available" if children.get("cribAvailable") is True else None,
    ])

    minimum_age = _integer(attributes.get("minimumCheckInAge"))
    add("Check-in age", [f"Minimum check-in age: {minimum_age}" if minimum_age else None])
    return policies


def _nearby_places(hotel: Mapping[str, Any]) -> list[Highlight]:
    places = []
    for poi in _items(hotel.get("poi")):
        name = _text(poi.get("name"))
        if name is None:
            continue
        category = _text(poi.get("category"))
        distance = _number(poi.get("distanceKm"))
        label = f"{name} ({category})" if category else name
        places.append(f"{label}: {distance:g} km away" if distance is not None else label)
    return [Highlight(header="Nearby places", contents=places)] if places else []


def _content_room(room: Mapping[str, Any]) -> Room:
    amenities = [_text(item.get("name")) for item in _items(room.get("roomAmenities"))]
    names = [name for name in amenities if name]
    known = {name.casefold() for name in names}
    facilities = [Facility(name=name) for name in _unique(names)]

    for flag, name in _ROOM_ATTRIBUTE_FLAGS.items():
        if _mapping(room.get("attributes")).get(flag) is True and name.casefold() not in known:
            known.add(name.casefold())
            facilities.append(Facility(name=name, category="room_attributes"))

    for view in _items(room.get("views")):
        label = _text(view.get("view"))
        if label:
            facilities.append(Facility(name=label, category="room_view"))

    area = _area_label(room.get("roomSizeSquare"), room.get("roomSizeUnit"))
    if area:
        facilities.append(Facility(name=area, category="room_size"))

    beds = []
    for bed in _items(room.get("bedTypes")):
        bed_type = _text(bed.get("bedType"))
        if bed_type is None:
            continue
        quantity = _integer(bed.get("quantity"))
        size = _text(bed.get("bedSize"))
        beds.append(" ".join(filter(None, [f"{quantity} x" if quantity else None, bed_type, f"({size})" if size else None])))

    return Room(
        name=_text(room.get("roomName")),
        persons=_integer(room.get("maxOccupancy")),
        bed_types=[BedGroup(beds=beds)] if beds else [],
        facilities=facilities,
    )


def _rooms_with_offers(hotel: Mapping[str, Any], offers: list[PublicOffer]) -> list[Room]:
    """
    The rooms that can be booked for the requested stay - one per offered
    room name - in offer order. Catalog rooms without an offer are left out:
    matching reads every room, so their amenities, views or sizes would
    otherwise confirm a requirement for a room that is not on offer.

    An offer takes a catalog room's description only when exactly one
    catalog room has its name; with no match, or several, it gets a minimal
    Room built from the offer alone (name, occupancy, option), so nothing
    of another room is attributed to it.
    """
    catalog: dict[str, list[Room]] = {}
    for room in (_content_room(item) for item in _items(hotel.get("rooms"))):
        if room.name:
            catalog.setdefault(room.name.casefold(), []).append(room)

    rooms: list[Room] = []
    by_name: dict[str, Room] = {}
    for offer in offers:
        key = offer.room_name.casefold() if offer.room_name else None
        room = by_name.get(key) if key else None
        if room is None:
            matches = catalog.get(key, []) if key else []
            room = matches[0] if len(matches) == 1 else Room(name=offer.room_name, persons=offer.max_occupancy)
            rooms.append(room)
            if key:
                by_name[key] = room
        room.available = True
        room.options.append(_offer_option(offer))
    return rooms


def _review_summary(hotel: Mapping[str, Any]) -> ReviewSummary | None:
    sentiment = _mapping(hotel.get("sentiment_analysis"))
    pros = [text for text in (_text(item) for item in sentiment.get("pros") or []) if text]
    cons = [text for text in (_text(item) for item in sentiment.get("cons") or []) if text]
    categories = [
        ReviewCategoryScore(
            name=_text(item.get("name")),
            rating=_number(item.get("rating")),
            description=_text(item.get("description")),
        )
        for item in _items(sentiment.get("categories"))
        if _text(item.get("name"))
    ]
    if not (pros or cons or categories):
        return None
    return ReviewSummary(
        source=REVIEW_SUMMARY_SOURCE,
        updated_at=_text(hotel.get("sentiment_updated_at")),
        pros=pros,
        cons=cons,
        categories=categories,
    )


def normalize_liteapi_listing(
    hotel: Mapping[str, Any],
    offers: list[PublicOffer],
    *,
    environment: str | None = None,
    fetched_at: datetime | None = None,
) -> ListingRaw:
    """
    Convert one /data/hotel `data` object plus its public offers into
    ListingRaw. Raises ValueError when the hotel has no id.
    """
    hotel_id = _text(hotel.get("id"))
    if hotel_id is None:
        raise ValueError("LiteAPI hotel without id")

    location = _mapping(hotel.get("location"))
    prices = [(offer.public_total, offer.currency) for offer in offers if offer.public_total is not None]
    best_price = min(prices, key=lambda item: item[0]) if prices else (None, None)
    hotel_type = (_text(hotel.get("hotelType")) or "").casefold()

    rooms = _rooms_with_offers(hotel, offers)
    occupancies = [room.persons for room in rooms if room.persons]

    return ListingRaw(
        id=hotel_id,
        city=_text(hotel.get("city")),
        address=_text(hotel.get("address")),
        name=_text(hotel.get("name")),
        url=None,  # no booking deep link yet
        price=best_price[0],
        currency=best_price[1],
        rating=_number(hotel.get("rating")),
        stars=_integer(hotel.get("starRating")),
        property_type=_PROPERTY_TYPES.get(hotel_type),
        max_occupancy=max(occupancies) if occupancies else None,
        description=clean_html_text(hotel.get("hotelDescription")),
        fine_print=clean_html_text(hotel.get("hotelImportantInformation")),
        policies=_hotel_policies(hotel),
        highlights=_nearby_places(hotel),
        facilities=_hotel_facilities(hotel),
        rooms=rooms,
        latitude=_number(location.get("latitude")),
        longitude=_number(location.get("longitude")),
        review_count=_integer(hotel.get("reviewCount")),
        review_summary=_review_summary(hotel),
        public_offers=offers,
        provenance=ListingProvenance(
            provider=PROVIDER,
            environment=environment,
            fetched_at=fetched_at or datetime.now(timezone.utc),
        ),
        raw=None,
    )


# --- retriever ---------------------------------------------------------------


class LiteApiRetriever:
    """
    CandidateRetriever over LiteAPI for one destination country.

    SearchRequest carries a city but no country, and LiteAPI needs both, so
    the country is configured on the retriever and only the configured
    cities are searched - no country is ever assumed for another city. Such
    a search is refused with RetrievalError("unsupported_destination").

    Occupancy: one room, adults only (LiteAPI needs children's ages, which
    SearchRequest does not hold), and, when required_adults is set, exactly
    that many adults. Anything else is refused with
    RetrievalError("unsupported_request") rather than guessed. Both refusals
    happen before any HTTP call and carry a user-safe message.
    """

    def __init__(
        self,
        client: LiteApiClient | None = None,
        *,
        country_code: str | None = None,
        supported_cities: list[str] | None = None,
        required_adults: int | None = None,
        guest_nationality: str = "US",
        rates_limit: int = DEFAULT_RATES_LIMIT,
        max_detail_calls: int = DEFAULT_MAX_DETAIL_CALLS,
        detail_concurrency: int = 3,
        max_rates_per_hotel: int = 3,
        provider_timeout_seconds: int = 10,
        environment: str = "sandbox",
    ) -> None:
        self._client = client
        self._country_code = country_code
        self._supported_cities = [city.strip() for city in supported_cities or [] if city.strip()]
        self._required_adults = required_adults
        self._guest_nationality = guest_nationality
        self._rates_limit = max(1, rates_limit)
        self._max_detail_calls = max(0, max_detail_calls)
        self._detail_concurrency = max(1, detail_concurrency)
        self._max_rates_per_hotel = max(1, max_rates_per_hotel)
        self._provider_timeout_seconds = provider_timeout_seconds
        self._environment = environment

    @classmethod
    def from_env(cls, client: LiteApiClient | None = None) -> LiteApiRetriever:
        """
        Retriever configured for the current live-data scope. Defaults are
        the scope validated in the sandbox smoke: Barcelona (ES), one room,
        two adults, rates for up to 20 hotels, details for the first 5.
        """
        cities = os.getenv("LITEAPI_SUPPORTED_CITIES", DEFAULT_SUPPORTED_CITIES)
        return cls(
            client,
            country_code=(os.getenv("LITEAPI_COUNTRY_CODE", DEFAULT_COUNTRY_CODE).strip().upper() or None),
            supported_cities=cities.split(","),
            required_adults=REQUIRED_ADULTS,
            rates_limit=_env_int("LITEAPI_RATES_LIMIT", DEFAULT_RATES_LIMIT),
            max_detail_calls=_env_int("LITEAPI_MAX_DETAIL_CALLS", DEFAULT_MAX_DETAIL_CALLS),
        )

    def _check_supported(self, req: SearchRequest) -> None:
        if not self._country_code:
            raise RetrievalError("configuration_error")

        cities = self._supported_cities
        if cities and req.city.strip().casefold() not in {city.casefold() for city in cities}:
            raise RetrievalError(
                "unsupported_destination",
                user_message=(
                    f"Live hotel search currently covers only {', '.join(cities)}. "
                    "Your current search has not been changed."
                ),
            )

        adults_ok = self._required_adults is None or req.adults == self._required_adults
        if req.children or req.rooms != 1 or not adults_ok:
            guests = f"{self._required_adults} adults" if self._required_adults else "adults"
            raise RetrievalError(
                "unsupported_request",
                user_message=(
                    f"Live hotel search currently supports one room for {guests}, without children. "
                    "Your current search has not been changed."
                ),
            )

    async def get_candidates(
        self,
        req: SearchRequest,
        max_items: int,
        trace: RequestTrace | None = None,
    ) -> list[ListingRaw]:
        if not req.city or req.check_in is None or req.check_out is None:
            raise ValueError("LiteAPI search requires city, check_in and check_out")
        self._check_supported(req)

        currency = _request_currency(req.currency)
        client = self._client or LiteApiClient.from_env()
        try:
            payload = await client.search_rates(
                {
                    "cityName": req.city,
                    "countryCode": self._country_code,
                    "checkin": req.check_in.isoformat(),
                    "checkout": req.check_out.isoformat(),
                    "occupancies": [{"adults": req.adults, "children": []}],
                    "currency": currency,
                    "guestNationality": self._guest_nationality,
                    "maxRatesPerHotel": self._max_rates_per_hotel,
                    "limit": min(max_items, self._rates_limit),
                    "timeout": self._provider_timeout_seconds,
                },
                trace=trace,
            )
            if "data" not in payload and payload.get("error") is not None:
                # Error object instead of results; the body is not logged.
                raise RetrievalError("provider_error")

            offers_by_hotel = parse_rates_response(payload, requested_currency=currency)
            del payload
            hotel_ids = list(offers_by_hotel)[: min(max_items, self._max_detail_calls)]
            fetched_at = datetime.now(timezone.utc)

            semaphore = asyncio.Semaphore(self._detail_concurrency)

            async def details(hotel_id: str) -> Mapping[str, Any] | RetrievalError:
                async with semaphore:
                    try:
                        response = await client.get_hotel_details(
                            hotel_id, timeout_seconds=self._provider_timeout_seconds, trace=trace
                        )
                    except RetrievalError as exc:
                        return exc
                return _mapping(response.get("data"))

            results = await asyncio.gather(*(details(hotel_id) for hotel_id in hotel_ids))
        finally:
            if self._client is None:
                await client.aclose()

        listings: list[ListingRaw] = []
        errors: list[RetrievalError] = []
        for hotel_id, hotel in zip(hotel_ids, results):
            if isinstance(hotel, RetrievalError):
                logger.warning("LiteAPI details unavailable for hotel %s: %s", hotel_id, hotel.code)
                errors.append(hotel)
                continue
            if not hotel:
                logger.warning("LiteAPI returned no content for hotel %s; skipped", hotel_id)
                continue
            try:
                listings.append(
                    normalize_liteapi_listing(
                        hotel,
                        offers_by_hotel[hotel_id],
                        environment=self._environment,
                        fetched_at=fetched_at,
                    )
                )
            except Exception:
                logger.exception("Failed to normalize LiteAPI hotel %s", hotel_id)

        if not listings and errors:
            raise errors[0]
        return listings


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _request_currency(value: str | None) -> str:
    code = (value or "").strip().upper()
    return code if re.fullmatch(r"[A-Z]{3}", code) else "USD"
