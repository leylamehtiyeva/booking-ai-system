from __future__ import annotations

from typing import Any, Mapping

from app.schemas.listing import ListingRaw


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")

        if not cleaned:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

        return None

    if isinstance(value, str):
        cleaned = value.strip()

        if not cleaned:
            return None

        try:
            number = float(cleaned)
        except ValueError:
            return None

        if number.is_integer():
            return int(number)

    return None


def normalize_fast_listing(
    item: Mapping[str, Any],
    *,
    requested_city: str | None = None,
) -> ListingRaw:
    """
    Convert one Voyager Fast result into the internal ListingRaw model.

    Provider field mapping:

        name      -> name
        url       -> url
        price     -> price
        currency  -> currency
        rating    -> rating
        stars     -> stars
        roomType  -> room_type
        persons   -> max_occupancy

    Rich fields such as description, facilities and rooms are not
    invented when Fast does not provide them.

    The complete provider payload is preserved in `raw`.
    """

    return ListingRaw(
        city=_clean_text(requested_city),
        name=_clean_text(item.get("name")),
        url=_clean_text(item.get("url")),
        price=_safe_float(item.get("price")),
        currency=_clean_text(item.get("currency")),
        rating=_safe_float(item.get("rating")),
        stars=_safe_int(item.get("stars")),
        room_type=_clean_text(item.get("roomType")),
        max_occupancy=_safe_int(item.get("persons")),
        raw=dict(item),
    )