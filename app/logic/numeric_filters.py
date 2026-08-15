from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Optional, Tuple
from app.schemas.filters import SearchFilters
from app.schemas.listing import ListingRaw
from app.schemas.match import Evidence, EvidenceSource, Ternary
from app.services.currency_rates import convert_amount_to_usd


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_WORD_TO_NUMBER = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
}

@dataclass(frozen=True)
class NumericMatchResult:
    attribute: str
    value: Ternary
    actual_value: float | int | None
    evidence: List[Evidence]
    why: str


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _word_to_number(token: str) -> Optional[int]:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _collect_text_candidates(
    listing: ListingRaw,
) -> List[Tuple[str, str]]:
    """
    Collect canonical textual evidence that may
    contain bedroom count, bathroom count or area.

    Important:
    - room.name may contain "Three-Bedroom Apartment"
    - room facilities may contain "2 bathrooms" or "85 m²"
    - bed_types are intentionally NOT used here:
      number of beds is not number of bedrooms
    """

    out: List[Tuple[str, str]] = []

    def add(
        path: str,
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            return

        value = value.strip()

        if not value:
            return

        out.append(
            (
                _normalize_text(value),
                path,
            )
        )

    add(
        "listing.name",
        listing.name,
    )

    add(
        "listing.description",
        listing.description,
    )

    for i, room in enumerate(
        listing.rooms or []
    ):
        add(
            f"rooms[{i}].name",
            room.name,
        )

        for j, facility in enumerate(
            room.facilities or []
        ):
            if isinstance(facility, str):
                add(
                    (
                        f"rooms[{i}]"
                        f".facilities[{j}]"
                    ),
                    facility,
                )
                continue

            add(
                (
                    f"rooms[{i}]"
                    f".facilities[{j}].name"
                ),
                facility.name,
            )

            add(
                (
                    f"rooms[{i}]"
                    f".facilities[{j}].overview"
                ),
                facility.overview,
            )

    return out


def _parse_bedroom_mentions(text: str) -> List[int]:
    found: List[int] = []

    for m in re.finditer(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[-\s]+bedroom(?:s)?\b",
        text,
    ):
        n = _word_to_number(m.group(1))
        if n is not None:
            found.append(n)
    for m in re.finditer(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+bed\b",
        text,
    ):
        n = _word_to_number(m.group(1))
        if n is not None:
            found.append(n)

    # studio = 0 bedrooms
    if re.search(r"\bstudio\b", text):
        found.append(0)

    return found


def extract_bedroom_count(listing: ListingRaw) -> tuple[int | None, List[Evidence]]:
    """
    Пытаемся детерминированно извлечь количество bedrooms.

    Эвристика:
    - ищем bedroom mentions в listing.name / description / rooms[].*
    - если найдено несколько кандидатов:
        1) предпочитаем rooms[*] и listing.name
        2) среди них берём максимальное значение
    """
    candidates: List[tuple[int, Evidence, int]] = []

    for text, path in _collect_text_candidates(listing):
        values = _parse_bedroom_mentions(text)
        for val in values:
            priority = 1 if path.startswith("rooms[") or path == "listing.name" else 0
            candidates.append(
                (
                    val,
                    Evidence(
                        source=EvidenceSource.STRUCTURED,
                        path=path,
                        snippet=text,
                    ),
                    priority,
                )
            )

    if not candidates:
        return None, []

    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    best_value, best_evidence, _ = candidates[0]
    return best_value, [best_evidence]

_BATHROOM_NUMERIC_RE = re.compile(
    r"\b(?P<count>\d+(?:\.\d+)?)\s*(?:bathroom|bathrooms|bath|baths)\b",
    flags=re.IGNORECASE,
)

_BATHROOM_WORD_RE = re.compile(
    r"\b(?P<count_word>one|two|three|four|five|six)\s*(?:bathroom|bathrooms|bath|baths)\b",
    flags=re.IGNORECASE,
)


def _extract_bathroom_mentions_from_text(
    text: str,
    *,
    source: EvidenceSource,
    path: str,
) -> list[tuple[float, Evidence]]:
    out: list[tuple[float, Evidence]] = []
    if not text:
        return out

    for m in _BATHROOM_NUMERIC_RE.finditer(text):
        try:
            count = float(m.group("count"))
        except (TypeError, ValueError):
            continue

        snippet = m.group(0)
        out.append(
            (
                count,
                Evidence(
                    source=source,
                    path=path,
                    snippet=snippet,
                ),
            )
        )

    for m in _BATHROOM_WORD_RE.finditer(text):
        word = (m.group("count_word") or "").lower()
        count = _WORD_TO_NUMBER.get(word)
        if count is None:
            continue

        snippet = m.group(0)
        out.append(
            (
                count,
                Evidence(
                    source=source,
                    path=path,
                    snippet=snippet,
                ),
            )
        )

    return out


def extract_bathroom_count(
    listing: ListingRaw,
) -> tuple[
    float | None,
    list[Evidence],
]:
    """
    Extract bathroom count from canonical
    listing and room evidence.

    Sources:
    - listing name
    - listing description
    - room name
    - room facility name / overview
    """

    candidates: list[
        tuple[float, Evidence]
    ] = []

    for text, path in (
        _collect_text_candidates(listing)
    ):
        candidates.extend(
            _extract_bathroom_mentions_from_text(
                text,
                source=EvidenceSource.STRUCTURED,
                path=path,
            )
        )

    if not candidates:
        return None, []

    best_count, best_evidence = max(
        candidates,
        key=lambda x: x[0],
    )

    return best_count, [best_evidence]

_AREA_PATTERNS = [
    (
        re.compile(
            r"\b(\d+(?:[\.,]\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2|square meters?|square metres?)\b"
        ),
        1.0,
    ),
    (
        re.compile(
            r"\b(\d+(?:[\.,]\d+)?)\s*(?:sq\.?\s*ft|ft²|ft2|square feet|feet²)\b"
        ),
        0.092903,
    ),
]


def _parse_area_mentions(text: str) -> List[float]:
    out: List[float] = []

    for pattern, multiplier in _AREA_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1).replace(",", ".")
            try:
                area = round(float(raw) * multiplier, 2)
                out.append(area)
            except ValueError:
                continue

    return out


def extract_area_sqm(listing: ListingRaw) -> tuple[float | None, List[Evidence]]:
    """
    Пытаемся извлечь площадь в sqm.

    Поддерживаем:
    - 80 sqm
    - 80 m² / m2
    - 2196 sq ft / feet²  -> конвертируем в sqm
    """
    candidates: List[tuple[float, Evidence, int]] = []

    for text, path in _collect_text_candidates(listing):
        values = _parse_area_mentions(text)
        for val in values:
            priority = 1 if path.startswith("rooms[") or path == "listing.name" else 0
            candidates.append(
                (
                    val,
                    Evidence(
                        source=EvidenceSource.STRUCTURED,
                        path=path,
                        snippet=text,
                    ),
                    priority,
                )
            )

    if not candidates:
        return None, []

    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    best_value, best_evidence, _ = candidates[0]
    return best_value, [best_evidence]


def match_bedrooms_filters(
    bedroom_count: int | None,
    filters: SearchFilters | None,
    evidence: List[Evidence] | None = None,
) -> NumericMatchResult | None:
    if filters is None or (filters.bedrooms_min is None and filters.bedrooms_max is None):
        return None

    evidence = evidence or []

    if bedroom_count is None:
        return NumericMatchResult(
            attribute="bedrooms",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="BEDROOMS: could not extract bedroom count",
        )

    if filters.bedrooms_min is not None and bedroom_count < filters.bedrooms_min:
        return NumericMatchResult(
            attribute="bedrooms",
            value=Ternary.NO,
            actual_value=bedroom_count,
            evidence=evidence,
            why=f"BEDROOMS: {bedroom_count} < required min {filters.bedrooms_min}",
        )

    if filters.bedrooms_max is not None and bedroom_count > filters.bedrooms_max:
        return NumericMatchResult(
            attribute="bedrooms",
            value=Ternary.NO,
            actual_value=bedroom_count,
            evidence=evidence,
            why=f"BEDROOMS: {bedroom_count} > allowed max {filters.bedrooms_max}",
        )

    if filters.bedrooms_min is not None and filters.bedrooms_max is not None:
        why = f"BEDROOMS: {bedroom_count} within [{filters.bedrooms_min}, {filters.bedrooms_max}]"
    elif filters.bedrooms_min is not None:
        why = f"BEDROOMS: {bedroom_count} >= required {filters.bedrooms_min}"
    else:
        why = f"BEDROOMS: {bedroom_count} <= allowed {filters.bedrooms_max}"

    return NumericMatchResult(
        attribute="bedrooms",
        value=Ternary.YES,
        actual_value=bedroom_count,
        evidence=evidence,
        why=why,
    )


def match_area_filters(
    area_sqm: float | None,
    filters: SearchFilters | None,
    evidence: List[Evidence] | None = None,
) -> NumericMatchResult | None:
    if filters is None or (filters.area_sqm_min is None and filters.area_sqm_max is None):
        return None

    evidence = evidence or []

    if area_sqm is None:
        return NumericMatchResult(
            attribute="area_sqm",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="AREA: could not extract area",
        )

    pretty = int(area_sqm) if float(area_sqm).is_integer() else round(area_sqm, 1)

    if filters.area_sqm_min is not None and area_sqm < filters.area_sqm_min:
        return NumericMatchResult(
            attribute="area_sqm",
            value=Ternary.NO,
            actual_value=area_sqm,
            evidence=evidence,
            why=f"AREA: {pretty} sqm < required min {filters.area_sqm_min}",
        )

    if filters.area_sqm_max is not None and area_sqm > filters.area_sqm_max:
        return NumericMatchResult(
            attribute="area_sqm",
            value=Ternary.NO,
            actual_value=area_sqm,
            evidence=evidence,
            why=f"AREA: {pretty} sqm > allowed max {filters.area_sqm_max}",
        )

    if filters.area_sqm_min is not None and filters.area_sqm_max is not None:
        why = f"AREA: {pretty} sqm within [{filters.area_sqm_min}, {filters.area_sqm_max}]"
    elif filters.area_sqm_min is not None:
        why = f"AREA: {pretty} sqm >= required {filters.area_sqm_min}"
    else:
        why = f"AREA: {pretty} sqm <= allowed {filters.area_sqm_max}"

    return NumericMatchResult(
        attribute="area_sqm",
        value=Ternary.YES,
        actual_value=area_sqm,
        evidence=evidence,
        why=why,
    )
    
def match_bathroom_filters(
    bathroom_count: float | None,
    filters: SearchFilters | None,
    evidence: List[Evidence] | None = None,
) -> NumericMatchResult | None:
    if filters is None:
        return None

    if filters.bathrooms_min is None and filters.bathrooms_max is None:
        return None

    evidence = evidence or []

    if bathroom_count is None:
        return NumericMatchResult(
            attribute="bathrooms",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="BATHROOMS: could not extract bathroom count",
        )

    if filters.bathrooms_min is not None and bathroom_count < filters.bathrooms_min:
        return NumericMatchResult(
            attribute="bathrooms",
            value=Ternary.NO,
            actual_value=bathroom_count,
            evidence=evidence,
            why=f"BATHROOMS: {bathroom_count} < required min {filters.bathrooms_min}",
        )

    if filters.bathrooms_max is not None and bathroom_count > filters.bathrooms_max:
        return NumericMatchResult(
            attribute="bathrooms",
            value=Ternary.NO,
            actual_value=bathroom_count,
            evidence=evidence,
            why=f"BATHROOMS: {bathroom_count} > allowed max {filters.bathrooms_max}",
        )

    if filters.bathrooms_min is not None and filters.bathrooms_max is not None:
        why = (
            f"BATHROOMS: {bathroom_count} within range "
            f"[{filters.bathrooms_min}, {filters.bathrooms_max}]"
        )
    elif filters.bathrooms_min is not None:
        why = f"BATHROOMS: {bathroom_count} >= required min {filters.bathrooms_min}"
    else:
        why = f"BATHROOMS: {bathroom_count} <= allowed max {filters.bathrooms_max}"

    return NumericMatchResult(
        attribute="bathrooms",
        value=Ternary.YES,
        actual_value=bathroom_count,
        evidence=evidence,
        why=why,
    )    
    
def _normalize_currency(value: str | None) -> str | None:
    if not value:
        return None

    s = value.strip().upper()

    mapping = {
        "US$": "USD",
        "$": "USD",
        "USD": "USD",
        "AZN": "AZN",
        "MANAT": "AZN",
        "MANATS": "AZN",
        "EUR": "EUR",
        "€": "EUR",
        "GBP": "GBP",
        "£": "GBP",
    }
    return mapping.get(s, s)


def _night_count(check_in: date | None, check_out: date | None) -> int | None:
    if check_in is None or check_out is None:
        return None
    delta = (check_out - check_in).days
    if delta <= 0:
        return None
    return delta


def extract_total_price(listing: ListingRaw) -> tuple[float | None, str | None, List[Evidence]]:
    """
    Booking JSON in your current flow appears to expose total stay price at top-level:
    listing.price + listing.currency.
    """
    price = getattr(listing, "price", None)
    currency = _normalize_currency(getattr(listing, "currency", None))

    if price is not None:
        evidence = [
            Evidence(
                source=EvidenceSource.STRUCTURED,
                path="listing.price",
                snippet=f"price={price}, currency={currency}",
            )
        ]
        return float(price), currency, evidence

    # fallback: first room option price
    rooms = getattr(listing, "rooms", []) or []
    for i, room in enumerate(rooms):
        options = getattr(room, "options", []) or []
        for j, opt in enumerate(options):
            opt_price = getattr(opt, "price", None)
            opt_currency = _normalize_currency(getattr(opt, "currency", None))
            if opt_price is not None:
                evidence = [
                    Evidence(
                        source=EvidenceSource.STRUCTURED,
                        path=f"rooms[{i}].options[{j}].price",
                        snippet=f"price={opt_price}, currency={opt_currency}",
                    )
                ]
                return float(opt_price), opt_currency, evidence

    return None, None, []


def match_price_filters(
    total_price: float | None,
    listing_currency: str | None,
    filters: SearchFilters | None,
    *,
    check_in: date | None,
    check_out: date | None,
    evidence: List[Evidence] | None = None,
) -> NumericMatchResult | None:
    if filters is None or filters.price is None:
        return None
    
    price_filter = filters.price

    if (
        price_filter.min_amount is None
        and price_filter.max_amount is None
    ):
        return None

    price_filter = filters.price
    evidence = evidence or []

    if total_price is None:
        return NumericMatchResult(
            attribute="price_total",
            value=Ternary.UNCERTAIN,
            actual_value=None,
            evidence=evidence,
            why="PRICE: could not extract total listing price",
        )

    listing_currency_norm = _normalize_currency(listing_currency)
    request_currency_norm = _normalize_currency(price_filter.currency)

    required_min_total = price_filter.min_amount
    required_max_total = price_filter.max_amount

    if request_currency_norm is not None and listing_currency_norm == "USD" and request_currency_norm != "USD":
        converted_min, fx_snapshot = (
            convert_amount_to_usd(required_min_total, request_currency_norm)
            if required_min_total is not None
            else (None, None)
        )
        converted_max, fx_snapshot_max = (
            convert_amount_to_usd(required_max_total, request_currency_norm)
            if required_max_total is not None
            else (None, None)
        )

        fx_snapshot = fx_snapshot or fx_snapshot_max

        if (required_min_total is not None and converted_min is None) or (
            required_max_total is not None and converted_max is None
        ):
            stale_note = " using stale cached FX rates" if fx_snapshot and fx_snapshot.is_stale else ""
            return NumericMatchResult(
                attribute="price_total",
                value=Ternary.UNCERTAIN,
                actual_value=total_price,
                evidence=evidence,
                why=(
                    f"PRICE: could not convert request currency {request_currency_norm} to USD"
                    f" for provider-side comparison{stale_note}"
                ),
            )

        required_min_total = converted_min
        required_max_total = converted_max

    elif (
        request_currency_norm is not None
        and listing_currency_norm is not None
        and request_currency_norm != listing_currency_norm
    ):
        return NumericMatchResult(
            attribute="price_total",
            value=Ternary.UNCERTAIN,
            actual_value=total_price,
            evidence=evidence,
            why=(
                f"PRICE: unsupported currency comparison listing={listing_currency_norm}, "
                f"request={request_currency_norm}"
            ),
        )

    if price_filter.scope == "per_night":
        nights = _night_count(check_in, check_out)
        if nights is None:
            return NumericMatchResult(
                attribute="price_total",
                value=Ternary.UNCERTAIN,
                actual_value=total_price,
                evidence=evidence,
                why="PRICE: could not derive night count for per-night budget",
            )

        if required_min_total is not None:
            required_min_total = required_min_total * nights
        if required_max_total is not None:
            required_max_total = required_max_total * nights

    elif price_filter.scope in ("total_stay", None):
        pass

    else:
        return NumericMatchResult(
            attribute="price_total",
            value=Ternary.UNCERTAIN,
            actual_value=total_price,
            evidence=evidence,
            why=f"PRICE: unsupported scope {price_filter.scope}",
        )

    pretty_total = round(total_price, 2)

    if required_min_total is not None and total_price < required_min_total:
        return NumericMatchResult(
            attribute="price_total",
            value=Ternary.NO,
            actual_value=total_price,
            evidence=evidence,
            why=f"PRICE: {pretty_total} < required min total {round(required_min_total, 2)}",
        )

    if required_max_total is not None and total_price > required_max_total:
        return NumericMatchResult(
            attribute="price_total",
            value=Ternary.NO,
            actual_value=total_price,
            evidence=evidence,
            why=f"PRICE: {pretty_total} > allowed max total {round(required_max_total, 2)}",
        )

    if required_min_total is not None and required_max_total is not None:
        why = (
            f"PRICE: {pretty_total} within total range "
            f"[{round(required_min_total, 2)}, {round(required_max_total, 2)}]"
        )
    elif required_min_total is not None:
        why = f"PRICE: {pretty_total} >= required total {round(required_min_total, 2)}"
    else:
        why = f"PRICE: {pretty_total} <= allowed total {round(required_max_total, 2)}"

    return NumericMatchResult(
        attribute="price_total",
        value=Ternary.YES,
        actual_value=total_price,
        evidence=evidence,
        why=why,
    )   
   
    
def evaluate_numeric_filters(
    listing: ListingRaw,
    filters: SearchFilters | None,
    *,
    check_in: date | None = None,
    check_out: date | None = None,
) -> List[NumericMatchResult]:
    """
    Единая точка входа для numeric extraction + matching.
    """
    bedroom_count, bedroom_evidence = extract_bedroom_count(listing)
    area_sqm, area_evidence = extract_area_sqm(listing)
    bathroom_count, bathroom_evidence = extract_bathroom_count(listing)
    total_price, listing_currency, price_evidence = extract_total_price(listing)

    results: List[NumericMatchResult] = []

    br = match_bedrooms_filters(
        bedroom_count=bedroom_count,
        filters=filters,
        evidence=bedroom_evidence,
    )
    if br is not None:
        results.append(br)

    ar = match_area_filters(
        area_sqm=area_sqm,
        filters=filters,
        evidence=area_evidence,
    )
    if ar is not None:
        results.append(ar)

    ba = match_bathroom_filters(
        bathroom_count=bathroom_count,
        filters=filters,
        evidence=bathroom_evidence,
    )
    if ba is not None:
        results.append(ba)

    pr = match_price_filters(
        total_price=total_price,
        listing_currency=listing_currency,
        filters=filters,
        check_in=check_in,
        check_out=check_out,
        evidence=price_evidence,
    )
    if pr is not None:
        results.append(pr)

    return results

