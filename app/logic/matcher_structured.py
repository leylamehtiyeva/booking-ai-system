from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import Evidence, EvidenceSource, FieldMatch, MatchReport, Ternary
from app.schemas.query import SearchRequest
from app.logic.field_rules import FIELD_RULES
from app.logic.listing_signals import collect_listing_signals, find_best_negative_signal_match, find_best_signal_match




def _priority_value(priority) -> str:
    return getattr(priority, "value", priority)


def _mapping_status_value(mapping_status) -> str:
    return getattr(mapping_status, "value", mapping_status)


def _constraints_by_priority(req):
    constraints = list(req.constraints or [])

    must_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "must"
    ]
    nice_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "nice"
    ]
    forbidden_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "forbidden"
    ]
    return must_constraints, nice_constraints, forbidden_constraints


def _known_mapped_fields(constraints) -> list[Field]:
    out: list[Field] = []
    seen: set[Field] = set()

    for c in constraints:
        if _mapping_status_value(getattr(c, "mapping_status", None)) != "known":
            continue

        for f in getattr(c, "mapped_fields", []) or []:
            if f not in seen:
                seen.add(f)
                out.append(f)

    return out

def _normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def extract_facility_texts(facilities: List[Any]) -> List[str]:
    """
    facilities can be:
    - list[str]
    - list[dict] with keys like {"name": "..."}
    - mixed
    We normalize into list[str].
    """
    out: List[str] = []
    for x in facilities or []:
        if isinstance(x, str):
            out.append(_normalize_text(x))
        elif isinstance(x, dict):
            name = x.get("name")
            if isinstance(name, str) and name.strip():
                out.append(_normalize_text(name))
    return out


def collect_all_facilities(listing: ListingRaw) -> List[Tuple[str, str]]:
    """
    Collect facilities from listing + rooms with evidence path.
    Returns list of (facility_text, evidence’s "path").
    """
    results: List[Tuple[str, str]] = []

    # listing-level
    for t in extract_facility_texts(getattr(listing, "facilities", []) or []):
        results.append((t, "listing.facilities"))

    # rooms-level
    for i, room in enumerate(getattr(listing, "rooms", []) or []):
        room_fac = extract_facility_texts(getattr(room, "facilities", []) or [])
        for t in room_fac:
            results.append((t, f"rooms[{i}].facilities"))

    return results


@dataclass(frozen=True)
class KeywordRule:
    field: Field
    keywords: Tuple[str, ...]


def build_rules() -> List[KeywordRule]:
    """
    Minimal keyword rules for structured facilities matching.

    Important: this is NOT a user synonym dictionary.
    These are Booking-style amenity strings we expect to see in structured data.
    """
    return [
        KeywordRule(Field.KITCHEN, ("kitchen", "kitchenette", "cooking")),
        KeywordRule(Field.KETTLE, ("kettle", "electric kettle")),
        KeywordRule(Field.PRIVATE_BATHROOM, ("private bathroom",)),
        KeywordRule(Field.WIFI, ("wifi", "wi-fi", "wireless internet")),
        KeywordRule(Field.AIR_CONDITIONING, ("air conditioning", "ac")),
    ]


def match_field_in_facilities(
    field: Field,
    facilities_with_paths: List[Tuple[str, str]],
    rules: List[KeywordRule],
) -> FieldMatch:
    rule = next((r for r in rules if r.field == field), None)

    # 1) no rule -> cannot decide deterministically
    if rule is None:
        return FieldMatch(value=Ternary.UNCERTAIN, confidence=0.0, evidence=[])

    # 2) rule exists -> search
    for fac_text, path in facilities_with_paths:
        for kw in rule.keywords:
            if kw in fac_text:
                return FieldMatch(
                    value=Ternary.YES,
                    confidence=0.95,
                    evidence=[
                        Evidence(
                            source=EvidenceSource.STRUCTURED,
                            path=path,
                            snippet=fac_text,
                        )
                    ],
                )

    return FieldMatch(value=Ternary.NO, confidence=0.7, evidence=[])


def match_listing_structured(listing: ListingRaw, request: SearchRequest) -> MatchReport:
    must_constraints, nice_constraints, _ = _constraints_by_priority(request)

    must_fields = _known_mapped_fields(must_constraints)
    nice_fields = _known_mapped_fields(nice_constraints)

    requested_fields = list({*must_fields, *nice_fields})

    field_matches = {
        f: _match_field_via_rules(listing, f)
        for f in requested_fields
    }

    hard_fail = [
        f
        for f in must_fields
        if field_matches.get(f) and field_matches[f].value == Ternary.NO
    ]

    return MatchReport(
        listing_id=listing.id or listing.url or listing.name or "unknown",
        matches=field_matches,
        hard_fail_fields=hard_fail,
    )


def _match_field_via_rules(listing: ListingRaw, field: Field) -> FieldMatch:
    signals = collect_listing_signals(listing)
    rule = FIELD_RULES.get(field)

    if rule is None:
        return FieldMatch(value=Ternary.UNCERTAIN, evidence=[])

    best_positive = find_best_signal_match(
        signals=signals,
        aliases=rule.aliases,
        preferred_path_prefixes=rule.preferred_path_prefixes,
    )
    if best_positive is not None:
        return FieldMatch(
            value=Ternary.YES,
            confidence=0.95,
            evidence=[
                Evidence(
                    source=EvidenceSource.STRUCTURED,
                    path=best_positive.path,
                    snippet=best_positive.raw_text,
                )
            ],
        )

    best_negative = find_best_negative_signal_match(
        signals=signals,
        negative_aliases=rule.negative_aliases,
        preferred_path_prefixes=rule.preferred_path_prefixes,
    )
    if best_negative is not None:
        return FieldMatch(
            value=Ternary.NO,
            confidence=0.9,
            evidence=[
                Evidence(
                    source=EvidenceSource.STRUCTURED,
                    path=best_negative.path,
                    snippet=best_negative.raw_text,
                )
            ],
        )

    return FieldMatch(value=Ternary.UNCERTAIN, confidence=0.3, evidence=[])