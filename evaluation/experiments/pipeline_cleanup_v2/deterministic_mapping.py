"""
Deterministic structured-attribute -> subclaim mapping.

Applies ONLY when both hold:
  1. candidate.source_type in {"facilities", "room_facilities"}
  2. candidate.text, casefolded and stripped, exactly matches one of
     the aliases below for THIS subclaim

Anything that doesn't exactly match (including prose living in a
facilities .overview field, e.g. "WiFi is available in all areas and
is free of charge.") falls through to NLI - no fuzzy/semantic
inference from structured tags.

Minimal alias list, validated against what actually appears in the
4-property dataset - not extrapolated to hypothetical Booking tag
variants that were never observed here.
"""

from __future__ import annotations

STRUCTURED_SOURCE_TYPES = {"facilities", "room_facilities"}

DETERMINISTIC_ALIASES: dict[str, set[str]] = {
    "RW1": {"desk"},
    "RW2a": {"free wifi"},
    "FC2": {"family rooms"},
    "Q1": {"soundproofing", "soundproof rooms"},
}


def deterministic_support(subclaim: str, source_type: str, text: str) -> bool:
    if source_type not in STRUCTURED_SOURCE_TYPES:
        return False
    aliases = DETERMINISTIC_ALIASES.get(subclaim)
    if not aliases:
        return False
    return text.strip().casefold() in aliases
