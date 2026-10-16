"""
Which part of a listing an area measurement describes.

"Rooms: 25 sqm, rooftop terrace: 200 sqm" has two areas, and only the first
is the size of what the user books. The scope of each mention is decided by
the word it is attached to, in this order:

1. a scope word right after it, with no punctuation between ("35 sqm room",
   "10 sqm balcony");
2. the nearest scope word before it in the same sentence ("Each room is
   35 sqm", "hotel ... and 120 square meters");
3. the listing's subject - the first scope word of the text ("Apartment in
   Baku with a kitchen. Size is 45 sqm.").

"unit" is the bookable room or apartment, "other" is any other part of the
property (building, terrace, plot, ...), "unknown" is everything else.
"""

from __future__ import annotations

import re
from typing import Literal

AreaScope = Literal["unit", "other", "unknown"]

_OTHER_WORDS = (
    r"living rooms?|meeting rooms?|conference rooms?|room service|terraces?|rooftops?|balcon(?:y|ies)|patios?"
    r"|decks?|gardens?|yards?|courtyards?|grounds|plots?|land|buildings?|hotels?|resorts?|complex|estate"
    r"|pools?|lobby|lobbies|restaurants?|spa|gym|lounges?|parking|halls?"
)
_UNIT_WORDS = (
    r"rooms?|suites?|apartments?|studios?|flats?|units?|villas?|houses?|cottages?|chalets?|lofts?"
    r"|condos?|bungalows?|guestrooms?|accommodation"
)
_SCOPE_WORD = re.compile(rf"\b(?:(?P<other>{_OTHER_WORDS})|(?P<unit>{_UNIT_WORDS}))\b", re.I)
_SENTENCE_END = re.compile(r"[.!?;\n](?!\d)")
_WORDS_ONLY = re.compile(r"^\s+(?:[a-z-]+\s+){0,2}$", re.I)

_NUMBER = r"\d+(?:[\.,]\d+)?"
_AREA_UNITS = [
    (r"sqm|sq\.?\s*m|m²|m2|square meters?|square metres?", 1.0),
    (r"sq\.?\s*ft|ft²|ft2|square feet|feet²", 0.092903),
]


def _scope_of(match: re.Match) -> AreaScope:
    return "other" if match.group("other") else "unit"


def scope_at(text: str, start: int, end: int) -> AreaScope:
    """Scope of the area mention at text[start:end]."""
    following = _SCOPE_WORD.search(text, end)
    if following and _WORDS_ONLY.match(text[end : following.start()] or " "):
        return _scope_of(following)

    sentence_start = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    preceding = list(_SCOPE_WORD.finditer(text, sentence_start, start))
    if preceding:
        return _scope_of(preceding[-1])

    subject = _SCOPE_WORD.search(text)
    if subject and subject.start() < start:
        return _scope_of(subject)
    return "unknown"


def area_mentions(text: str) -> list[tuple[float, AreaScope]]:
    """Every area in `text`, in sqm, with its scope. A range gives both ends."""
    out: list[tuple[float, AreaScope]] = []
    taken: list[tuple[int, int]] = []
    for units, multiplier in _AREA_UNITS:
        unit = rf"\s*(?:{units})\b"
        ranges = re.compile(rf"\b({_NUMBER})\s*(?:-|–|to)\s*({_NUMBER}){unit}", re.I)
        singles = re.compile(rf"\b({_NUMBER}){unit}", re.I)
        for pattern in (ranges, singles):
            for match in pattern.finditer(text):
                if any(match.start() < e and s < match.end() for s, e in taken):
                    continue
                taken.append(match.span())
                scope = scope_at(text, match.start(), match.end())
                for raw in match.groups():
                    try:
                        out.append((round(float(raw.replace(",", ".")) * multiplier, 2), scope))
                    except ValueError:
                        continue
    return out


def mentions_other_part(text: str) -> bool:
    """True when `text` names a non-unit part of a property (e.g. "large terrace")."""
    return any(match.group("other") for match in _SCOPE_WORD.finditer(text or ""))
