from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from app.schemas.listing import ListingRaw
from typing import Sequence
import re


def split_into_sentences(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def normalize_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


@dataclass(frozen=True)
class ListingSignal:
    """
    Normalized piece of evidence extracted from raw provider JSON.

    path:
        where it came from in the raw listing structure
    text:
        normalized searchable text
    raw_text:
        original string before normalization
    """
    path: str
    text: str
    raw_text: str


def _stringify_maybe_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
            elif isinstance(item, dict):
                # common patterns
                for key in ("name", "title", "content", "text", "value"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v)

                contents = item.get("contents")
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, str) and c.strip():
                            out.append(c)
        return out

    if isinstance(value, dict):
        out: List[str] = []
        for key in ("name", "title", "content", "text", "value"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v)

        contents = value.get("contents")
        if isinstance(contents, list):
            for c in contents:
                if isinstance(c, str) and c.strip():
                    out.append(c)

        return out


def _add_signal(signals: List[ListingSignal], path: str, raw_text: str) -> None:
    raw_text = str(raw_text).strip()
    if not raw_text:
        return

    signals.append(
        ListingSignal(
            path=path,
            text=normalize_text(raw_text),
            raw_text=raw_text,
        )
    )


def _add_many(signals: List[ListingSignal], path: str, values: Iterable[str]) -> None:
    for v in values:
        _add_signal(signals, path, v)


def collect_listing_signals(listing: ListingRaw) -> List[ListingSignal]:
    """
    Build a normalized evidence layer over raw provider JSON.

    Important:
    - tolerant to missing fields
    - tolerant to extra fields
    - does NOT make match decisions
    """
    signals: List[ListingSignal] = []

    # listing-level plain text
    if getattr(listing, "name", None):
        _add_signal(signals, "listing.name", listing.name)

    if getattr(listing, "property_type", None):
        _add_signal(signals, "listing.property_type", listing.property_type)

    if getattr(listing, "description", None):
        for sent in split_into_sentences(listing.description):
            _add_signal(signals, "listing.description", sent)

    # listing-level facilities
    _add_many(
        signals,
        "listing.facilities",
        _stringify_maybe_list(getattr(listing, "facilities", [])),
    )

    # rooms
    for i, room in enumerate(getattr(listing, "rooms", []) or []):
        if getattr(room, "name", None):
            _add_signal(signals, f"rooms[{i}].name", room.name)

        room_type = getattr(room, "roomType", None)
        if room_type:
            _add_signal(signals, f"rooms[{i}].roomType", room_type)

        bed_types = getattr(room, "bedTypes", None)
        _add_many(signals, f"rooms[{i}].bedTypes", _stringify_maybe_list(bed_types))

        _add_many(
            signals,
            f"rooms[{i}].facilities",
            _stringify_maybe_list(getattr(room, "facilities", [])),
        )

        for j, opt in enumerate(getattr(room, "options", []) or []):
            if getattr(opt, "name", None):
                _add_signal(signals, f"rooms[{i}].options[{j}].name", opt.name)

            your_choices = getattr(opt, "yourChoices", None)
            _add_many(
                signals,
                f"rooms[{i}].options[{j}].yourChoices",
                _stringify_maybe_list(your_choices),
            )

    # common listing extras
    highlights = getattr(listing, "highlights", None)
    if isinstance(highlights, list):
        for i, hl in enumerate(highlights):
            if isinstance(hl, dict):
                _add_many(
                    signals,
                    f"highlights[{i}]",
                    _stringify_maybe_list(hl),
                )

    policies = getattr(listing, "policies", None)
    if isinstance(policies, list):
        for i, pol in enumerate(policies):
            if isinstance(pol, dict):
                _add_many(
                    signals,
                    f"policies[{i}]",
                    _stringify_maybe_list(pol),
                )

    return signals



def signal_contains_alias(signal_text: str, alias: str) -> bool:
    return alias in signal_text


def find_best_signal_match(
    signals: Sequence[ListingSignal],
    aliases: Sequence[str],
    preferred_path_prefixes: Sequence[str] = (),
) -> ListingSignal | None:
    """
    Return best matching signal for given aliases.

    Ranking:
    1. earlier preferred path prefix wins
    2. longer alias match wins
    3. earlier signal order wins
    """
    candidates: list[tuple[int, int, int, ListingSignal]] = []

    for idx, s in enumerate(signals):
        for alias in aliases:
            if signal_contains_alias(s.text, alias):
                path_rank = len(preferred_path_prefixes)
                for i, prefix in enumerate(preferred_path_prefixes):
                    if s.path.startswith(prefix):
                        path_rank = i
                        break

                candidates.append((path_rank, -len(alias), idx, s))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]

def find_best_negative_signal_match(
    signals: Sequence[ListingSignal],
    negative_aliases: Sequence[str],
    preferred_path_prefixes: Sequence[str] = (),
) -> ListingSignal | None:
    return find_best_signal_match(
        signals=signals,
        aliases=negative_aliases,
        preferred_path_prefixes=preferred_path_prefixes,
    )