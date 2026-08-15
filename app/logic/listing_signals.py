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


def _add_facility_signals(
    signals: list[ListingSignal],
    path: str,
    facilities: Any,
) -> None:
    for i, facility in enumerate(
        facilities or []
    ):
        # Temporary compatibility with
        # old fixtures.
        if isinstance(facility, str):
            _add_signal(
                signals,
                path,
                facility,
            )
            continue

        name = getattr(
            facility,
            "name",
            None,
        )

        if name:
            _add_signal(
                signals,
                f"{path}[{i}].name",
                name,
            )

        overview = getattr(
            facility,
            "overview",
            None,
        )

        if overview:
            _add_signal(
                signals,
                f"{path}[{i}].overview",
                overview,
            )


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


def collect_listing_signals(
    listing: ListingRaw,
) -> list[ListingSignal]:
    """
    Build normalized evidence from our canonical
    ListingRaw model.

    This layer must not depend on provider-specific
    field names such as roomType, bedTypes,
    yourChoices or finePrint.
    """

    signals: list[ListingSignal] = []

    # Listing-level text
    if listing.name:
        _add_signal(
            signals,
            "listing.name",
            listing.name,
        )

    if listing.property_type:
        _add_signal(
            signals,
            "listing.property_type",
            listing.property_type,
        )

    if listing.description:
        for sentence in split_into_sentences(
            listing.description
        ):
            _add_signal(
                signals,
                "listing.description",
                sentence,
            )

    if listing.fine_print:
        _add_signal(
            signals,
            "listing.fine_print",
            listing.fine_print,
        )

    # Listing-level facilities
    _add_facility_signals(
        signals,
        "listing.facilities",
        listing.facilities,
    )

    # Rooms
    for i, room in enumerate(
        listing.rooms or []
    ):
        if room.name:
            _add_signal(
                signals,
                f"rooms[{i}].name",
                room.name,
            )

        # Canonical bed structure
        for j, bed_group in enumerate(
            room.bed_types or []
        ):
            if bed_group.room:
                _add_signal(
                    signals,
                    (
                        f"rooms[{i}]"
                        f".bed_types[{j}].room"
                    ),
                    bed_group.room,
                )

            for bed in bed_group.beds or []:
                _add_signal(
                    signals,
                    (
                        f"rooms[{i}]"
                        f".bed_types[{j}].beds"
                    ),
                    bed,
                )

        _add_facility_signals(
            signals,
            f"rooms[{i}].facilities",
            room.facilities,
        )

        for j, option in enumerate(
            room.options or []
        ):
            if option.name:
                _add_signal(
                    signals,
                    (
                        f"rooms[{i}]"
                        f".options[{j}].name"
                    ),
                    option.name,
                )

            for choice in (
                option.choices or []
            ):
                _add_signal(
                    signals,
                    (
                        f"rooms[{i}]"
                        f".options[{j}].choices"
                    ),
                    choice,
                )

            # Structured boolean evidence.
            if option.free_cancellation is True:
                _add_signal(
                    signals,
                    (
                        f"rooms[{i}]"
                        f".options[{j}]"
                        ".free_cancellation"
                    ),
                    "free cancellation",
                )

    # Highlights
    for i, highlight in enumerate(
        listing.highlights or []
    ):
        if highlight.header:
            _add_signal(
                signals,
                f"highlights[{i}].header",
                highlight.header,
            )

        for content in (
            highlight.contents or []
        ):
            _add_signal(
                signals,
                f"highlights[{i}].contents",
                content,
            )

    # Policies
    for i, policy in enumerate(
        listing.policies or []
    ):
        if policy.title:
            _add_signal(
                signals,
                f"policies[{i}].title",
                policy.title,
            )

        if policy.content:
            _add_signal(
                signals,
                f"policies[{i}].content",
                policy.content,
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