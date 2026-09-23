"""
Precise controlled-structured-tag detection, by path pattern only
(never by text length or the coarser source_type bucket, which lumps
genuine Facility.name tags together with .overview prose and
RoomOption.choices rate-plan strings under the same "facilities"/
"room_facilities" label).

A path counts as a controlled structured tag only if it is exactly
Facility.name at the listing level or the room level:
    listing.facilities[N].name
    rooms[N].facilities[M].name

Everything else - including listing.facilities[N].overview,
rooms[N].options[M].choices, rooms[N].name, highlights, description,
policies, fine_print - is free text and must go through the semantic
verifier.
"""

from __future__ import annotations

import re

_FACILITY_NAME_PATTERN = re.compile(r"^(listing\.facilities\[\d+\]|rooms\[\d+\]\.facilities\[\d+\])\.name$")


def is_controlled_structured_tag(path: str | None) -> bool:
    if not path:
        return False
    return bool(_FACILITY_NAME_PATTERN.match(path))
