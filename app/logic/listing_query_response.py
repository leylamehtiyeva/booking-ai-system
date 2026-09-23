from __future__ import annotations

from app.schemas.conversation_response import (
    ListingQueryConversationOutcome,
    ListingQueryResponseItem,
)
from app.schemas.match import Ternary
from app.schemas.result_query_match import ConstraintMatchResult

_ORDINALS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)

_SHORT_VERDICT = {
    Ternary.YES: "yes",
    Ternary.NO: "no",
    Ternary.UNCERTAIN: "not confirmed",
}


def _ordinal(position: int) -> str:
    if 1 <= position <= len(_ORDINALS):
        return _ORDINALS[position - 1]
    return f"#{position}"


def _reference(item: ListingQueryResponseItem) -> str:
    if item.title:
        return item.title
    return f"the {_ordinal(item.position)} hotel"


def _label(constraint: ConstraintMatchResult) -> str:
    """
    Field.PARKING -> "parking", Field.PRIVATE_BATHROOM -> "private
    bathroom" - a trivial local substitution, not a new taxonomy. Falls
    back to the constraint's own raw_text when there is no canonical
    Field (unresolved constraints), which is the existing text already
    carried on ConstraintMatchResult - nothing new is invented.
    """
    if constraint.field is not None:
        return constraint.field.value.replace("_", " ")
    return constraint.raw_text


def _verdict_sentence(ref: str, label: str, value: Ternary) -> str:
    if value == Ternary.YES:
        return f"{ref} has {label}."
    if value == Ternary.NO:
        return f"{ref} does not have {label}."
    return f"I couldn't verify {label} for {ref} from the listing information."


def render_listing_query_response(outcome: ListingQueryConversationOutcome) -> str:
    """
    Pure presentation over an already-final ListingQueryConversationOutcome.

    Takes nothing except the typed outcome - no ListingRaw, ShownResultSet,
    SearchRequest, router decision, candidate selector, evidence resolver,
    trace, or matcher. Never re-derives or adjusts a verdict: every
    sentence is a direct, mechanical restatement of
    ConstraintMatchResult.value. Multiple constraints per candidate are
    rendered as independent bullet lines - never collapsed into an
    invented overall pass/fail, since the matcher never computed one.
    """
    if not outcome.items:
        return ""

    multi_constraint = len(outcome.items[0].match.constraint_results) > 1

    blocks: list[str] = []
    for item in outcome.items:
        ref = _reference(item)
        constraint_results = item.match.constraint_results

        if not multi_constraint:
            label = _label(constraint_results[0])
            blocks.append(_verdict_sentence(ref, label, constraint_results[0].value))
            continue

        lines = [f"{ref}:"]
        for constraint in constraint_results:
            lines.append(f"- {_label(constraint)}: {_SHORT_VERDICT[constraint.value]}")
        blocks.append("\n".join(lines))

    separator = "\n\n" if multi_constraint else "\n"
    return separator.join(blocks)
