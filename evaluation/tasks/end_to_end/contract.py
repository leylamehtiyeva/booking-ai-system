"""
End-to-end decision contract: how one Booking AI search response maps to
the golden YES / NO / UNCERTAIN labels.

Three layers, kept separate on purpose:

1. Candidate eligibility (app/logic/result_selection.py) - a listing is
   shown only if nothing about it is CONFIRMED to violate the request
   (structured NO, explicit negative evidence, negative fallback verdict,
   property-type mismatch, price NO). Unknown is not a violation: the
   system is open-world - a facility missing from a listing's data is
   "not confirmed", never "absent".
2. Constraint satisfaction - per requested requirement: matched (confirmed
   by evidence), uncertain (no evidence either way), failed.
3. Overall decision - what the system is entitled to claim to the user,
   read from the top-ranked shown result (strong results always rank
   first, see select_ranked_items):

   - YES        top result is match_tier "strong": every requested must
                requirement (incl. property type and numeric filters) is
                confirmed and nothing is uncertain or negative;
   - UNCERTAIN  results are shown but the best one is "partial"/"weak"
                (some requirement unconfirmed), or the system asked a
                clarification question;
   - NO         nothing eligible was shown.

   A typed turn failure (provider/retrieval unavailable) is not a
   decision at all - it is scored as an error.

What the contract does not claim: "strong" covers only the requirements
the system extracted. A requirement the intent extraction missed, or kept
as nice-to-have (e.g. a subjective preference), does not stop a "strong"
result - those cases show up as critical false YES and are triaged as
extraction errors or product gaps, not hidden by the contract.

The legacy contract (any non-empty result = YES) is kept for comparison
with reports produced before this file existed.
"""

from __future__ import annotations

from typing import Any, Literal

CONTRACT_VERSION = "e2e-decision-v2"
LEGACY_CONTRACT_VERSION = "e2e-decision-v1-any-result"

Decision = Literal["YES", "NO", "UNCERTAIN"]

FAILURE_RESPONSE_TYPES = frozenset({"processing_unavailable", "routing_unavailable"})


def infer_decision(response: dict[str, Any]) -> tuple[Decision | None, str]:
    """
    (decision, basis) under CONTRACT_VERSION. decision is None only for a
    typed turn failure, which is scored as an error, not a decision.
    """
    if response.get("response_type") in FAILURE_RESPONSE_TYPES:
        return None, "turn_failure"

    if response.get("need_clarification") is True:
        return "UNCERTAIN", "clarification"

    results = response.get("results") or []
    if not results:
        return "NO", "no_results"

    top = results[0]
    if top.get("eligibility_status") not in (None, "eligible"):
        # select_ranked_items never shows ineligible results - flag it
        # instead of crediting a claim the system should not have made.
        return "UNCERTAIN", "ineligible_result_shown"

    tier = top.get("match_tier")
    if tier == "strong":
        return "YES", "strong_match"
    if tier in ("partial", "weak"):
        return "UNCERTAIN", f"{tier}_match"
    return "UNCERTAIN", "unclassified_result"


_LEGACY_NO_RESULT_MARKERS = (
    "ничего не найдено",
    "no listings remained",
    "no results",
    "nothing found",
    "no suitable listings",
)


def infer_legacy_decision(response: dict[str, Any]) -> Decision:
    """The pre-v2 rule, unchanged: any non-empty result list is YES."""
    results = response.get("results") or []

    if results:
        return "YES"

    if response.get("need_clarification") is True:
        questions = response.get("questions") or []
        debug_notes = response.get("debug_notes") or []
        text = " ".join(questions + debug_notes).lower()

        if any(marker in text for marker in _LEGACY_NO_RESULT_MARKERS):
            return "NO"

        return "UNCERTAIN"

    return "NO"


def selected_listing_ids(response: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in response.get("results") or []:
        listing_id = item.get("result_id") or item.get("id") or item.get("listing_id")
        if listing_id:
            ids.append(str(listing_id))
    return ids
