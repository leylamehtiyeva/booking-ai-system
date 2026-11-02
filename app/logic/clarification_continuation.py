"""
Deterministic PendingClarification business rules (no LLM, no text).

- build_pending_clarification: whether an abstained routing decision
  becomes a pending clarification, and which actions an answer may pick.
- resolve_pending_transition: the state transition for the next turn,
  decided from the router's typed output only.

Neither function reads the user's message: whether a reply answers,
cancels or replaces the pending clarification is a semantic judgement
made by the routing layer (PendingAwareConversationActionDecision.
pending_resolution), with PendingClarification as its input. Python only
checks that judgement against the pending state's own constraints.
"""

from __future__ import annotations

from enum import Enum

from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationDecisionStatus,
    PendingAwareConversationActionDecision,
    PendingClarification,
    PendingResolution,
)
from app.schemas.query import SearchRequest
from app.schemas.shown_result_set import ShownResultSet


class PendingTransition(str, Enum):
    # Run decision.action on the ORIGINAL request; pending cleared.
    CONTINUE = "continue"
    # Drop the original request, change nothing; pending cleared.
    CANCEL = "cancel"
    # Ask again, execute nothing; pending kept as it was.
    KEEP = "keep"
    # Route this turn on its own; the original request is never applied.
    SUPERSEDE = "supersede"


def build_pending_clarification(
    *,
    reason: ClarificationReason,
    user_message: str,
    previous_state: SearchRequest | None,
    shown_result_set: ShownResultSet | None,
) -> PendingClarification | None:
    """
    V1 policy: only AMBIGUOUS_ACTION creates pending state, and only when
    both readings can actually run - an existing search to update AND a
    non-empty shown result set to inspect. With only one reading possible
    the clarification is still asked, but nothing is carried over (the
    pre-V1 behaviour).
    """
    if reason != ClarificationReason.AMBIGUOUS_ACTION:
        return None

    if not user_message.strip():
        return None

    if previous_state is None:
        return None

    if shown_result_set is None or not shown_result_set.items:
        return None

    return PendingClarification(
        kind=ClarificationReason.AMBIGUOUS_ACTION,
        original_user_message=user_message,
        possible_actions=[
            ConversationAction.UPDATE_SEARCH,
            ConversationAction.LISTING_QUESTION,
        ],
        result_set_id=shown_result_set.result_set_id,
    )


def resolve_pending_transition(
    pending: PendingClarification,
    decision: PendingAwareConversationActionDecision,
) -> PendingTransition:
    """
    The router's pending_resolution is accepted only when it is consistent
    with the rest of the decision and the pending state; an inconsistent
    "continue" or "cancel" changes nothing - it is treated as unresolved,
    never guessed into an action or a cancellation.
    """
    resolution = decision.pending_resolution

    if resolution == PendingResolution.NEW_REQUEST:
        return PendingTransition.SUPERSEDE

    if resolution == PendingResolution.CANCEL_PENDING:
        # A cancel that also carries an executable action, or that the
        # router itself was unsure about, would silently discard what the
        # user asked for - ask again instead.
        if (
            decision.action == ConversationAction.GENERAL_CHAT
            and decision.decision_status == ConversationDecisionStatus.RESOLVED
        ):
            return PendingTransition.CANCEL
        return PendingTransition.KEEP

    if resolution == PendingResolution.UNRESOLVED:
        return PendingTransition.KEEP

    # CONTINUE_PENDING. V1 can only continue an ambiguous action; any
    # other kind is routed as an ordinary turn (the pre-V1 behaviour).
    if pending.kind != ClarificationReason.AMBIGUOUS_ACTION:
        return PendingTransition.SUPERSEDE

    if (
        decision.decision_status != ConversationDecisionStatus.RESOLVED
        or decision.action not in pending.possible_actions
    ):
        return PendingTransition.KEEP

    return PendingTransition.CONTINUE
