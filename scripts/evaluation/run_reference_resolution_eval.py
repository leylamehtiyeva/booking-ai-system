"""
Small, standalone live-LLM check for the reference-resolution +
abstention router extension (decision_status / clarification_reason /
result_scope / target_result_id). Not part of the main router eval
harness (evaluation/tasks/conversation_router/runner.py) - that harness
scores action accuracy only; this script additionally scores scope,
reference and clarification correctness, and separately counts the two
asymmetric error types: a dangerous false UPDATE_SEARCH (a transient
message wrongly executed as a persistent search change) and an
unnecessary clarification (an unambiguous message wrongly abstained on).

Uses the same executable-outcome gating conversation_flow.py runs in
production (decision_status gate, then _resolve_target_reference for
LISTING_QUESTION) - see compute_executable_outcome() below, which must
be kept in sync with handle_user_message()'s own gate if that changes.

Makes real LLM API calls - run only when you intend to spend the
corresponding quota/cost.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.logic.conversation_flow import _resolve_target_reference
from app.logic.conversation_router import (
    ConversationRoutingError,
    route_conversation_async,
)
from app.observability.trace import RequestTrace
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationActionDecision,
    ConversationDecisionStatus,
)
from app.schemas.listing import ListingRaw
from app.schemas.shown_result_set import ShownResult, ShownResultSet
from evaluation.tasks.conversation_router.adapter import build_router_input
from evaluation.tasks.conversation_router.dataset import (
    ConversationRouterEvalCase,
    load_conversation_router_dataset,
)

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "conversation_router"
    / "reference_resolution_minimal_pairs.jsonl"
)

STABILITY_CASE_IDS = ["ref-011", "ref-013", "ref-010", "ref-006"]
STABILITY_REPEATS = 5


def _shown_result_set_for_case(
    case: ConversationRouterEvalCase,
) -> ShownResultSet | None:
    if not case.shown_results:
        return None

    return ShownResultSet(
        result_set_id=f"eval-{case.id}",
        items=[
            ShownResult(
                result_id=stub.result_id,
                listing=ListingRaw(id=stub.result_id, name=stub.title),
            )
            for stub in case.shown_results
        ],
    )


@dataclass
class ExecutableOutcome:
    executed: bool
    decision_status: str
    clarification_reason: str | None
    executed_action: str | None
    result_scope: str | None
    target_result_id: str | None


def compute_executable_outcome(
    decision: ConversationActionDecision,
    shown_result_set: ShownResultSet | None,
) -> ExecutableOutcome:
    """Mirrors conversation_flow.handle_user_message's gate exactly."""
    if decision.decision_status == ConversationDecisionStatus.NEEDS_CLARIFICATION:
        reason = (
            decision.clarification_reason
            if decision.clarification_reason != ClarificationReason.NONE
            else ClarificationReason.AMBIGUOUS_ACTION
        )
        return ExecutableOutcome(
            executed=False,
            decision_status="needs_clarification",
            clarification_reason=reason.value,
            executed_action=None,
            result_scope=None,
            target_result_id=None,
        )

    if decision.action.value == "listing_question":
        scope, target, status = _resolve_target_reference(
            decision, shown_result_set
        )

        if status == "invalid_reference":
            return ExecutableOutcome(
                executed=False,
                decision_status="needs_clarification",
                clarification_reason="ambiguous_reference",
                executed_action=None,
                result_scope=None,
                target_result_id=None,
            )

        return ExecutableOutcome(
            executed=True,
            decision_status="resolved",
            clarification_reason=None,
            executed_action="listing_question",
            result_scope=scope,
            target_result_id=target,
        )

    return ExecutableOutcome(
        executed=True,
        decision_status="resolved",
        clarification_reason=None,
        executed_action=decision.action.value,
        result_scope=None,
        target_result_id=None,
    )


async def run_case(
    case: ConversationRouterEvalCase, *, llm_profile_name: str = "gemini_default"
) -> dict:
    router_input = build_router_input(case)
    shown_result_set = _shown_result_set_for_case(case)
    trace = RequestTrace()

    try:
        decision = await route_conversation_async(
            router_input=router_input,
            trace=trace,
            llm_profile_name=llm_profile_name,
        )
    except ConversationRoutingError as exc:
        return {
            "case": case,
            "router_input": router_input,
            "error": exc.code,
            "decision": None,
            "outcome": None,
        }

    outcome = compute_executable_outcome(decision, shown_result_set)

    return {
        "case": case,
        "router_input": router_input,
        "error": None,
        "decision": decision,
        "outcome": outcome,
    }


def _expected_scope_value(case: ConversationRouterEvalCase) -> str | None:
    return case.expected_result_scope.value if case.expected_result_scope else None


def _classify_errors(outcome_row: dict) -> list[str]:
    case: ConversationRouterEvalCase = outcome_row["case"]
    errors: list[str] = []

    if outcome_row["error"] is not None:
        errors.append(f"router_execution_error:{outcome_row['error']}")
        return errors

    outcome: ExecutableOutcome = outcome_row["outcome"]
    expected_status = case.expected_decision_status.value

    if outcome.decision_status != expected_status:
        errors.append("clarification_status_mismatch")

    # Action is only meaningfully scored when both sides agree the turn
    # should execute - an ambiguous_action ground-truth row's
    # expected_action is a nominal placeholder, not a real target.
    if (
        expected_status == "resolved"
        and outcome.decision_status == "resolved"
        and outcome.executed_action != case.expected_action.value
    ):
        errors.append("action_mismatch")

    if (
        expected_status == "resolved"
        and case.expected_action.value == "listing_question"
        and outcome.decision_status == "resolved"
    ):
        if outcome.result_scope != _expected_scope_value(case):
            errors.append("scope_mismatch")
        if outcome.target_result_id != case.expected_target_result_id:
            errors.append("reference_mismatch")

    return errors


def _is_dangerous_false_update_search(outcome_row: dict) -> bool:
    case: ConversationRouterEvalCase = outcome_row["case"]
    if outcome_row["error"] is not None:
        return False
    outcome: ExecutableOutcome = outcome_row["outcome"]

    expected_update = (
        case.expected_decision_status.value == "resolved"
        and case.expected_action.value == "update_search"
    )
    actually_updated = (
        outcome.decision_status == "resolved" and outcome.executed_action == "update_search"
    )
    return actually_updated and not expected_update


def _is_unnecessary_clarification(outcome_row: dict) -> bool:
    case: ConversationRouterEvalCase = outcome_row["case"]
    if outcome_row["error"] is not None:
        return False
    outcome: ExecutableOutcome = outcome_row["outcome"]

    return (
        case.expected_decision_status.value == "resolved"
        and outcome.decision_status == "needs_clarification"
    )


def _print_case_report(outcome_row: dict, errors: list[str]) -> None:
    case: ConversationRouterEvalCase = outcome_row["case"]
    router_input = outcome_row["router_input"]

    print("=" * 78)
    print(f"[{case.id}] {case.category}")
    print(f"input: {case.user_message!r}")
    print(
        "shown context: "
        + json.dumps(router_input.latest_result_context, ensure_ascii=False)
    )
    print(
        "expected: decision_status="
        f"{case.expected_decision_status.value} "
        f"clarification_reason={case.expected_clarification_reason.value} "
        f"action={case.expected_action.value} "
        f"result_scope={_expected_scope_value(case)} "
        f"target_result_id={case.expected_target_result_id}"
    )

    if outcome_row["error"] is not None:
        print(f"actual: ROUTER EXECUTION FAILED ({outcome_row['error']})")
    else:
        decision = outcome_row["decision"]
        outcome: ExecutableOutcome = outcome_row["outcome"]
        print(
            "actual (raw LLM):   action="
            f"{decision.action.value} "
            f"decision_status={decision.decision_status.value} "
            f"clarification_reason={decision.clarification_reason.value} "
            f"result_scope={decision.result_scope.value} "
            f"target_result_id={decision.target_result_id!r}"
        )
        print(
            "actual (executed):  decision_status="
            f"{outcome.decision_status} "
            f"clarification_reason={outcome.clarification_reason} "
            f"executed_action={outcome.executed_action} "
            f"result_scope={outcome.result_scope} "
            f"target_result_id={outcome.target_result_id}"
        )

    dangerous = _is_dangerous_false_update_search(outcome_row)
    unnecessary = _is_unnecessary_clarification(outcome_row)
    flags = []
    if dangerous:
        flags.append("DANGEROUS_FALSE_UPDATE_SEARCH")
    if unnecessary:
        flags.append("UNNECESSARY_CLARIFICATION")

    print(f"errors: {errors or 'none'}" + (f"   flags: {flags}" if flags else ""))


async def run_dataset(llm_profile_name: str) -> list[tuple[dict, list[str]]]:
    cases = load_conversation_router_dataset(DATASET_PATH)

    rows = []
    for case in cases:
        outcome_row = await run_case(case, llm_profile_name=llm_profile_name)
        errors = _classify_errors(outcome_row)
        rows.append((outcome_row, errors))
        _print_case_report(outcome_row, errors)

    return rows


def print_summary(rows: list[tuple[dict, list[str]]]) -> None:
    total = len(rows)

    def not_execution_error(errors: list[str]) -> bool:
        return not any(e.startswith("router_execution_error") for e in errors)

    action_scoped = [
        (o, e)
        for o, e in rows
        if o["case"].expected_decision_status.value == "resolved"
        and not_execution_error(e)
    ]
    action_correct = sum(
        1 for o, e in action_scoped if "action_mismatch" not in e
    )

    clarification_correct = sum(
        1
        for o, e in rows
        if "clarification_status_mismatch" not in e and not_execution_error(e)
    )

    scope_cases = [
        (o, e)
        for o, e in rows
        if o["case"].expected_action.value == "listing_question"
        and o["case"].expected_decision_status.value == "resolved"
        and not_execution_error(e)
    ]
    scope_correct = sum(1 for o, e in scope_cases if "scope_mismatch" not in e)
    reference_correct = sum(
        1 for o, e in scope_cases if "reference_mismatch" not in e
    )

    dangerous_count = sum(1 for o, _ in rows if _is_dangerous_false_update_search(o))
    unnecessary_count = sum(1 for o, _ in rows if _is_unnecessary_clarification(o))

    print("=" * 78)
    print("SUMMARY")
    print(
        f"action correctness:        {action_correct}/{len(action_scoped)} "
        f"({action_correct / len(action_scoped):.0%})  [resolved-expected cases]"
        if action_scoped
        else "action correctness:        n/a"
    )
    if scope_cases:
        print(
            f"scope correctness:         {scope_correct}/{len(scope_cases)} "
            f"({scope_correct / len(scope_cases):.0%})"
        )
        print(
            f"reference correctness:     {reference_correct}/{len(scope_cases)} "
            f"({reference_correct / len(scope_cases):.0%})"
        )
    print(
        f"clarification correctness: {clarification_correct}/{total} "
        f"({clarification_correct / total:.0%})"
    )
    print()
    print(f"dangerous false UPDATE_SEARCH: {dangerous_count}")
    print(f"unnecessary clarification:     {unnecessary_count}")

    failures = [(o, e) for o, e in rows if e]
    print(f"\nfailures: {len(failures)}/{total}")
    for o, e in failures:
        print(f"  - {o['case'].id}: {e}")


async def run_stability_check(llm_profile_name: str) -> None:
    cases = {c.id: c for c in load_conversation_router_dataset(DATASET_PATH)}

    print("\n" + "=" * 78)
    print(f"STABILITY CHECK ({STABILITY_REPEATS} repeats per case)")

    for case_id in STABILITY_CASE_IDS:
        case = cases[case_id]
        print("-" * 78)
        print(f"[{case_id}] {case.user_message!r}")

        for i in range(1, STABILITY_REPEATS + 1):
            outcome_row = await run_case(case, llm_profile_name=llm_profile_name)
            if outcome_row["error"] is not None:
                print(f"  run {i}: ROUTER EXECUTION FAILED ({outcome_row['error']})")
                continue
            outcome: ExecutableOutcome = outcome_row["outcome"]
            print(
                f"  run {i}: decision_status={outcome.decision_status} "
                f"executed_action={outcome.executed_action} "
                f"clarification_reason={outcome.clarification_reason}"
            )


async def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "gemini_default"
    rows = await run_dataset(profile)
    print_summary(rows)

    if "--stability" in sys.argv:
        await run_stability_check(profile)


if __name__ == "__main__":
    asyncio.run(main())
