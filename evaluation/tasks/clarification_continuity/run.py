"""
Live acceptance evaluation - Clarification Continuity V1.

Runs fixed multi-turn cases through process_turn (the FastAPI/Streamlit
entry point) against a deterministic session fixture: one SearchRequest
(Baku, WiFi must), one ShownResultSet of three hotels and, for most
cases, a PendingClarification for "Only the ones with parking?".

Real LLM calls (--live): conversation router (pending-aware), Intent
Update, initial intent extraction (only if the router picks
start_search) and factual question extraction - the semantic stages
under test. Always faked: hotel retrieval (orchestrate_search_request),
factual matching (no textual fallback / semantic verifier calls) and
response wording (deterministic renderer).

Modes:
    --list      print cases and the nominal call budget, no calls
    --dry-run   run the harness with scripted stand-ins, no API calls
    --live      real Gemini calls (paid) - requires explicit authorization

Scoring (evaluation only, not application logic): expected values are
fixed per case; extracted facts are mapped to concepts by canonical
field or normalized_text, and anything else is reported as "other:*".
Each failed turn is attributed to the first failing layer:
infrastructure -> routing (model) -> application (Python) ->
extraction (Intent Update / factual extractor model).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import ExitStack
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "clarification_continuity" / "cases_v1.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "outputs" / "clarification_continuity"

REFERENCE_DATE = date(2026, 11, 1)
ORIGINAL = "Only the ones with parking?"
RESULT_SET_ID = "rs-acceptance-v1"
HOTELS = [
    ("lp-acc-hilton", "Hilton Baku"),
    ("lp-acc-marriott", "Marriott Absheron Baku"),
    ("lp-acc-fairmont", "Fairmont Baku Flame Towers"),
]

SEMANTIC_STAGES = {"intent_update", "search_intent_extraction", "factual_question_extraction"}


def load_cases(selected: set[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if selected is None or c["id"] in selected]


def nominal_calls(case: dict[str, Any]) -> dict[str, int]:
    """Calls one run of the case makes if every stage succeeds first time."""
    calls = {"router": 0, "intent_update": 0, "factual_extraction": 0}
    for turn in case["turns"]:
        expect = turn["expect"]
        calls["router"] += 1
        actions = expect.get("action", [])
        if "update_search" in actions or "start_search" in actions:
            continued = expect.get("pending_resolution") == ["continue_pending"]
            calls["intent_update"] += 2 if continued else 1
        if "facts" in expect:
            calls["factual_extraction"] += 1
    return calls


# --- fixture -------------------------------------------------------------------------


def _fixture():
    from app.schemas.constraints import (
        ConstraintCategory,
        ConstraintMappingStatus,
        ConstraintPriority,
        EvidenceStrategy,
        UserConstraint,
    )
    from app.schemas.conversation_route import (
        ClarificationReason,
        ConversationAction,
        PendingClarification,
    )
    from app.schemas.fields import Field
    from app.schemas.filters import PriceConstraint, SearchFilters
    from app.schemas.listing import ListingRaw
    from app.schemas.property_semantics import PropertyType
    from app.schemas.query import SearchRequest
    from app.schemas.shown_result_set import ShownResult, ShownResultSet

    search = SearchRequest(
        city="Baku",
        check_in=date(2026, 12, 10),
        check_out=date(2026, 12, 14),
        adults=2,
        property_types=[PropertyType.HOTEL],
        filters=SearchFilters(price=PriceConstraint(max_amount=200, currency="USD", scope="per_night")),
        constraints=[
            UserConstraint(
                id="c-wifi",
                raw_text="WiFi",
                normalized_text="wifi",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.AMENITY,
                mapping_status=ConstraintMappingStatus.KNOWN,
                mapped_fields=[Field.WIFI],
                evidence_strategy=EvidenceStrategy.STRUCTURED,
            )
        ],
    )
    shown = ShownResultSet(
        result_set_id=RESULT_SET_ID,
        items=[ShownResult(result_id=rid, listing=ListingRaw(id=rid, name=name, city="Baku")) for rid, name in HOTELS],
    )
    pending = PendingClarification(
        kind=ClarificationReason.AMBIGUOUS_ACTION,
        original_user_message=ORIGINAL,
        possible_actions=[ConversationAction.UPDATE_SEARCH, ConversationAction.LISTING_QUESTION],
        result_set_id=RESULT_SET_ID,
    )
    return search, shown, pending


# --- recording and fakes ----------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.router: list[dict[str, Any]] = []
        self.intent_update: list[dict[str, Any]] = []
        self.intent_extraction: list[dict[str, Any]] = []
        self.factual: list[dict[str, Any]] = []
        self.searched: list[Any] = []
        self.matched: list[dict[str, Any]] = []

    def reset(self) -> None:
        # Cleared in place: the installed wrappers hold these list objects.
        for calls in (self.router, self.intent_update, self.intent_extraction, self.factual, self.searched, self.matched):
            calls.clear()


def _wrap(recorder_list: list, real, describe):
    async def wrapped(*args, **kwargs):
        entry = describe(*args, **kwargs)
        try:
            result = await real(*args, **kwargs)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {getattr(exc, 'code', exc)}"
            recorder_list.append(entry)
            raise
        entry["output"] = result.model_dump(mode="json") if hasattr(result, "model_dump") else repr(result)
        recorder_list.append(entry)
        return result

    return wrapped


def _install_patches(stack: ExitStack, recorder: Recorder, *, dry_run: bool, script: dict[str, Any]) -> None:
    from app.logic import conversation_flow, intent_router, intent_update
    from app.logic.conversation_response_generator import generate_deterministic_conversation_response
    from app.logic.conversation_response_llm import ConversationResponseGenerationResult
    from app.logic.result_query_matching import ResultQueryMatchOutcome
    from app.schemas.match import Ternary
    from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
    from app.schemas.search_response import NormalizedSearchResponse, SearchStatus
    from app.schemas.shown_result_set import ShownResultSet
    from app.services import conversation_service

    async def fake_orchestrate(state, **kwargs):
        recorder.searched.append(state)
        return (
            NormalizedSearchResponse(status=SearchStatus.NO_RESULTS, results=[]),
            ShownResultSet(result_set_id="rs-after-search", items=[]),
        )

    async def fake_matcher(*, result_query, candidate_selection, **kwargs):
        recorder.matched.append(
            {"candidates": [c.result_id for c in candidate_selection.candidates]}
        )
        return ResultQueryMatchOutcome(
            results=[
                ResultQueryMatch(
                    result_id=c.result_id,
                    constraint_results=[
                        ConstraintMatchResult(
                            constraint_id=q.id,
                            raw_text=q.raw_text,
                            field=q.mapped_fields[0] if q.mapped_fields else None,
                            value=Ternary.UNCERTAIN,
                        )
                        for q in result_query.constraints
                    ],
                )
                for c in candidate_selection.candidates
            ],
            status="ok",
        )

    async def deterministic_response(response_input, *, trace=None):
        return ConversationResponseGenerationResult(
            text=generate_deterministic_conversation_response(response_input),
            source="deterministic",
        )

    stack.enter_context(mock.patch.object(conversation_flow, "orchestrate_search_request", fake_orchestrate))
    stack.enter_context(mock.patch.object(conversation_flow, "match_result_query_against_candidates", fake_matcher))
    stack.enter_context(
        mock.patch.object(conversation_service, "generate_conversation_response_with_llm", deterministic_response)
    )

    if dry_run:
        route_real = _dry_route(script)
        update_real = _dry_update(script)
        extraction_real = _dry_unused("route_intent_adk_async")
        factual_real = _dry_factual(script)
    else:
        route_real = conversation_flow.route_conversation_async
        update_real = intent_update.route_intent_update_patch_async
        extraction_real = intent_router.route_intent_adk_async
        factual_real = conversation_flow.extract_factual_constraints_async

    stack.enter_context(
        mock.patch.object(
            conversation_flow,
            "route_conversation_async",
            _wrap(
                recorder.router,
                route_real,
                lambda *, router_input, **kw: {
                    "message": router_input.user_message,
                    "pending_in_input": router_input.pending_clarification is not None,
                },
            ),
        )
    )
    stack.enter_context(
        mock.patch.object(
            intent_update,
            "route_intent_update_patch_async",
            _wrap(
                recorder.intent_update,
                update_real,
                lambda previous_state, user_message, **kw: {
                    "message": user_message,
                    "confirmed_search_change": kw.get("confirmed_search_change", False),
                    "answered_request": kw.get("answered_request"),
                },
            ),
        )
    )
    stack.enter_context(
        mock.patch.object(
            intent_router,
            "route_intent_adk_async",
            _wrap(recorder.intent_extraction, extraction_real, lambda user_text, **kw: {"message": user_text}),
        )
    )
    stack.enter_context(
        mock.patch.object(
            conversation_flow,
            "extract_factual_constraints_async",
            _wrap(
                recorder.factual,
                factual_real,
                lambda user_message, **kw: {
                    "message": user_message,
                    "original_question": kw.get("original_question"),
                },
            ),
        )
    )


# Scripted stand-ins for --dry-run: they return the first acceptable
# expected outcome, so a dry run exercises the harness and the
# application path, never a model.


def _dry_route(script):
    from app.schemas.conversation_route import (
        ClarificationReason,
        ConversationAction,
        ConversationActionDecision,
        ConversationDecisionStatus,
        PendingAwareConversationActionDecision,
        ResultScope,
    )

    async def route(*, router_input, **kwargs):
        expect = script["expect"]
        if router_input.pending_clarification is None:
            return ConversationActionDecision(
                action=ConversationAction.LISTING_QUESTION,
                reason="dry-run",
                decision_status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
                clarification_reason=ClarificationReason.AMBIGUOUS_ACTION,
                result_scope=ResultScope.CURRENT_RESULTS,
                target_result_id="",
            )
        resolution = expect["pending_resolution"][0]
        unresolved = resolution == "unresolved"
        action = (expect.get("action") or ["listing_question"])[0]
        scope = (expect.get("result_scope") or ["specific_result" if action == "listing_question" else "not_applicable"])[0]
        position = expect.get("target_position")
        return PendingAwareConversationActionDecision(
            action=ConversationAction(action if not unresolved else "update_search"),
            reason="dry-run",
            decision_status=ConversationDecisionStatus.NEEDS_CLARIFICATION if unresolved else ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.AMBIGUOUS_ACTION if unresolved else ClarificationReason.NONE,
            result_scope=ResultScope(scope),
            target_result_id=HOTELS[position - 1][0] if position else "",
            pending_resolution=resolution,
        )

    return route


def _parking(priority):
    from app.schemas.constraints import (
        ConstraintCategory,
        ConstraintMappingStatus,
        EvidenceStrategy,
        UserConstraint,
    )
    from app.schemas.fields import Field

    return UserConstraint(
        raw_text="parking",
        normalized_text="parking",
        priority=priority,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.PARKING],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _dry_update(script):
    from app.schemas.constraints import ConstraintPriority
    from app.schemas.intent_patch import SearchIntentPatch

    async def update(previous_state, user_message, *, trace=None, confirmed_search_change=False, answered_request=None):
        search = script["expect"].get("search", {})
        if confirmed_search_change:
            return SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.MUST)])
        if answered_request is not None:
            if search.get("parking") == "nice":
                return SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE)])
            return SearchIntentPatch()
        if search.get("city") and search["city"] != "Baku":
            return SearchIntentPatch(set_city=search["city"])
        return SearchIntentPatch()

    return update


def _dry_factual(script):
    from app.schemas.constraints import (
        ConstraintCategory,
        ConstraintMappingStatus,
        ConstraintPriority,
        EvidenceStrategy,
        UserConstraint,
    )
    from app.schemas.result_query import ResultQuery

    async def factual(user_message, *, trace=None, original_question=None):
        constraints = []
        for fact in script["expect"].get("facts", []):
            if fact == "parking":
                constraints.append(_parking(ConstraintPriority.MUST))
            else:
                constraints.append(
                    UserConstraint(
                        raw_text=fact,
                        normalized_text=fact,
                        priority=ConstraintPriority.MUST,
                        category=ConstraintCategory.AMENITY,
                        mapping_status=ConstraintMappingStatus.UNRESOLVED,
                        evidence_strategy=EvidenceStrategy.TEXTUAL,
                    )
                )
        return ResultQuery(constraints=constraints)

    return factual


def _dry_unused(name):
    async def unused(*args, **kwargs):
        raise AssertionError(f"dry-run: {name} is not scripted")

    return unused


# --- scoring ---------------------------------------------------------------------------


def _concept(constraint: dict[str, Any]) -> str:
    fields = constraint.get("mapped_fields") or []
    text = (constraint.get("normalized_text") or "").lower()
    if "parking" in fields or "parking" in text:
        return "parking"
    if "wifi" in fields or text in {"wifi", "wi-fi"}:
        return "wifi"
    if "pool" in text:
        return "pool"
    return f"other:{text}"


def _priorities(search: dict[str, Any] | None, concept: str) -> list[str]:
    return [c["priority"] for c in (search or {}).get("constraints", []) if _concept(c) == concept]


def _unrelated_changes(fixture: dict[str, Any], final: dict[str, Any] | None) -> list[str]:
    if final is None:
        return ["search_request_cleared"]
    changed = []
    for key in ("city", "check_in", "check_out", "adults", "children", "rooms", "property_types", "filters", "budget_max", "currency"):
        if fixture.get(key) != final.get(key):
            changed.append(key)

    def others(search):
        return sorted(
            (_concept(c), c["priority"]) for c in search.get("constraints", []) if _concept(c) != "parking"
        )

    if others(fixture) != others(final):
        changed.append("other_constraints")
    return changed


def score_turn(*, case, index, expect, before, turn, recorder, fixture_search_json) -> dict[str, Any]:
    router = (turn.debug.parsed_intent or {}).get("router") or {}
    final_search = turn.session.search_request.model_dump(mode="json") if turn.session.search_request else None
    issues: dict[str, list[str]] = {"infrastructure": [], "routing": [], "application": [], "extraction": []}

    if turn.failure is not None:
        issues["infrastructure"].append(f"{turn.failure.stage}:{turn.failure.code}")

    # routing (model)
    for field in ("pending_resolution", "action", "decision_status", "clarification_reason", "result_scope"):
        if field in expect and router.get(field) not in expect[field]:
            issues["routing"].append(f"{field}={router.get(field)!r}, expected one of {expect[field]}")
    if expect.get("target_unresolved") and router.get("target_result_id") in {rid for rid, _ in HOTELS}:
        issues["routing"].append(
            f"target_result_id={router.get('target_result_id')!r} picks a hotel the user did not identify"
        )
    if "target_position" in expect:
        expected_target = HOTELS[expect["target_position"] - 1][0]
        if router.get("target_result_id") != expected_target:
            issues["routing"].append(f"target_result_id={router.get('target_result_id')!r}, expected {expected_target!r}")

    # application (Python state transitions given the decision)
    after = turn.session.pending_clarification
    pending_after = expect.get("pending_after")
    if pending_after == "cleared" and after is not None:
        issues["application"].append("pending not cleared")
    if pending_after == "kept" and after != before.pending_clarification:
        issues["application"].append("pending not kept unchanged")
    if pending_after == "created" and (after is None or after.original_user_message != turn_message(case, index)):
        issues["application"].append("pending not created from this message")
    semantic_calls = len(recorder.intent_update) + len(recorder.intent_extraction) + len(recorder.factual)
    if expect.get("downstream") == "none" and (semantic_calls or recorder.searched or recorder.matched):
        issues["application"].append("downstream stage ran on a non-executing turn")
    if expect.get("asks_reference") and not all(name in turn.assistant_message for _, name in HOTELS):
        issues["application"].append("no reference question naming the shown hotels")
    if "candidates" in expect:
        got = recorder.matched[0]["candidates"] if recorder.matched else []
        want = [rid for rid, _ in HOTELS] if expect["candidates"] == "all" else [HOTELS[p - 1][0] for p in expect["candidates"]]
        if got != want:
            issues["application"].append(f"candidates={got}, expected {want}")

    # extraction (Intent Update / factual extractor)
    search_expect = expect.get("search") or {}
    if search_expect.get("unchanged") and final_search != fixture_search_json:
        issues["extraction"].append("search changed on a read-only turn")
    for concept in ("parking", "wifi"):
        want = search_expect.get(concept)
        if want is None:
            continue
        got = _priorities(final_search, concept)
        if (want == "absent" and got) or (want != "absent" and got != [want]):
            issues["extraction"].append(f"{concept} priorities={got}, expected {want}")
    if "city" in search_expect and (final_search or {}).get("city", "").lower() != search_expect["city"].lower():
        issues["extraction"].append(f"city={(final_search or {}).get('city')!r}, expected {search_expect['city']!r}")

    unrelated = _unrelated_changes(fixture_search_json, final_search)
    allowed = set(search_expect.get("allow_changes", []))
    if router.get("action") == "start_search":
        allowed |= {"check_in", "check_out", "adults", "property_types", "filters", "budget_max", "other_constraints", "currency"}
    unexpected = [c for c in unrelated if c not in allowed]
    if not search_expect.get("unchanged") and "search" in expect and unexpected:
        issues["extraction"].append(f"unrelated state changed: {unexpected}")

    facts = None
    if recorder.factual and "output" in recorder.factual[-1]:
        facts = [_concept(c) for c in recorder.factual[-1]["output"].get("constraints", [])]
    if "facts" in expect and sorted(facts or []) != sorted(expect["facts"]):
        issues["extraction"].append(f"facts={facts}, expected {expect['facts']}")

    failing = next((layer for layer, found in issues.items() if found), None)
    calls = turn.debug.telemetry["llm"]["calls"] if turn.debug and turn.debug.telemetry else []
    return {
        "case_id": case["id"],
        "group": case["group"],
        "lang": case["lang"],
        "turn": index + 1,
        "message": turn_message(case, index),
        "router_calls": list(recorder.router),
        "pending_in": before.pending_clarification.model_dump(mode="json") if before.pending_clarification else None,
        "router_decision": router,
        "expected": expect,
        "pending_out": after.model_dump(mode="json") if after else None,
        "final_search_request": final_search,
        "unrelated_changes": unrelated,
        "intent_update_calls": list(recorder.intent_update),
        "intent_extraction_calls": list(recorder.intent_extraction),
        "factual_calls": list(recorder.factual),
        "extracted_facts": facts,
        "searches_run": len(recorder.searched),
        "matched_candidates": list(recorder.matched),
        "assistant_message": turn.assistant_message,
        "llm_calls": [{"step": c.get("step"), "model": c.get("model"), "success": c.get("success"), "error": c.get("error")} for c in calls],
        "llm_call_count": len(calls),
        "latency_ms": turn.telemetry.latency_ms,
        "estimated_cost_usd": turn.telemetry.estimated_cost_usd,
        "passed": failing is None,
        "failure_layer": failing,
        "issues": {k: v for k, v in issues.items() if v},
    }


def turn_message(case, index) -> str:
    return case["turns"][index]["message"]


# --- run -------------------------------------------------------------------------------


async def run(cases, *, repeats: int, dry_run: bool, max_calls: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from app.config import clock
    from app.schemas.session import ConversationSession
    from app.schemas.turn import TurnConfig
    from app.services.conversation_service import process_turn

    config = TurnConfig(persist_telemetry=False, include_debug=True)
    fixture_search, fixture_shown, fixture_pending = _fixture()
    fixture_search_json = fixture_search.model_dump(mode="json")
    recorder = Recorder()
    script: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    total_calls = 0
    completed_repeats = 0
    stopped = None

    with ExitStack() as stack:
        _install_patches(stack, recorder, dry_run=dry_run, script=script)
        for repeat in range(1, repeats + 1):
            for case in cases:
                if stopped:
                    break
                session = ConversationSession(
                    search_request=fixture_search,
                    shown_result_set=fixture_shown,
                    pending_clarification=fixture_pending if case["pending"] else None,
                )
                for index, turn_spec in enumerate(case["turns"]):
                    if total_calls >= max_calls:
                        stopped = f"--max-calls {max_calls} reached in repeat {repeat} before {case['id']} t{index + 1}"
                        print(f"STOPPED: {stopped}")
                        break
                    recorder.reset()
                    script["expect"] = turn_spec["expect"]
                    started = time.perf_counter()
                    with clock.use_reference_date(REFERENCE_DATE):
                        turn = await process_turn(session, turn_spec["message"], config=config)
                    record = score_turn(
                        case=case,
                        index=index,
                        expect=turn_spec["expect"],
                        before=session,
                        turn=turn,
                        recorder=recorder,
                        fixture_search_json=fixture_search_json,
                    )
                    record["repeat"] = repeat
                    record["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
                    records.append(record)
                    total_calls += record["llm_call_count"]
                    status = "PASS" if record["passed"] else f"FAIL[{record['failure_layer']}]"
                    print(f"r{repeat} {case['id']} t{index + 1} {status} {record['issues'] or ''}")
                    session = turn.session
            if stopped:
                break
            completed_repeats = repeat
    return records, {"requested_repeats": repeats, "completed_repeats": completed_repeats, "stopped": stopped}


def summarize(records: list[dict[str, Any]], *, mode: str, models: dict[str, str], completion: dict[str, Any]) -> dict[str, Any]:
    def rate(rows):
        return {"turns": len(rows), "passed": sum(r["passed"] for r in rows)}

    by_step: dict[str, int] = {}
    for r in records:
        for c in r["llm_calls"]:
            by_step[c["step"]] = by_step.get(c["step"], 0) + 1
    return {
        "mode": mode,
        "models": models,
        "completion": completion,
        "overall": rate(records),
        "by_repeat": {str(n): rate([r for r in records if r["repeat"] == n]) for n in sorted({r["repeat"] for r in records})},
        "by_group": {g: rate([r for r in records if r["group"] == g]) for g in sorted({r["group"] for r in records})},
        "by_lang": {g: rate([r for r in records if r["lang"] == g]) for g in sorted({r["lang"] for r in records})},
        "failures_by_layer": {
            layer: sum(1 for r in records if r["failure_layer"] == layer)
            for layer in ("infrastructure", "routing", "application", "extraction")
        },
        "llm_calls_total": sum(r["llm_call_count"] for r in records),
        "llm_calls_by_step": by_step,
        "estimated_cost_usd_total": round(sum(r["estimated_cost_usd"] for r in records), 6),
        "wall_ms_total": round(sum(r["wall_ms"] for r in records), 1),
    }


def write_report(out_dir: Path, records, summary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Clarification Continuity V1 - {summary['mode']}",
        "",
        f"Models: {summary['models']}",
        f"Completion: {summary['completion']}; by repeat: {summary['by_repeat']}",
        f"Passed {summary['overall']['passed']}/{summary['overall']['turns']} turns; "
        f"failures by layer: {summary['failures_by_layer']}",
        f"LLM calls: {summary['llm_calls_total']} {summary['llm_calls_by_step']}; "
        f"estimated cost ${summary['estimated_cost_usd_total']} (len/4 estimate, excludes thinking tokens)",
        "",
        "| repeat | case | turn | lang | result | layer | issues | pending_resolution | action | calls | ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        issues = "; ".join(f"{k}: {', '.join(v)}" for k, v in r["issues"].items()).replace("|", "/")
        router = r["router_decision"]
        lines.append(
            f"| {r['repeat']} | {r['case_id']} | {r['turn']} | {r['lang']} | {'PASS' if r['passed'] else 'FAIL'} | "
            f"{r['failure_layer'] or ''} | {issues} | {router.get('pending_resolution', '')} | {router.get('action', '')} | "
            f"{r['llm_call_count']} | {r['wall_ms']} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--cases", help="comma-separated case ids")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=150)
    args = parser.parse_args()

    cases = load_cases(set(args.cases.split(",")) if args.cases else None)

    if args.list:
        total = {"router": 0, "intent_update": 0, "factual_extraction": 0}
        for case in cases:
            calls = nominal_calls(case)
            total = {k: total[k] + calls[k] for k in total}
            print(f"{case['id']} [{case['group']}/{case['lang']}] turns={len(case['turns'])} calls={calls}")
            for t in case["turns"]:
                print(f"    {t['message']!r} -> {t['expect']}")
        print(f"nominal calls per repeat: {total} = {sum(total.values())}")
        return

    if args.live:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")

    from app.config.llm import get_gemini_model, get_llm_profile
    from app.config.settings import ROUTER_LLM_PROFILE

    models = {
        "router": f"{ROUTER_LLM_PROFILE} -> {get_llm_profile(ROUTER_LLM_PROFILE).model}",
        "intent_update/extraction/factual": get_gemini_model(),
    }
    print(f"models: {models}")

    records, completion = asyncio.run(run(cases, repeats=args.repeats, dry_run=args.dry_run, max_calls=args.max_calls))
    mode_name = "live" if args.live else "dry-run"
    summary = summarize(records, mode=mode_name, models=models, completion=completion)
    out_dir = OUTPUT_ROOT / f"{mode_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    write_report(out_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
