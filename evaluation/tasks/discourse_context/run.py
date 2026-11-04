"""
Structured discourse context experiment - A/B evaluation runner.

Hypothesis: a small structured representation of conversational context
(C2 open question, C3 focused results + last answered factual question)
improves incomplete multi-turn follow-ups without breaking working flows.

    A = current architecture (TurnConfig default)
    B = same, with structured discourse context enabled
        (TurnConfig.discourse_context=True - added in Step 2; until then
        --variant B stops with an explanation)

Both variants run the same frozen dialogues
(evaluation/datasets/discourse_context/dialogues_v1.jsonl) from the same
session fixtures built from real fixture listings, through process_turn.

Real LLM calls (--live): router, Intent Update / intent extraction and
factual question extraction. Always deterministic: retrieval (first two
fixture listings matching city, property type and availability window -
ranking is out of scope), fact matching (no textual fallback calls) and
response wording.

Scoring uses observable outcomes only, never the golden labels as input:
operation (answer / ask_reference / ask_action / update / chat), checked
listings by position in the result set shown at that turn, facts asked,
SearchRequest expectations and whether the shown set was kept or
replaced. `context_expect` is diagnostic only (variant B, Step 2).

Modes: --list, --dry-run (scripted stand-ins, no API calls), --live.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import ExitStack
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from evaluation.tasks.clarification_continuity.run import Recorder, _wrap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "discourse_context" / "dialogues_v1.jsonl"
FIXTURES_PATH = PROJECT_ROOT / "fixtures" / "listings_sample.json"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "outputs" / "discourse_context"

REFERENCE_DATE = date(2026, 11, 2)
CHECK_IN, CHECK_OUT = date(2027, 4, 12), date(2027, 4, 15)
VARIANT_FLAG = "discourse_context"

# Session fixtures: (city, property type, shown listing ids in display order).
SESSIONS = {
    "baku_hotels": ("Baku", "hotel", ["baku-hotel-3", "baku-hotel-5"]),
    "baku_hotels_gym": ("Baku", "hotel", ["baku-hotel-3", "baku-hotel-2"]),
    "baku_apartments": ("Baku", "apartment", ["baku-apt-2", "baku-apt-12"]),
    "baku_apartments_balcony": ("Baku", "apartment", ["baku-apt-2", "baku-apt-7"]),
    "tbilisi_hotels": ("Tbilisi", "hotel", ["tbilisi-hotel-1", "tbilisi-hotel-2"]),
}

ACTION_QUESTION = "update the search, or look at the current results"


def load_dialogues(selected: set[str] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [d for d in rows if selected is None or d["id"] in selected]


def _fixture_listings() -> dict[str, Any]:
    from app.schemas.listing import ListingRaw

    return {row["id"]: ListingRaw.model_validate(row) for row in json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))}


def build_session(name: str):
    from app.schemas.filters import PriceConstraint, SearchFilters
    from app.schemas.query import SearchRequest
    from app.schemas.session import ConversationSession
    from app.schemas.shown_result_set import ShownResult, ShownResultSet

    city, property_type, ids = SESSIONS[name]
    listings = _fixture_listings()
    search = SearchRequest(
        city=city,
        check_in=CHECK_IN,
        check_out=CHECK_OUT,
        adults=2,
        property_types=[property_type],
        filters=SearchFilters(price=PriceConstraint(max_amount=600, currency="USD", scope="per_night")),
    )
    shown = ShownResultSet(
        result_set_id=f"rs-{name}",
        items=[ShownResult(result_id=i, listing=listings[i]) for i in ids],
    )
    return ConversationSession(search_request=search, shown_result_set=shown)


# --- deterministic stand-ins ------------------------------------------------------------


def _install_fakes(stack: ExitStack, recorder: Recorder) -> None:
    from app.logic import conversation_flow
    from app.logic.conversation_response_generator import generate_deterministic_conversation_response
    from app.logic.conversation_response_llm import ConversationResponseGenerationResult
    from app.logic.result_query_matching import ResultQueryMatchOutcome
    from app.retrieval.fixtures import _covers_dates
    from app.schemas.match import Ternary
    from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
    from app.schemas.search_response import NormalizedSearchResponse, SearchStatus
    from app.schemas.shown_result_set import ShownResult, ShownResultSet
    from app.services import conversation_service

    listings = list(_fixture_listings().values())

    async def fixture_orchestrate(state, **kwargs):
        recorder.searched.append(state)
        wanted_types = {getattr(t, "value", t) for t in (state.property_types or [])}
        picked = [
            item
            for item in listings
            if (item.city or "").lower() == (state.city or "").lower()
            and (not wanted_types or getattr(item.property_type, "value", item.property_type) in wanted_types)
            and (state.check_in is None or _covers_dates(item, state.check_in, state.check_out))
        ][:2]
        digest = hashlib.sha1(state.model_dump_json().encode()).hexdigest()[:10]
        return (
            NormalizedSearchResponse(status=SearchStatus.RESULTS if picked else SearchStatus.NO_RESULTS, results=[]),
            ShownResultSet(
                result_set_id=f"rs-{digest}",
                items=[ShownResult(result_id=item.id, listing=item) for item in picked],
            ),
        )

    async def fake_matcher(*, result_query, candidate_selection, **kwargs):
        recorder.matched.append({"candidates": [c.result_id for c in candidate_selection.candidates]})
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
            text=generate_deterministic_conversation_response(response_input), source="deterministic"
        )

    stack.enter_context(mock.patch.object(conversation_flow, "orchestrate_search_request", fixture_orchestrate))
    stack.enter_context(mock.patch.object(conversation_flow, "match_result_query_against_candidates", fake_matcher))
    stack.enter_context(
        mock.patch.object(conversation_service, "generate_conversation_response_with_llm", deterministic_response)
    )


def _install_recorders(stack: ExitStack, recorder: Recorder, *, dry_run: bool, script: dict[str, Any]) -> None:
    from app.logic import conversation_flow, intent_router, intent_update

    route_real = _dry_route(script) if dry_run else conversation_flow.route_conversation_async
    update_real = _dry_update(script) if dry_run else intent_update.route_intent_update_patch_async
    factual_real = _dry_factual(script) if dry_run else conversation_flow.extract_factual_constraints_async
    extraction_real = intent_router.route_intent_adk_async

    stack.enter_context(
        mock.patch.object(
            conversation_flow,
            "route_conversation_async",
            _wrap(recorder.router, route_real, lambda *, router_input, **kw: {"message": router_input.user_message}),
        )
    )
    stack.enter_context(
        mock.patch.object(
            intent_update,
            "route_intent_update_patch_async",
            _wrap(recorder.intent_update, update_real, lambda previous_state, user_message, **kw: {"message": user_message}),
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
            _wrap(recorder.factual, factual_real, lambda user_message, **kw: {"message": user_message}),
        )
    )


# Scripted stand-ins for --dry-run: they follow the expected outcome so a
# dry run exercises the harness and the application path, never a model.


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
        shown = (router_input.latest_result_context or {}).get("shown_results", [])
        op = expect["operation"]
        fields = dict(
            action=ConversationAction.GENERAL_CHAT,
            reason="dry-run",
            decision_status=ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.NONE,
            result_scope=ResultScope.NOT_APPLICABLE,
            target_result_id="",
        )
        if op == "answer":
            candidates = expect["candidates"]
            fields["action"] = ConversationAction.LISTING_QUESTION
            if candidates == "all":
                fields["result_scope"] = ResultScope.CURRENT_RESULTS
            else:
                fields["result_scope"] = ResultScope.SPECIFIC_RESULT
                fields["target_result_id"] = shown[candidates[0] - 1]["result_id"]
        elif op in {"ask_reference", "ask_action"}:
            fields["action"] = ConversationAction.LISTING_QUESTION
            fields["decision_status"] = ConversationDecisionStatus.NEEDS_CLARIFICATION
            if op == "ask_reference":
                fields["clarification_reason"] = ClarificationReason.AMBIGUOUS_REFERENCE
                fields["result_scope"] = ResultScope.SPECIFIC_RESULT
            else:
                fields["clarification_reason"] = ClarificationReason.AMBIGUOUS_ACTION
                fields["result_scope"] = ResultScope.CURRENT_RESULTS
        elif op == "update":
            fields["action"] = ConversationAction.UPDATE_SEARCH

        if router_input.pending_clarification is not None:
            resolution = "continue_pending" if op == "answer" else "new_request"
            return PendingAwareConversationActionDecision(**fields, pending_resolution=resolution)
        return ConversationActionDecision(**fields)

    return route


def _constraint(text, priority="must"):
    from app.schemas.constraints import (
        ConstraintCategory,
        ConstraintMappingStatus,
        EvidenceStrategy,
        UserConstraint,
    )
    from app.schemas.fields import Field

    known = {f.value for f in Field}
    field = text.replace(" ", "_")
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=priority,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN if field in known else ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[Field(field)] if field in known else [],
        evidence_strategy=EvidenceStrategy.STRUCTURED if field in known else EvidenceStrategy.TEXTUAL,
    )


def _dry_update(script):
    from app.schemas.intent_patch import SearchIntentPatch

    async def update(previous_state, user_message, **kwargs):
        search = script["expect"].get("search", {})
        equals = search.get("equals", {})
        return SearchIntentPatch(
            set_city=equals.get("city"),
            set_check_in=equals.get("check_in"),
            set_check_out=equals.get("check_out"),
            add_constraints=[_constraint(k, v) for k, v in search.get("constraints", {}).items()],
        )

    return update


def _dry_factual(script):
    from app.schemas.result_query import ResultQuery

    async def factual(user_message, **kwargs):
        return ResultQuery(constraints=[_constraint(f) for f in script["expect"].get("facts", [])])

    return factual


# --- scoring ---------------------------------------------------------------------------

_CONCEPTS = {
    "breakfast": ("breakfast", "завтрак"),
    "gym": ("gym", "fitness", "спортзал"),
    "kitchen": ("kitchen", "кухн"),
    "balcony": ("balcony", "балкон"),
    "sea view": ("sea view", "вид на море", "sea_view"),
    "wifi": ("wifi", "wi-fi"),
}


def _concept(constraint: dict[str, Any]) -> str:
    """Evaluation-only mapping of an extracted fact to a scoring concept."""
    text = " ".join([constraint.get("normalized_text") or "", *(constraint.get("mapped_fields") or [])]).lower()
    for concept, needles in _CONCEPTS.items():
        if any(n in text for n in needles):
            return concept
    return f"other:{text}"


def _observed_operation(turn, recorder, shown_titles: list[str]) -> str:
    ran_factual = bool(recorder.factual)
    ran_update = bool(recorder.intent_update or recorder.intent_extraction)
    message = turn.assistant_message
    if ran_factual and recorder.matched:
        return "answer"
    if ran_update:
        return "update"
    if turn.outcome_kind == "clarification":
        if shown_titles and all(t in message for t in shown_titles):
            return "ask_reference"
        if ACTION_QUESTION in message:
            return "ask_action"
        return "clarification_other"
    if turn.outcome_kind == "general_chat" and not ran_factual and not ran_update:
        return "chat"
    return f"other:{turn.outcome_kind}"


def score_turn(*, expect, before, turn, recorder, initial_ids: list[str]) -> dict[str, Any]:
    issues: list[str] = []
    shown_before = before.shown_result_set
    shown_ids = [i.result_id for i in shown_before.items] if shown_before else []
    shown_titles = [i.listing.name for i in shown_before.items] if shown_before else []

    if turn.failure is not None:
        issues.append(f"infrastructure:{turn.failure.stage}:{turn.failure.code}")

    operation = _observed_operation(turn, recorder, shown_titles)
    if operation != expect["operation"]:
        issues.append(f"operation={operation}, expected {expect['operation']}")

    checked = recorder.matched[0]["candidates"] if recorder.matched else []
    if "candidates" in expect:
        want = shown_ids if expect["candidates"] == "all" else [shown_ids[p - 1] for p in expect["candidates"]]
        if checked != want:
            issues.append(f"checked={checked}, expected {want}")
    if expect.get("not_from_initial_set") and set(checked) & set(initial_ids):
        issues.append(f"checked a listing from the original result set: {sorted(set(checked) & set(initial_ids))}")

    facts = None
    if recorder.factual and "output" in recorder.factual[-1]:
        facts = sorted(_concept(c) for c in recorder.factual[-1]["output"].get("constraints", []))
    if "facts" in expect and facts != sorted(expect["facts"]):
        issues.append(f"facts={facts}, expected {sorted(expect['facts'])}")

    before_search = before.search_request.model_dump(mode="json") if before.search_request else None
    after_search = turn.session.search_request.model_dump(mode="json") if turn.session.search_request else None
    search_expect = expect.get("search", {})
    if search_expect.get("unchanged") and after_search != before_search:
        issues.append("search changed")
    for key, value in search_expect.get("equals", {}).items():
        if (after_search or {}).get(key) != value:
            issues.append(f"search.{key}={(after_search or {}).get(key)!r}, expected {value!r}")
    for key in search_expect.get("preserve", []):
        if (after_search or {}).get(key) != (before_search or {}).get(key):
            issues.append(f"search.{key} not preserved")
    for concept, priority in search_expect.get("constraints", {}).items():
        got = [c["priority"] for c in (after_search or {}).get("constraints", []) if _concept(c) == concept]
        if got != [priority]:
            issues.append(f"constraint {concept} priorities={got}, expected {priority}")

    after_set = turn.session.shown_result_set.result_set_id if turn.session.shown_result_set else None
    before_set = shown_before.result_set_id if shown_before else None
    shown_state = "same" if after_set == before_set else "replaced"
    if expect.get("shown") and shown_state != expect["shown"]:
        issues.append(f"shown={shown_state}, expected {expect['shown']}")

    calls = turn.debug.telemetry["llm"]["calls"] if turn.debug and turn.debug.telemetry else []
    return {
        "operation": operation,
        "checked": checked,
        "facts": facts,
        "shown_state": shown_state,
        "router_decision": (turn.debug.parsed_intent or {}).get("router") or {},
        "pending_out": turn.session.pending_clarification.model_dump(mode="json")
        if turn.session.pending_clarification
        else None,
        "discourse_out": _discourse_snapshot(turn.session),
        "search_after": after_search,
        "assistant_message": turn.assistant_message,
        "factual_calls": list(recorder.factual),
        "intent_update_calls": list(recorder.intent_update),
        "intent_extraction_calls": list(recorder.intent_extraction),
        "llm_calls": [{"step": c.get("step"), "success": c.get("success"), "error": c.get("error")} for c in calls],
        "llm_call_count": len(calls),
        "latency_ms": turn.telemetry.latency_ms,
        "estimated_cost_usd": turn.telemetry.estimated_cost_usd,
        "passed": not issues,
        "issues": issues,
    }


def _discourse_snapshot(session) -> Any:
    # Variant B state (Step 2); None while the feature does not exist.
    value = getattr(session, VARIANT_FLAG, None)
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


# --- run -------------------------------------------------------------------------------


def make_config(variant: str):
    from app.schemas.turn import TurnConfig

    if variant == "A":
        return TurnConfig(persist_telemetry=False, include_debug=True)
    if VARIANT_FLAG not in TurnConfig.model_fields:
        raise SystemExit(
            f"Variant B needs TurnConfig.{VARIANT_FLAG} (Step 2). Only variant A can run now."
        )
    return TurnConfig(persist_telemetry=False, include_debug=True, **{VARIANT_FLAG: True})


async def run(dialogues, *, variant: str, repeats: int, dry_run: bool, max_calls: int):
    from app.config import clock
    from app.services.conversation_service import process_turn

    config = make_config(variant)
    recorder = Recorder()
    script: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    total_calls, completed, stopped = 0, 0, None

    with ExitStack() as stack:
        _install_fakes(stack, recorder)
        _install_recorders(stack, recorder, dry_run=dry_run, script=script)
        for repeat in range(1, repeats + 1):
            for dialogue in dialogues:
                if stopped:
                    break
                session = build_session(dialogue["session"])
                initial_ids = [i.result_id for i in session.shown_result_set.items]
                for index, spec in enumerate(dialogue["turns"]):
                    if total_calls >= max_calls:
                        stopped = f"--max-calls {max_calls} reached in repeat {repeat} before {dialogue['id']} t{index + 1}"
                        break
                    recorder.reset()
                    script["expect"] = spec["expect"]
                    started = time.perf_counter()
                    with clock.use_reference_date(REFERENCE_DATE):
                        turn = await process_turn(session, spec["message"], config=config)
                    record = score_turn(
                        expect=spec["expect"], before=session, turn=turn, recorder=recorder, initial_ids=initial_ids
                    )
                    record.update(
                        variant=variant,
                        repeat=repeat,
                        dialogue=dialogue["id"],
                        cls=dialogue["class"],
                        lang=dialogue["lang"],
                        turn=index + 1,
                        message=spec["message"],
                        hypothesis_turn=bool(spec.get("hypothesis_turn")),
                        expected=spec["expect"],
                        context_expect=spec.get("context_expect"),
                        wall_ms=round((time.perf_counter() - started) * 1000, 1),
                    )
                    records.append(record)
                    total_calls += record["llm_call_count"]
                    status = "PASS" if record["passed"] else "FAIL"
                    print(f"{variant} r{repeat} {dialogue['id']} t{index + 1} {status} {record['issues'] or ''}")
                    session = turn.session
            if stopped:
                print(f"STOPPED: {stopped}")
                break
            completed = repeat
    return records, {"requested_repeats": repeats, "completed_repeats": completed, "stopped": stopped}


def summarize(records, *, variant, mode, models, completion):
    def rate(rows):
        return {"turns": len(rows), "passed": sum(r["passed"] for r in rows)}

    dialogue_runs: dict[tuple, bool] = {}
    for r in records:
        key = (r["dialogue"], r["repeat"])
        dialogue_runs[key] = dialogue_runs.get(key, True) and r["passed"]
    per_dialogue: dict[str, list[bool]] = {}
    for (dialogue, _), ok in sorted(dialogue_runs.items()):
        per_dialogue.setdefault(dialogue, []).append(ok)

    return {
        "variant": variant,
        "mode": mode,
        "models": models,
        "completion": completion,
        "turns": rate(records),
        "hypothesis_turns": rate([r for r in records if r["hypothesis_turn"]]),
        "by_class": {c: rate([r for r in records if r["cls"] == c]) for c in sorted({r["cls"] for r in records})},
        "by_lang": {c: rate([r for r in records if r["lang"] == c]) for c in sorted({r["lang"] for r in records})},
        "dialogue_success_rate": round(sum(dialogue_runs.values()) / max(1, len(dialogue_runs)), 3),
        "pass_hat_k": {d: all(v) for d, v in per_dialogue.items()},
        "llm_calls_total": sum(r["llm_call_count"] for r in records),
        "estimated_cost_usd_total": round(sum(r["estimated_cost_usd"] for r in records), 6),
        "wall_ms_total": round(sum(r["wall_ms"] for r in records), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--dialogues", help="comma-separated dialogue ids")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=150)
    args = parser.parse_args()

    dialogues = load_dialogues(set(args.dialogues.split(",")) if args.dialogues else None)

    if args.list:
        for d in dialogues:
            print(f"{d['id']} [{d['class']}/{d['lang']}] {d['capability']} session={d['session']}")
            for t in d["turns"]:
                mark = "*" if t.get("hypothesis_turn") else " "
                print(f"  {mark} {t['message']!r} -> {t['expect']}")
        print(f"{len(dialogues)} dialogues, {sum(len(d['turns']) for d in dialogues)} turns, "
              f"{sum(1 for d in dialogues for t in d['turns'] if t.get('hypothesis_turn'))} hypothesis turns")
        return

    if args.live:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")

    from app.config.llm import get_gemini_model, get_llm_profile
    from app.config.settings import ROUTER_LLM_PROFILE

    models = {"router": get_llm_profile(ROUTER_LLM_PROFILE).model, "extractors": get_gemini_model()}
    print(f"variant {args.variant}, models: {models}")

    records, completion = asyncio.run(
        run(dialogues, variant=args.variant, repeats=args.repeats, dry_run=args.dry_run, max_calls=args.max_calls)
    )
    mode_name = "live" if args.live else "dry-run"
    summary = summarize(records, variant=args.variant, mode=mode_name, models=models, completion=completion)
    out_dir = OUTPUT_ROOT / f"{mode_name}_{args.variant}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
