"""
MANUAL, LOCAL-ONLY migration comparison runner: legacy fallback-only
evaluation vs. the new AUTHORITATIVE_FOR_SUPPORTED soft-evidence
evaluation, on REAL current Apify hotel data, for a raw user query you
type yourself.

NOT part of any test suite. NOT imported anywhere. NOT wired into any
production entry point. Uses only existing, already-shipped production
code (app.logic.intent_router, app.retrieval, app.logic.listing_evaluation) -
nothing here reimplements or bypasses production logic. The one
exception is a second, FREE call to the already-computed
app.logic.soft_constraint_decision_policy.decide_constraint per hotel/
constraint for richer display (same pure function AUTHORITATIVE mode
already called once internally to build its constraint_resolution_results
entry - calling it again here on the same, already-computed
SoftPreferenceEvidence makes no new API call and cannot change any
result).

Why TWO independent evaluate_listings calls (not one shadow-mode call):
Phase C's AUTHORITATIVE_FOR_SUPPORTED mode intentionally SKIPS the
legacy textual fallback for constraints the new pipeline resolves - so
a single evaluate_listings call can no longer show "legacy vs new" the
way the old SHADOW-mode comparison did (SHADOW always ran the legacy
fallback for every eligible constraint, since old vs new was designed
to be directly comparable in that mode). To compare LEGACY's real
production behavior against AUTHORITATIVE's real production behavior,
they must be two separate evaluate_listings invocations, each run
exactly as it is in production for that mode.

Flow:
    raw --query text
      -> app.logic.intent_router.build_search_request_adk_async   (ONCE)
         (the SAME ADK intent-extraction agent production uses)
      -> real SearchRequest with parsed UserConstraint[]
      -> app.retrieval.get_candidates(source="apify")              (ONCE, REAL Apify call)
      -> the SAME retrieved listings are deep-copied into two
         independent lists (see _isolated_copies) before being handed
         to two SEPARATE evaluate_listings calls, each with its own
         RequestTrace, so neither run's mutation of its own working
         dicts (item["score"], item["constraint_resolution_results"],
         item["soft_preference_evidence"], ...) can leak into the
         other - see _isolated_copies' docstring for why this is
         defensive rather than a response to a known bug.

    RUN A - LEGACY:
      evaluate_listings(req, listings_a, fallback_policy=None,
                         soft_evidence_pipeline_policy=None)
      -> normal production default: soft-evidence pipeline fully OFF
         (see listing_evaluation._build_soft_evidence_pipeline_policy),
         legacy fallback behaves exactly as it does today.

    RUN B - AUTHORITATIVE_FOR_SUPPORTED:
      evaluate_listings(req, listings_b, fallback_policy=None,
                         soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(
                             enabled=True,
                             integration_mode=SoftEvidenceIntegrationMode.AUTHORITATIVE_FOR_SUPPORTED,
                             shadow_hotel_top_k=MAX_HOTELS))
      -> supported soft constraints (quiet, remote_work, family_friendly,
         breakfast_quality, cleanliness, nightlife, bed_comfort) are
         resolved by the new deterministic soft-evidence policy and
         skip the legacy fallback for that constraint on that listing;
         unsupported constraints still use fallback_policy=None's
         normal production legacy fallback, exactly as designed.

Paid/external calls this script makes when run:
    1. One real Gemini call for intent/constraint extraction
       (app.logic.intent_router - same call production makes).
    2. One real Apify call to fetch current hotel listings.
    3. RUN A: legacy textual-fallback Gemini call(s), if any MUST/NICE
       constraint needs them (existing behavior, unchanged).
    4. RUN B: real gemini-embedding-001 calls (1 shared query batch +
       up to MAX_HOTELS pool-embedding batches) + real gemini-2.5-flash
       semantic-verifier calls for supported constraints, PLUS legacy
       textual-fallback Gemini call(s) for any unsupported constraint.

Two runs -> roughly double the paid-call volume of the old single-call
SHADOW comparison script. Both runs use the SAME parsed SearchRequest
and the SAME retrieved (deep-copied) listings - only ONE Apify call and
ONE intent-extraction call happen total, regardless of the two
evaluation runs.

Run manually from your terminal - see the accompanying summary for the
exact command. This script is intentionally NOT executed by the
assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Loads .env the same way every other script in this repo does
# (e.g. scripts/benchmarks/benchmark_apify_retrieval.py) - no custom
# credential handling.
load_dotenv()

from app.logic.intent_router import build_search_request_adk_async
from app.logic.listing_evaluation import ListingEvaluationResult, evaluate_listings
from app.logic.soft_constraint_decision_policy import decide_constraint
from app.observability.trace import RequestTrace
from app.retrieval import get_candidates
from app.schemas.listing import ListingRaw
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence_pipeline_policy import SoftEvidenceIntegrationMode, SoftEvidencePipelinePolicy

MAX_HOTELS = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual migration comparison: LEGACY-only evaluation vs. "
            "AUTHORITATIVE_FOR_SUPPORTED evaluation, as two independent "
            "runs over the same real Apify listings."
        )
    )
    parser.add_argument(
        "--query",
        required=True,
        help='Raw user search text, e.g. "Find a quiet hotel in Baku that is good for remote work"',
    )
    parser.add_argument(
        "--checkin",
        default=None,
        help="Override/fallback check-in date (YYYY-MM-DD), used only if the query itself did not parse one.",
    )
    parser.add_argument(
        "--checkout",
        default=None,
        help="Override/fallback check-out date (YYYY-MM-DD), used only if the query itself did not parse one.",
    )
    parser.add_argument(
        "--adults",
        type=int,
        default=None,
        help="Override adults count. If omitted, uses whatever the query parsed (SearchRequest's own default is 2).",
    )
    parser.add_argument(
        "--rooms",
        type=int,
        default=None,
        help="Override rooms count. If omitted, uses whatever the query parsed (SearchRequest's own default is 1).",
    )
    return parser.parse_args()


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_subheader(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


async def _resolve_search_request(args: argparse.Namespace):
    """
    Runs the query through the EXISTING production intent/constraint
    extraction path (one real Gemini call), then applies CLI
    checkin/checkout/adults/rooms only as a fallback for whatever the
    parser did not produce - never overriding a value the parser
    already resolved from the query itself.
    """
    trace = RequestTrace()

    print("USER QUERY:")
    print(f"  {args.query}")

    req = await build_search_request_adk_async(args.query, trace=trace)

    cli_checkin = _parse_iso_date(args.checkin)
    cli_checkout = _parse_iso_date(args.checkout)

    if req.check_in is None and cli_checkin is not None:
        req.check_in = cli_checkin
    if req.check_out is None and cli_checkout is not None:
        req.check_out = cli_checkout
    if args.adults is not None:
        req.adults = args.adults
    if args.rooms is not None:
        req.rooms = args.rooms

    return req, trace


def _print_parsed_constraints(req) -> None:
    _print_header("PARSED CONSTRAINTS")
    if not req.constraints:
        print("  (none extracted from the query)")
        return

    for c in req.constraints:
        mapped = [f.value if hasattr(f, "value") else str(f) for f in (c.mapped_fields or [])]
        print(f"  - raw_text:        {c.raw_text}")
        print(f"    normalized_text: {c.normalized_text}")
        print(f"    priority:        {c.priority.value}")
        print(f"    mapping_status:  {c.mapping_status.value}")
        print(f"    mapped_fields:   {mapped or '—'}")
        print()


def _print_search_parameters(req) -> None:
    _print_header("SEARCH PARAMETERS")
    print(f"  city:       {_fmt(req.city)}")
    print(f"  check_in:   {_fmt(req.check_in)}")
    print(f"  check_out:  {_fmt(req.check_out)}")
    print(f"  adults:     {req.adults}")
    print(f"  rooms:      {req.rooms}")


def _isolated_copies(listings: list[ListingRaw], n: int) -> list[list[ListingRaw]]:
    """
    Returns `n` independent deep copies of `listings` (via ListingRaw's
    own pydantic .model_copy(deep=True)) - one list per evaluation run.

    Defensive, not a response to a confirmed bug: nothing in
    evaluate_listings/its helpers is known to mutate a ListingRaw
    object's own fields today (only the per-run `ranked` dict items -
    item["score"], item["constraint_resolution_results"],
    item["soft_preference_evidence"], etc. - which are already fresh
    dicts built once per evaluate_listings call, never shared across
    calls). Deep-copying the listing objects themselves removes any
    possibility of Run A and Run B observing each other's state,
    regardless of whether that assumption ever changes later - the
    retrieved data is still the SAME single real Apify response either
    way, just not the same Python objects.
    """
    return [[listing.model_copy(deep=True) for listing in listings] for _ in range(n)]


@dataclass
class RunResult:
    label: str
    result: ListingEvaluationResult
    trace: RequestTrace
    wall_clock_ms: float


async def _run_evaluation(
    label: str,
    req,
    listings: list[ListingRaw],
    *,
    soft_evidence_pipeline_policy: SoftEvidencePipelinePolicy | None,
) -> RunResult:
    """
    One isolated evaluate_listings call with its own RequestTrace, wall-
    clock-timed here in the script (evaluate_listings/RequestTrace do
    not report their own total call latency - see the module docstring
    of app.observability.trace - so this manual measurement is the only
    place total per-run wall time is available, and it is never written
    back into `trace` or any production telemetry object).
    """
    trace = RequestTrace()
    started = time.perf_counter()
    result = await evaluate_listings(
        req,
        listings,
        fallback_policy=None,  # normal production default in BOTH runs
        soft_evidence_pipeline_policy=soft_evidence_pipeline_policy,
        semantic_verifier_policy=SemanticVerifierPolicy(),
        trace=trace,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return RunResult(label=label, result=result, trace=trace, wall_clock_ms=elapsed_ms)


def _decision_by_constraint_id(ranked_item: dict | None) -> dict[str, dict]:
    if ranked_item is None:
        return {}
    out: dict[str, dict] = {}
    for r in ranked_item.get("constraint_resolution_results") or []:
        cid = r.get("constraint_id")
        if cid:
            out[cid] = r
    return out


def _print_resolution_entry(r: dict) -> None:
    print(f"    decision={r.get('decision')}  priority={r.get('priority')}  "
          f"status={r.get('resolution_status')}  source_stage={r.get('source_stage')}  "
          f"confidence={r.get('confidence')}")
    print(f"    reason: {r.get('reason')}")
    for ev in r.get("evidence") or []:
        print(f"    evidence[{ev.get('source')}]: {ev.get('snippet')!r} (path={ev.get('path')})")


def _print_atomic_claims_for_constraint(evidence, constraint) -> None:
    """
    Re-derives the full SoftConstraintDecision for one constraint from
    the ALREADY-COMPUTED SoftPreferenceEvidence for this hotel - a pure,
    free, no-API-call re-call of the exact production function
    (app.logic.soft_constraint_decision_policy.decide_constraint)
    AUTHORITATIVE mode already called once internally. Used here purely
    for richer manual-inspection output (factual_state, families,
    claim_ids, decisive_claim_ids) that the adapted
    constraint_resolution_results dict does not itself carry.
    """
    decision = decide_constraint(constraint, evidence)
    if decision is None:
        return  # unsupported by the new pipeline - nothing to show here

    print(f"      families:            {decision.families}")
    print(f"      preference_direction: {decision.preference_direction.value}")
    print(f"      factual_state:       {decision.factual_state.value}")
    print(f"      claim_ids consulted: {decision.claim_ids}")
    print(f"      decisive_claim_ids:  {decision.decisive_claim_ids}")
    if decision.technical_notes:
        print(f"      technical_notes:     {decision.technical_notes}")

    claims_by_id = {c.claim_id: c for c in evidence.claims}
    for claim_id in decision.claim_ids:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            print(f"      * {claim_id}: (not evaluated)")
            continue
        print(f"      * {claim_id}  relation={claim.relation.value}  "
              f"retrieval_status={claim.retrieval_status.value}  "
              f"retrieved_candidate_count={claim.retrieved_candidate_count}")
        if claim.retrieval_errors:
            print(f"          retrieval_errors: {claim.retrieval_errors}")
        for ei in claim.evidence_items:
            method = ei.resolution_method.value  # "deterministic" or "gemini"
            print(f"          - [{method}/{ei.resolution_status.value}] relation={ei.relation}  "
                  f"score={ei.retrieval_score}")
            print(f"            text: {ei.evidence_text!r}")
            print(f"            source: {ei.source_type} / {ei.source_path}")
            if ei.verifier_reason:
                print(f"            verifier_reason: {ei.verifier_reason}")
            if ei.error:
                print(f"            error: {ei.error}")


def _print_per_hotel_comparison(
    req,
    original_listings: list[ListingRaw],
    run_a: RunResult,
    run_b: RunResult,
) -> None:
    items_a = {item["listing_id"]: item for item in run_a.result.ranked_items}
    items_b = {item["listing_id"]: item for item in run_b.result.ranked_items}

    for listing in original_listings:
        listing_id = listing.id
        _print_header(f"HOTEL: {listing.name}  (id={listing_id})")

        item_a = items_a.get(listing_id)
        item_b = items_b.get(listing_id)

        _print_subheader("LEGACY")
        if item_a is None:
            print("  ELIMINATED - not present in LEGACY's final ranked_items "
                  "(rejected by structured filtering, a MUST/FORBIDDEN fallback NO, or numeric filters).")
        else:
            print(f"  eligibility: SURVIVED   score={item_a.get('score')}")
            results = item_a.get("constraint_resolution_results") or []
            if not results:
                print("  constraint_resolution_results: (empty)")
            for r in results:
                print(f"  * {r.get('normalized_text')!r}")
                _print_resolution_entry(r)

        _print_subheader("AUTHORITATIVE_FOR_SUPPORTED")
        if item_b is None:
            print("  ELIMINATED - not present in AUTHORITATIVE's final ranked_items "
                  "(rejected by structured filtering, a MUST/FORBIDDEN NO/violation, or numeric filters).")
        else:
            print(f"  eligibility: SURVIVED   score={item_b.get('score')}")
            results = item_b.get("constraint_resolution_results") or []
            if not results:
                print("  constraint_resolution_results: (empty)")
            for r in results:
                print(f"  * {r.get('normalized_text')!r}")
                _print_resolution_entry(r)
                if r.get("source_stage") == "soft_evidence":
                    evidence = item_b.get("soft_preference_evidence")
                    constraint = next((c for c in (req.constraints or []) if c.id == r.get("constraint_id")), None)
                    if evidence is not None and constraint is not None:
                        print("      --- atomic claims (re-derived, no new API calls) ---")
                        _print_atomic_claims_for_constraint(evidence, constraint)

        # ---- per-hotel compact comparison table ----
        _print_subheader("compact comparison (this hotel)")
        by_id_a = _decision_by_constraint_id(item_a)
        by_id_b = _decision_by_constraint_id(item_b)
        all_constraint_ids = {c.id for c in (req.constraints or [])}

        header = f"  {'constraint':<32} {'LEGACY':<12} {'AUTHORITATIVE':<14} {'source(auth)':<14}"
        print(header)
        for c in req.constraints or []:
            ra = by_id_a.get(c.id)
            rb = by_id_b.get(c.id)
            legacy_decision = ra.get("decision") if ra else "—"
            auth_decision = rb.get("decision") if rb else "—"
            auth_source = rb.get("source_stage") if rb else "—"
            changed = " *** CHANGED ***" if legacy_decision != auth_decision else ""
            label = c.normalized_text[:31]
            print(f"  {label:<32} {str(legacy_decision):<12} {str(auth_decision):<14} {str(auth_source):<14}{changed}")

        eligibility_changed = (item_a is None) != (item_b is None)
        if eligibility_changed:
            print("\n  *** ELIGIBILITY CHANGED BETWEEN MODES FOR THIS HOTEL ***")
            print(f"    LEGACY survived:        {item_a is not None}")
            print(f"    AUTHORITATIVE survived: {item_b is not None}")


def _print_final_candidate_set_summary(
    original_listings: list[ListingRaw],
    run_a: RunResult,
    run_b: RunResult,
) -> None:
    _print_header("FINAL CANDIDATE SET — LEGACY vs AUTHORITATIVE")

    ids_a = {item["listing_id"] for item in run_a.result.ranked_items}
    ids_b = {item["listing_id"] for item in run_b.result.ranked_items}
    names = {listing.id: listing.name for listing in original_listings}

    both = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"  survived BOTH modes:           {[names.get(i, i) for i in both] or 'none'}")
    print(f"  survived LEGACY only:          {[names.get(i, i) for i in only_a] or 'none'}")
    print(f"  survived AUTHORITATIVE only:   {[names.get(i, i) for i in only_b] or 'none'}")
    print(f"  candidate set identical:       {ids_a == ids_b}")


def _sum_or_none(values: list[Any]) -> Any:
    known = [v for v in values if v is not None]
    if not known:
        return None if len(known) != len(values) else 0
    return round(sum(known), 6)


def _print_legacy_telemetry(run: RunResult) -> None:
    _print_header("TELEMETRY — LEGACY")
    trace = run.trace

    fallback_calls = [c for c in trace.llm_calls if c.step == "constraint_textual_fallback"]

    print(f"  manual wall-clock (script-measured, around evaluate_listings): {run.wall_clock_ms} ms")
    print("\n  fallback LLM calls (step=constraint_textual_fallback):")
    print(f"    calls:            {len(fallback_calls)}")
    print(f"    latency_ms total: {_sum_or_none([c.latency_ms for c in fallback_calls])}"
          f"  (legacy fallback call sites do not record latency_ms - expect None/0 here)")
    print(f"    tokens total:     {_sum_or_none([c.total_tokens for c in fallback_calls])}")
    print(f"    cost_usd total:   {_sum_or_none([c.estimated_cost_usd for c in fallback_calls])}")
    print(f"\n  soft_preference_evidence attached to any item: "
          f"{any(item.get('soft_preference_evidence') is not None for item in run.result.ranked_items)}"
          f"  (must be False in LEGACY - proves the new pipeline did not run)")


def _print_authoritative_telemetry(run: RunResult) -> None:
    _print_header("TELEMETRY — AUTHORITATIVE_FOR_SUPPORTED")
    trace = run.trace

    fallback_calls = [c for c in trace.llm_calls if c.step == "constraint_textual_fallback"]
    verifier_calls = [c for c in trace.llm_calls if c.step == "semantic_evidence_verifier"]
    embedding_calls = [c for c in trace.external_calls if c.provider == "gemini_embedding"]

    print(f"  manual wall-clock (script-measured, around evaluate_listings): {run.wall_clock_ms} ms")

    authoritative_step = next((s for s in trace.steps if s.name == "soft_evidence_authoritative_layer"), None)
    print(f"\n  soft_evidence_authoritative_layer step wall-clock (actual, production-traced): "
          f"{_fmt(authoritative_step.latency_ms if authoritative_step else None)} ms")

    print("\n  embedding calls (provider=gemini_embedding):")
    print(f"    calls:            {len(embedding_calls)}")
    print(f"    summed latency_ms: {_sum_or_none([c.latency_ms for c in embedding_calls])}"
          f"  (sum of individual calls - may exceed wall-clock under concurrency, expected)")
    print(f"    cost_usd total:   {_sum_or_none([c.estimated_cost_usd for c in embedding_calls])}"
          f"  (real API does not return usage metadata - expect None; not estimated from text length)")

    print("\n  Gemini semantic verifier calls (step=semantic_evidence_verifier):")
    print(f"    calls:            {len(verifier_calls)}")
    print(f"    summed latency_ms: {_sum_or_none([c.latency_ms for c in verifier_calls])}"
          f"  (sum of individual calls - may exceed wall-clock under concurrency, expected)")
    print(f"    tokens total:     {_sum_or_none([c.total_tokens for c in verifier_calls])}")
    print(f"    cost_usd total:   {_sum_or_none([c.estimated_cost_usd for c in verifier_calls])}")

    print("\n  legacy fallback calls used for UNSUPPORTED constraints only "
          "(step=constraint_textual_fallback):")
    print(f"    calls:            {len(fallback_calls)}")
    print(f"    tokens total:     {_sum_or_none([c.total_tokens for c in fallback_calls])}")
    print(f"    cost_usd total:   {_sum_or_none([c.estimated_cost_usd for c in fallback_calls])}")

    if trace.soft_evidence_summary:
        print("\n  soft_evidence_summary:")
        for k, v in trace.soft_evidence_summary.items():
            print(f"    {k}: {v}")


async def main() -> None:
    args = _parse_args()

    req, setup_trace = await _resolve_search_request(args)

    _print_parsed_constraints(req)
    _print_search_parameters(req)

    if not req.city or not req.check_in or not req.check_out:
        print()
        print("Cannot continue: city/check_in/check_out are required and were not")
        print("resolved from the query or from --checkin/--checkout. Re-run with an")
        print("explicit --checkin/--checkout (and a city mentioned in --query).")
        return

    print()
    print(f"Retrieving up to {MAX_HOTELS} REAL current hotels from Apify for {req.city} (ONE call, shared by both runs)...")
    listings = await get_candidates(req, max_items=MAX_HOTELS, source="apify", trace=setup_trace)
    listings = listings[:MAX_HOTELS]
    print(f"Retrieved {len(listings)} listing(s).")

    if not listings:
        print("No listings returned - nothing to compare.")
        return

    listings_a, listings_b = _isolated_copies(listings, 2)

    print()
    print("Running RUN A (LEGACY)...")
    run_a = await _run_evaluation(
        "LEGACY", req, listings_a,
        soft_evidence_pipeline_policy=None,  # -> _build_soft_evidence_pipeline_policy() -> enabled=False
    )
    print(f"  done in {run_a.wall_clock_ms} ms, {len(run_a.result.ranked_items)} listing(s) survived.")

    print("Running RUN B (AUTHORITATIVE_FOR_SUPPORTED)...")
    run_b = await _run_evaluation(
        "AUTHORITATIVE_FOR_SUPPORTED", req, listings_b,
        soft_evidence_pipeline_policy=SoftEvidencePipelinePolicy(
            enabled=True,
            integration_mode=SoftEvidenceIntegrationMode.AUTHORITATIVE_FOR_SUPPORTED,
            shadow_hotel_top_k=MAX_HOTELS,
        ),
    )
    print(f"  done in {run_b.wall_clock_ms} ms, {len(run_b.result.ranked_items)} listing(s) survived.")

    _print_per_hotel_comparison(req, listings, run_a, run_b)
    _print_final_candidate_set_summary(listings, run_a, run_b)
    _print_legacy_telemetry(run_a)
    _print_authoritative_telemetry(run_b)


if __name__ == "__main__":
    asyncio.run(main())
