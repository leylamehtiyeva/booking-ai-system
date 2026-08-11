from __future__ import annotations
from app.schemas.property_semantics import OccupancyType, PropertyType
import asyncio
import json
import os
from app.observability.trace import RequestTrace
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from app.logic.result_selection import select_ranked_items
from google.genai import Client
from google.genai import types as genai_types
from pydantic import ValidationError
from app.agents.intent_router_agent import IntentRoute
from app.config.settings import MAX_ITEMS_HARD_CAP
from app.logic.matcher_structured import match_listing_structured
from app.logic.numeric_filters import evaluate_numeric_filters
from app.retrieval import Source, get_candidates
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.logic.property_semantics import match_occupancy_types, match_property_types
from app.logic.normalize_search_response import normalize_search_response
from app.logic.request_resolution import resolve_required_search_context
from app.logic.occupancy import evaluate_occupancy
from app.config.llm import get_gemini_model
from collections import Counter

from app.logic.constraint_evidence_resolution import (
    resolve_listing_constraints_with_fallback,
)
from app.schemas.fallback_policy import FallbackPolicy






def _fails_must(matches: dict[Field, Any], must_fields: List[Field] | None) -> bool:
    """Strict must-have filter: if any must field is explicitly NO -> reject."""
    for f in (must_fields or []):
        fm = matches.get(f)
        if fm is not None and fm.value == Ternary.NO:
            return True
    return False

def _fails_numeric_filters(numeric_results: List[Any] | None) -> bool:
    """
    Strict numeric filter: if any numeric constraint is explicitly NO -> reject.
    UNCERTAIN is allowed.
    """
    for r in (numeric_results or []):
        if r.value == Ternary.NO:
            return True
    return False


def _priority_value(priority: Any) -> str:
    return getattr(priority, "value", priority)


def _mapping_status_value(mapping_status: Any) -> str:
    return getattr(mapping_status, "value", mapping_status)


def _constraints_by_priority(req: SearchRequest) -> tuple[list[Any], list[Any], list[Any]]:
    constraints = list(req.constraints or [])

    must_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "must"
    ]
    nice_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "nice"
    ]
    forbidden_constraints = [
        c for c in constraints
        if _priority_value(getattr(c, "priority", None)) == "forbidden"
    ]
    return must_constraints, nice_constraints, forbidden_constraints


def _known_mapped_fields(constraints: list[Any]) -> list[Field]:
    out: list[Field] = []
    seen: set[Field] = set()

    for c in constraints:
        if _mapping_status_value(getattr(c, "mapping_status", None)) != "known":
            continue

        for f in getattr(c, "mapped_fields", []) or []:
            if f not in seen:
                seen.add(f)
                out.append(f)

    return out

def _constraint_key(c: Any) -> str:
    constraint_id = getattr(c, "id", None)
    if constraint_id:
        return f"id:{constraint_id}"

    normalized_text = str(getattr(c, "normalized_text", "") or "").strip().casefold()
    if normalized_text:
        return f"text:{normalized_text}"

    raw_text = str(getattr(c, "raw_text", "") or "").strip().casefold()
    return f"raw:{raw_text}"


def _resolution_key(result: dict[str, Any]) -> str:
    constraint_id = result.get("constraint_id")
    if constraint_id:
        return f"id:{constraint_id}"

    normalized_text = str(result.get("normalized_text", "") or "").strip().casefold()
    if normalized_text:
        return f"text:{normalized_text}"

    raw_text = str(result.get("raw_text", "") or "").strip().casefold()
    return f"raw:{raw_text}"


def _unresolved_must_textual_constraints(req: SearchRequest) -> list[Any]:
    out: list[Any] = []

    for c in req.constraints or []:
        if _priority_value(getattr(c, "priority", None)) != "must":
            continue

        if _mapping_status_value(getattr(c, "mapping_status", None)) != "unresolved":
            continue

        evidence_strategy = getattr(c, "evidence_strategy", None)
        evidence_strategy_value = getattr(evidence_strategy, "value", evidence_strategy)

        if evidence_strategy_value not in {"textual", "none", None}:
            continue

        out.append(c)

    return out


def _make_uncertain_resolution_for_unresolved_must(
    *,
    constraint: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    listing = item.get("listing")

    return {
        "listing_id": getattr(listing, "id", None),
        "listing_title": getattr(listing, "name", None) or item.get("listing_name"),

        "constraint_id": getattr(constraint, "id", None),
        "raw_text": getattr(constraint, "raw_text", None) or getattr(constraint, "normalized_text", ""),
        "normalized_text": getattr(constraint, "normalized_text", None) or getattr(constraint, "raw_text", ""),

        "resolver_type": "textual",
        "decision": "UNCERTAIN",
        "resolution_status": "uncertain",
        "confidence": 0.0,
        "reason": "Required textual constraint was not resolved by structured matching or textual fallback.",

        "evidence": [],
        "source_stage": "coverage_normalization",
        "structured_value_before": None,
        "explicit_negative": False,

        # Internal fields used before response normalization.
        "priority": "must",
        "mapping_status": "unresolved",
        "evidence_strategy": "textual",
    }


def _ensure_unresolved_must_constraints_are_represented(
    req: SearchRequest,
    ranked: list[dict[str, Any]],
) -> None:
    """
    Safety invariant:

    Every MUST constraint from constraints[] must have a final decision signal
    before final eligibility is computed.

    Unresolved textual MUST constraints cannot silently disappear.
    If fallback did not produce YES/NO/UNCERTAIN, we add synthetic UNCERTAIN.
    """
    unresolved_must_constraints = _unresolved_must_textual_constraints(req)

    if not unresolved_must_constraints:
        return

    for item in ranked:
        results = list(item.get("constraint_resolution_results") or [])
        existing_keys = {_resolution_key(r) for r in results if isinstance(r, dict)}

        for constraint in unresolved_must_constraints:
            key = _constraint_key(constraint)

            if key in existing_keys:
                continue

            results.append(
                _make_uncertain_resolution_for_unresolved_must(
                    constraint=constraint,
                    item=item,
                )
            )

        item["constraint_resolution_results"] = results

def _parse_iso_date(x: Any) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        try:
            return date.fromisoformat(x.strip())
        except ValueError:
            return None
    return None


def _covers_dates(lst: ListingRaw, check_in: date, check_out: date) -> bool:
    """MVP availability filter for fixtures (safety net).

    Apify typically returns listings already scoped to (city, dates),
    but mocks may contain mixed availability.
    """
    ad = getattr(lst, "available_dates", None)
    if ad is None:
        return True

    if isinstance(ad, dict):
        lst_in_raw = ad.get("check_in")
        lst_out_raw = ad.get("check_out")
    else:
        lst_in_raw = getattr(ad, "check_in", None)
        lst_out_raw = getattr(ad, "check_out", None)

    lst_in = _parse_iso_date(lst_in_raw)
    lst_out = _parse_iso_date(lst_out_raw)

    if lst_in is None or lst_out is None:
        return True

    return lst_in <= check_in and check_out <= lst_out


def _normalize_city(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().casefold()
    return text or None

def _should_apply_local_date_filter(source: Source) -> bool:
    """
    Local date filtering is valid for fixtures because fixtures store
    availability as a broad available_dates range.

    For Apify, dates are already sent to the Booking actor via checkIn/checkOut.
    Re-applying fixture-style available_dates filtering can incorrectly remove
    valid Apify results.
    """
    return source == "fixtures"


def _listing_city_matches(lst: ListingRaw, requested_city: str | None) -> bool:
    requested = _normalize_city(requested_city)
    if not requested:
        return False

    listing_city = _normalize_city(getattr(lst, "city", None))
    if not listing_city:
        return False

    return listing_city == requested


def _gemini_client() -> Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY/GOOGLE_API_KEY")
    return Client(api_key=api_key)


async def _repair_intent_with_llm(
    intent_raw: Dict[str, Any],
    errors: list[dict],
    model: str = get_gemini_model(),
) -> Dict[str, Any]:
    """Ask LLM to repair intent so it matches IntentRoute exactly.

    Contract:
    - constraints is the canonical semantic state
    - only schema-defined keys are allowed
    - invalid legacy enum-like items may be dropped from enum slots
    - preserve user meaning inside constraints whenever possible
    """
    allowed_values = [f.value for f in Field]
    
    system = (
        "You are a JSON repair assistant.\n"
        "Return ONLY a valid JSON object. No markdown. No code fences.\n"
        "Fix the JSON to match the target schema EXACTLY.\n"
        "constraints is the source of truth for user constraints.\n"
        "Do NOT invent new keys.\n"
        "If an invalid legacy enum-like item cannot be mapped safely, remove it from that enum list.\n"
        "Preserve meaningful user meaning inside constraints whenever possible.\n"
    )

    payload = {
        "allowed_fields": allowed_values,
        "validation_errors": errors,
        "input_intent": intent_raw,
        "target_schema": {
            "city": "string|null",
            "check_in": "YYYY-MM-DD|null",
            "check_out": "YYYY-MM-DD|null",
            "nights": "int|null",
            "adults": "int|null",
            "children": "int|null",
            "rooms": "int|null",
            "constraints": [
                {
                    "raw_text": "string",
                    "normalized_text": "string",
                    "priority": "must|nice|forbidden",
                    "category": "amenity|policy|location|layout|numeric|property_type|occupancy|other",
                    "mapping_status": "known|unresolved",
                    "mapped_fields": "list[canonical_key]",
                    "evidence_strategy": "structured|textual|none",
                }
            ],
            "filters": {
                "bedrooms_min": "int|null",
                "bedrooms_max": "int|null",
                "area_sqm_min": "float|null",
                "area_sqm_max": "float|null",
                "bathrooms_min": "float|null",
                "bathrooms_max": "float|null",
                "price": {
                    "min_amount": "float|null",
                    "max_amount": "float|null",
                    "currency": "string|null",
                    "scope": "per_night|total_stay|null",
                },
            },
            "property_types": [
                "ryokan|hotel|apartment|resort|villa|bed_and_breakfast|holiday_home|guest_house|hostel|capsule_hotel|homestay|chalet|lodge|campsite|country_house|love_hotel|house|aparthotel|guesthouse"
            ],
            "occupancy_types": [
                "entire_place|private_room|shared_room|hotel_room"
            ],
        },
    }

    def _call_sync() -> Dict[str, Any]:
        client = _gemini_client()
        resp = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=json.dumps(payload, ensure_ascii=False))],
                )
            ],
            config=genai_types.GenerateContentConfig(system_instruction=system),
        )
        text = (resp.text or "").strip()
        return json.loads(text)

    return await asyncio.to_thread(_call_sync)


def _salvage_only_enum_keys(intent_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Keep only safely parseable enum-based values and collect dropped legacy residue separately.

    Contract:
    - returned intent dict is safe to validate as IntentRoute
    - unknown_requests is not used as a salvage bucket
    - dropped_requests contains invalid legacy enum-like items that could not be preserved
    """
    dropped_requests: list[str] = []

    out: Dict[str, Any] = {
        "city": intent_dict.get("city"),
        "check_in": intent_dict.get("check_in"),
        "check_out": intent_dict.get("check_out"),
        "nights": intent_dict.get("nights"),
        "adults": intent_dict.get("adults"),
        "children": intent_dict.get("children"),
        "rooms": intent_dict.get("rooms"),
        "constraints": intent_dict.get("constraints") or [],
        "filters": intent_dict.get("filters") or {},
        "property_types": [],
        "occupancy_types": [],
    }

    def parse_enum_list(xs, enum_cls):
        ok = []
        for x in xs or []:
            if isinstance(x, enum_cls):
                ok.append(x)
                continue

            if isinstance(x, str):
                s = x.strip()

                try:
                    ok.append(enum_cls(s))
                    continue
                except Exception:
                    pass

                try:
                    ok.append(enum_cls[s])
                    continue
                except Exception:
                    pass

                try:
                    ok.append(enum_cls[s.upper()])
                    continue
                except Exception:
                    pass

                if s:
                    dropped_requests.append(s)
            else:
                dropped_requests.append(str(x))
        return ok

    out["property_types"] = parse_enum_list(intent_dict.get("property_types"), PropertyType)
    out["occupancy_types"] = parse_enum_list(intent_dict.get("occupancy_types"), OccupancyType)

    seen: set[str] = set()
    deduped_dropped: list[str] = []
    for item in dropped_requests:
        cleaned = item.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped_dropped.append(cleaned)

    return out, deduped_dropped

async def _validate_and_repair_intent(intent: Any, attempts: int = 2) -> Tuple[IntentRoute, List[str]]:
    """Validate intent as IntentRoute with up to N LLM repair attempts.

    Returns (intent_obj, dropped_requests).

    dropped_requests:
    - invalid legacy enum-like residue we could not preserve cleanly
    - NOT compatibility unknown_requests
    - NOT canonical unresolved constraints
    """
    intent_work: Dict[str, Any] = intent if isinstance(intent, dict) else {}
    dropped_requests: list[str] = []

    intent_obj: Optional[IntentRoute] = None
    for _ in range(max(0, attempts)):
        try:
            intent_obj = IntentRoute.model_validate(intent_work)
            break
        except ValidationError as e:
            try:
                intent_work = await _repair_intent_with_llm(intent_work, e.errors())
            except Exception:
                break

    if intent_obj is None:
        try:
            intent_obj = IntentRoute.model_validate(intent_work)
        except ValidationError:
            salvaged, salvage_dropped = _salvage_only_enum_keys(intent_work)
            dropped_requests.extend(salvage_dropped)
            intent_obj = IntentRoute.model_validate(salvaged)

    return intent_obj, dropped_requests





def _build_request(
    user_text: str,
    intent_obj: IntentRoute,
    city: str,
    check_in: date,
    check_out: date,
) -> SearchRequest:
    req = SearchRequest(
    city=city,
    check_in=check_in,
    check_out=check_out,
    adults=intent_obj.adults or 2,
    children=intent_obj.children or 0,
    rooms=intent_obj.rooms or 1,
    currency="USD",
    budget_max=None,
    min_guest_rating=None,
    filters=intent_obj.filters,
    property_types=intent_obj.property_types or None,
    occupancy_types=intent_obj.occupancy_types or None,
    constraints=intent_obj.constraints,
)
    # Canonical flow:
    # SearchRequest semantic state is carried by constraints.
    return req


def _rank_structured(req: SearchRequest, listings: List[ListingRaw]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []

    must_constraints, nice_constraints, _ = _constraints_by_priority(req)
    structured_must_fields = _known_mapped_fields(must_constraints)
    rejection_reasons = Counter()
    for lst in listings:
        report = match_listing_structured(lst, req)
        numeric_results = evaluate_numeric_filters(
            lst,
            req.filters,
            check_in=req.check_in,
            check_out=req.check_out,
        )
        property_result = match_property_types(lst, req.property_types)
        occupancy_result = match_occupancy_types(lst, req.occupancy_types)

        # Apply strict filtering only to mapped canonical MUST constraints
        if _fails_must(report.matches, structured_must_fields):
            continue

        # strict numeric filter
        if _fails_numeric_filters(numeric_results):
            rejection_reasons["numeric_filters"] += 1
            continue

        if property_result is not None and property_result.value == Ternary.NO:
            rejection_reasons["property_type"] += 1
            continue

        if occupancy_result is not None and occupancy_result.value == Ternary.NO:
            rejection_reasons["occupancy_type"] += 1
            continue

        score, must_yes, must_total, why = _score_listing(
            req,
            report.matches,
            numeric_results=numeric_results,
        )

        if property_result is not None:
            why.append(property_result.why)

        if occupancy_result is not None:
            why.append(occupancy_result.why)

        ranked.append(
            {
                "listing_name": lst.name,
                "listing_id": getattr(lst, "id", None),
                "report": report,
                "matches": report.matches,
                "numeric_results": numeric_results,
                "property_result": property_result,
                "occupancy_result": occupancy_result,
                "score": score,
                "matched_must_count": must_yes,
                "matched_must_total": must_total,
                "why": why,
                "listing": lst,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked






async def orchestrate_search(
    user_text: str,
    intent: Dict[str, Any],
    top_n: int = MAX_ITEMS_HARD_CAP,
    max_items: int = MAX_ITEMS_HARD_CAP,
    result_limit: int | None = None,
    candidate_pool_size: int | None = None,
    source: Source = "fixtures",
    fallback_policy: FallbackPolicy | None = None,
    trace: RequestTrace | None = None,
) -> Dict[str, Any]:
    """Search orchestration tool 
    """
    if result_limit is None:
        result_limit = top_n

    if candidate_pool_size is None:
        candidate_pool_size = max_items
        
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be > 0")

    if result_limit <= 0:
        raise ValueError("result_limit must be > 0")
    
    if trace is None:
        trace = RequestTrace()
    if candidate_pool_size > MAX_ITEMS_HARD_CAP:
        return {
            "need_clarification": True,
            "questions": [f"Too many items requested ({candidate_pool_size}). Please use <= {MAX_ITEMS_HARD_CAP}."],
        }
        
    with trace.step("validate_and_repair_intent"):    
        intent_obj, dropped_requests = await _validate_and_repair_intent(intent, attempts=2)


    with trace.step("resolve_required_search_context"):
        resolved = resolve_required_search_context(intent_obj)

    if resolved.need_clarification:
        return {
            "need_clarification": True,
            "questions": resolved.questions,
            "dropped_requests": dropped_requests,
        }

    with trace.step("build_search_request"):
        req = _build_request(
            user_text=user_text,
            intent_obj=intent_obj,
            city=resolved.city,
            check_in=resolved.check_in,
            check_out=resolved.check_out,
        )

    # Retrieve candidates
    try:
        with trace.step("retrieval", source=source, candidate_pool_size=candidate_pool_size):
            listings = await get_candidates(
                req,
                max_items=candidate_pool_size,
                source=source,
                trace=trace,
            )
    except NotImplementedError:
        return {
            "need_clarification": True,
            "questions": ["Apify retriever is not enabled yet. Using fixtures only for now."],
        }

    with trace.step(
        "city_filter",
        listings_count=len(listings),
        applied=(source == "fixtures"),
        source=source,
    ):
        if source == "fixtures":
            listings = [
                lst for lst in listings
                if _listing_city_matches(lst, req.city)
            ]

    with trace.step(
        "date_filter",
        listings_count=len(listings),
        applied=_should_apply_local_date_filter(source),
        source=source,
    ):
        if _should_apply_local_date_filter(source):
            listings = [
                lst for lst in listings
                if _covers_dates(lst, req.check_in, req.check_out)
            ]

    with trace.step("occupancy_filter", listings_count=len(listings)):
        occupancy_results = {
            getattr(lst, "id", None) or getattr(lst, "url", None) or str(i): evaluate_occupancy(lst, req)
            for i, lst in enumerate(listings)
        }

        filtered_listings = []
        for i, lst in enumerate(listings):
            key = getattr(lst, "id", None) or getattr(lst, "url", None) or str(i)
            occ = occupancy_results[key]
            if occ.passed:
                filtered_listings.append(lst)

        listings = filtered_listings

    filtered_listings = []
    for i, lst in enumerate(listings):
        key = getattr(lst, "id", None) or getattr(lst, "url", None) or str(i)
        occ = occupancy_results[key]
        if occ.passed:
            filtered_listings.append(lst)

    listings = filtered_listings

    # if no candidates after initial filters
    if not listings:
        return {
            "need_clarification": True,
            "questions": ["Nothing found. Try changing your requirements."],
            "request_summary": None,
            "top_results": [],
            "results": [],
            "results_count": 0,
            "debug_notes": [
                "No listings remained after initial city/date/occupancy filtering.",
                f"city={req.city}, check_in={req.check_in}, check_out={req.check_out}",
            ],
            "active_intent": req.model_dump(mode="json", exclude_none=True),
            "dropped_requests": dropped_requests,
        }

    # Structured ranking
    with trace.step("structured_ranking", listings_count=len(listings)):
        ranked = _rank_structured(req, listings)
    
    # Fallback constraint resolution
    if fallback_policy is None:
        fallback_policy = _build_fallback_policy(fallback_top_k=5)

    with trace.step("constraint_fallback_layer", ranked_count=len(ranked)):
        await _apply_constraint_fallback_layer(
            req,
            ranked,
            policy=fallback_policy,
            trace=trace,
        )

    with trace.step("constraint_coverage_normalization", ranked_count=len(ranked)):
        _ensure_unresolved_must_constraints_are_represented(req, ranked)

    # Final scoring
    with trace.step("fallback_scoring"):
        ranked = _apply_constraint_resolution_scoring(ranked)

    with trace.step("post_fallback_structured_filtering", ranked_count=len(ranked)):
        must_constraints, _, _ = _constraints_by_priority(req)
        structured_must_fields = _known_mapped_fields(must_constraints)

        ranked = [
            it
            for it in ranked
            if not _fails_must(it["matches"], structured_must_fields)
            and not _fails_numeric_filters(it.get("numeric_results"))
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
    
    if not ranked:
        debug_notes = ["No listings remained after structured filtering."]

        # --- PRICE ---
        price = req.filters.price if req.filters else None
        if price and price.max_amount is not None:
            debug_notes.append(
                f"Active price filter: max {price.max_amount} {price.currency or 'USD'} {price.scope or ''}".strip()
            )

        # --- BEDROOMS ---
        if req.filters and req.filters.bedrooms_min is not None:
            debug_notes.append(f"Active bedrooms filter: min {req.filters.bedrooms_min}")

        # --- AREA ---
        if req.filters and req.filters.area_sqm_min is not None:
            debug_notes.append(f"Active area filter: min {req.filters.area_sqm_min} sqm")


        # --- PROPERTY TYPE ---
        if req.property_types:
            debug_notes.append(
                "Active property types: " + ", ".join(pt.value for pt in req.property_types)
            )

        # --- CONSTRAINTS (source of truth) ---
        must_constraints = [
            c.normalized_text
            for c in (req.constraints or [])
            if getattr(c, "priority", None) == "must"
        ]
        if must_constraints:
            debug_notes.append(
                "Active must constraints: " + ", ".join(must_constraints)
            )

        nice_constraints = [
            c.normalized_text
            for c in (req.constraints or [])
            if getattr(c, "priority", None) == "nice"
        ]
        if nice_constraints:
            debug_notes.append(
                "Optional constraints: " + ", ".join(nice_constraints)
            )

        return {
            "need_clarification": True,
            "questions": ["Nothing found. Try changing your requirements."],
            "debug_notes": debug_notes,
            "active_intent": req.model_dump(mode="json", exclude_none=True),
            "dropped_requests": dropped_requests,
        }

    with trace.step("final_selection", ranked_count=len(ranked), top_n=result_limit):
        selected = select_ranked_items(ranked, top_n=result_limit)

    with trace.step("normalize_response", selected_count=len(selected)):
        normalized = normalize_search_response(
            req,
            selected,
            top_n=result_limit,
            dropped_requests=dropped_requests,
        )

        payload = normalized.model_dump(mode="json", exclude_none=True)
        payload["constraint_statuses"] = _build_constraint_statuses(selected[: max(0, result_limit)])

    return payload
    
def _format_match_why(field: Field, fm: Any) -> str:
    if fm is None:
        return f"{field.name}: missing match"

    if fm.value == Ternary.YES:
        if fm.evidence and fm.evidence[0].snippet:
            return f"{field.name}: {fm.evidence[0].snippet}"
        return f"{field.name}: matched"

    if fm.value == Ternary.UNCERTAIN:
        return f"{field.name}: maybe (needs check)"

    return f"{field.name}: not found"


def _score_listing(
    req: SearchRequest,
    matches: dict[Field, Any],
    numeric_results: List[Any] | None = None,
) -> Tuple[float, int, int, List[str]]:
    """
    Canonical scoring.

    Structured scoring is applied only to canonical constraints that:
    - have priority must/nice
    - are known
    - have mapped_fields

    Unresolved constraints are handled later by fallback and
    constraint_resolution_results scoring.
    """
    score = 0.0
    why: List[str] = []

    must_constraints, nice_constraints, _ = _constraints_by_priority(req)

    structured_must_fields = _known_mapped_fields(must_constraints)
    structured_nice_fields = _known_mapped_fields(nice_constraints)

    must_total = len(structured_must_fields)
    must_yes = 0

    for f in structured_must_fields:
        fm = matches.get(f)

        if fm is None:
            why.append(_format_match_why(f, fm))
            continue

        if fm.value == Ternary.YES:
            score += 10
            must_yes += 1
        elif fm.value == Ternary.UNCERTAIN:
            pass
        else:
            score -= 100

        why.append(_format_match_why(f, fm))

    for f in structured_nice_fields:
        fm = matches.get(f)
        if fm and fm.value == Ternary.YES:
            score += 1
            if fm.evidence and fm.evidence[0].snippet:
                why.append(f"+ {f.name}: {fm.evidence[0].snippet}")
            else:
                why.append(f"+ {f.name}: matched")

    for nr in (numeric_results or []):
        if nr.value == Ternary.YES:
            score += 10
        elif nr.value == Ternary.UNCERTAIN:
            pass
        else:
            score -= 100

        why.append(nr.why)

    return score, must_yes, must_total, why


def _build_fallback_policy(
    *,
    fallback_top_k: int,
) -> FallbackPolicy:
    return FallbackPolicy(
        enabled=True,
        top_k=fallback_top_k,
        must_only=True,
        run_for_unresolved=True,
        run_for_structured_uncertain=True,
        max_constraints_per_listing=3,
    )

async def _apply_constraint_fallback_layer(
    req: SearchRequest,
    ranked: list[dict],
    *,
    policy: FallbackPolicy,
    trace: RequestTrace | None = None,
) -> None:
    if not policy.enabled:
        for item in ranked:
            item["constraint_resolution_results"] = []
        return

    top_k = policy.normalized_top_k()

    for item in ranked[:top_k]:
        listing = item.get("listing")
        if listing is None:
            item["constraint_resolution_results"] = []
            continue

        results = await resolve_listing_constraints_with_fallback(
            listing=listing,
            constraints=req.constraints or [],
            structured_matches_by_field=item.get("matches", {}),
            policy=policy,
            trace=trace,
        )

        item["constraint_resolution_results"] = [
            r.model_dump(mode="json") for r in results
        ]

    for item in ranked[top_k:]:
        item["constraint_resolution_results"] = []


def _apply_constraint_resolution_scoring(ranked_items: list[dict]) -> list[dict]:
    for item in ranked_items:
        delta = 0.0
        extra_why: list[str] = []

        for r in item.get("constraint_resolution_results", []) or []:
            label = r.get("normalized_text") or "constraint"
            decision = str(r.get("decision") or "").upper()
            priority = str(r.get("priority") or "").lower()

            if priority == "must":
                if decision == "YES":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: {label} confirmed by listing text")
                elif decision == "NO":
                    delta -= 100.0
                    extra_why.append(f"CONSTRAINT_FAIL: must constraint '{label}' not satisfied")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: must constraint '{label}' not confirmed")

            elif priority in {"nice", "nice_to_have"}:
                if decision == "YES":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: nice-to-have '{label}' confirmed by listing text")
                elif decision == "NO":
                    extra_why.append(f"CONSTRAINT_NO_MATCH: nice-to-have '{label}' not satisfied")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: nice-to-have '{label}' not confirmed")

            elif priority == "forbidden":
                if decision == "YES":
                    delta -= 100.0
                    extra_why.append(f"CONSTRAINT_FAIL: forbidden constraint '{label}' detected")
                elif decision == "NO":
                    delta += 3.0
                    extra_why.append(f"CONSTRAINT_MATCH: forbidden constraint '{label}' not detected")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: forbidden constraint '{label}' unclear")

            else:
                # Defensive fallback: do not reward unknown priority.
                if decision == "YES":
                    extra_why.append(f"CONSTRAINT_MATCH: {label} confirmed, but priority is unknown")
                elif decision == "NO":
                    extra_why.append(f"CONSTRAINT_NO_MATCH: {label} not satisfied, but priority is unknown")
                else:
                    extra_why.append(f"CONSTRAINT_UNCERTAIN: {label} not confirmed")

        if delta != 0:
            item["score"] = float(item.get("score", 0.0)) + delta

        why = list(item.get("why") or [])
        why.extend(extra_why)
        item["why"] = why

    ranked_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked_items


def _build_constraint_statuses(ranked_items: list[dict]) -> list[dict]:
    statuses: list[dict] = []

    for item in ranked_items:
        for result in item.get("constraint_resolution_results", []) or []:
            statuses.append(result)

    return statuses
