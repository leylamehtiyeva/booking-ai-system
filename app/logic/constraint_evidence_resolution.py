from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal
from app.observability.trace import RequestTrace, LLMCallTrace
from pydantic import BaseModel, Field
from app.schemas.fallback_policy import FallbackPolicy

from app.logic.listing_signals import collect_listing_signals
from app.schemas.constraints import (
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field as CanonicalField
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary

ResolverType = Literal["textual", "geo", "hybrid"]
DecisionType = Literal["YES", "NO", "UNCERTAIN"]
ResolutionStatus = Literal["matched", "failed", "uncertain"]

SignalRelation = Literal[
    "supports",
    "contradicts",
    "weakly_supports",
    "weakly_contradicts",
    "irrelevant",
]

SignalStrength = Literal["strong", "medium", "weak"]


class EvidenceSignal(BaseModel):
    relation: SignalRelation
    strength: SignalStrength = "medium"
    snippet: str = ""
    source: str = "other"
    path: str | None = None
    explanation: str = ""


class ConstraintEvidenceAnalysis(BaseModel):
    signals: list[EvidenceSignal] = Field(default_factory=list)

    has_direct_support: bool = False
    has_direct_contradiction: bool = False
    has_only_weak_or_indirect_evidence: bool = False
    has_conflicting_evidence: bool = False
    evidence_missing: bool = True
    condition_or_extra_requirement_present: bool = False

    reason: str = ""



class ConstraintEvidence(BaseModel):
    snippet: str
    source: str
    path: str | None = None


class ConstraintResolutionRequest(BaseModel):
    listing_id: str | None = None
    listing_title: str | None = None

    constraint_id: str | None = None
    raw_text: str
    normalized_text: str

    priority: str
    category: str
    mapping_status: str
    evidence_strategy: str
    mapped_fields: list[str] = Field(default_factory=list)

    structured_value: Literal["YES", "NO", "UNCERTAIN"] | None = None
    resolver_type: ResolverType = "textual"

    listing_evidence: list[dict[str, str]] = Field(default_factory=list)


class ConstraintResolutionResult(BaseModel):
    listing_id: str | None = None
    listing_title: str | None = None

    constraint_id: str | None = None
    raw_text: str
    normalized_text: str

    resolver_type: ResolverType
    decision: DecisionType
    resolution_status: ResolutionStatus
    confidence: float | None = None
    reason: str

    evidence: list[ConstraintEvidence] = Field(default_factory=list)
    analysis: ConstraintEvidenceAnalysis | None = None

    source_stage: Literal["fallback"] = "fallback"
    structured_value_before: Literal["YES", "NO", "UNCERTAIN"] | None = None
    explicit_negative: bool = False


def _gemini_client():
    try:
        from google.genai import Client
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    return Client(api_key=api_key)


def _genai_types():
    try:
        from google.genai import types as genai_types
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e
    return genai_types


def _decision_to_status(decision: DecisionType) -> ResolutionStatus:
    return {
        "YES": "matched",
        "NO": "failed",
        "UNCERTAIN": "uncertain",
    }[decision]
    
def _normalize_evidence_strategy_for_resolution(strategy: str | None) -> str:
    if strategy == "geo":
        return "textual"
    return strategy or "textual"


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _source_from_path(path: str) -> str:
    if path.startswith("listing.facilities"):
        return "facilities"
    if path.startswith("rooms["):
        return "room_facilities"
    if path.startswith("policies["):
        return "policies"
    if path.startswith("highlights["):
        return "highlights"
    if path.startswith("listing.description"):
        return "description"
    if path.startswith("listing.name"):
        return "title"
    if path.startswith("listing.property_type"):
        return "property_type"
    return "other"


def _has_explicit_negative(analysis: ConstraintEvidenceAnalysis) -> bool:
    return any(
        signal.relation == "contradicts" and signal.strength == "strong"
        for signal in analysis.signals
    ) or analysis.has_direct_contradiction
    
    
    
def _decide_from_analysis(analysis: ConstraintEvidenceAnalysis) -> DecisionType:
    strong_support = any(
        signal.relation == "supports" and signal.strength == "strong"
        for signal in analysis.signals
    )

    strong_contradiction = any(
        signal.relation == "contradicts" and signal.strength == "strong"
        for signal in analysis.signals
    )

    if strong_contradiction or analysis.has_direct_contradiction:
        return "NO"

    if (
        strong_support
        and analysis.has_direct_support
        and not analysis.has_conflicting_evidence
        and not analysis.condition_or_extra_requirement_present
    ):
        return "YES"

    return "UNCERTAIN"


def _normalize_result(
    raw: dict[str, Any],
    req: ConstraintResolutionRequest,
) -> ConstraintResolutionResult:
    signals = [
        EvidenceSignal(
            relation=str(signal.get("relation", "irrelevant")).strip(),
            strength=str(signal.get("strength", "medium")).strip(),
            snippet=str(signal.get("snippet", "")).strip(),
            source=str(signal.get("source", "other")).strip() or "other",
            path=signal.get("path"),
            explanation=str(signal.get("explanation", "")).strip(),
        )
        for signal in raw.get("signals", [])
        if isinstance(signal, dict)
    ]

    analysis = ConstraintEvidenceAnalysis(
        signals=signals,
        has_direct_support=bool(raw.get("has_direct_support", False)),
        has_direct_contradiction=bool(raw.get("has_direct_contradiction", False)),
        has_only_weak_or_indirect_evidence=bool(
            raw.get("has_only_weak_or_indirect_evidence", False)
        ),
        has_conflicting_evidence=bool(raw.get("has_conflicting_evidence", False)),
        evidence_missing=bool(raw.get("evidence_missing", not signals)),
        condition_or_extra_requirement_present=bool(
            raw.get("condition_or_extra_requirement_present", False)
        ),
        reason=str(raw.get("reason") or "").strip(),
    )

    decision = _decide_from_analysis(analysis)

    evidence = [
        ConstraintEvidence(
            snippet=signal.snippet,
            source=signal.source,
            path=signal.path,
        )
        for signal in analysis.signals
        if signal.relation in {"supports", "contradicts", "weakly_supports", "weakly_contradicts"}
        and signal.snippet
    ]

    explicit_negative = _has_explicit_negative(analysis)

    reason = analysis.reason
    if not reason:
        if decision == "YES":
            reason = f"{req.normalized_text} is explicitly supported by listing text."
        elif decision == "NO":
            reason = f"{req.normalized_text} is explicitly contradicted by listing text."
        else:
            reason = f"{req.normalized_text} is not explicitly confirmed in the listing."

    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    return ConstraintResolutionResult(
        listing_id=req.listing_id,
        listing_title=req.listing_title,
        constraint_id=req.constraint_id,
        raw_text=req.raw_text,
        normalized_text=req.normalized_text,
        resolver_type=req.resolver_type,
        decision=decision,
        resolution_status=_decision_to_status(decision),
        confidence=confidence,
        reason=reason,
        evidence=evidence,
        analysis=analysis,
        structured_value_before=req.structured_value,
        explicit_negative=explicit_negative,
    )


def _prepare_listing_evidence(
    listing: ListingRaw,
    max_items: int = 40,
    max_chars: int = 240,
) -> list[dict[str, str]]:
    source_rank = {
        "facilities": 0,
        "room_facilities": 1,
        "policies": 2,
        "highlights": 3,
        "description": 4,
        "title": 5,
        "property_type": 6,
        "other": 7,
    }

    prepared: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for signal in collect_listing_signals(listing):
        path = (signal.path or "").strip()
        raw_text = (signal.raw_text or signal.text or "").strip()
        if not path or not raw_text:
            continue

        snippet = raw_text[:max_chars]
        key = (path, snippet.casefold())
        if key in seen:
            continue
        seen.add(key)

        prepared.append(
            {
                "source": _source_from_path(path),
                "path": path,
                "text": snippet,
            }
        )

    prepared.sort(key=lambda x: (source_rank.get(x["source"], 999), len(x["text"])))
    return prepared[:max_items]


def _build_system_prompt() -> str:
    return """
You analyze whether listing evidence supports, contradicts, or does not prove one user constraint.

You are NOT the final decision maker.
Do NOT return YES, NO, or UNCERTAIN.
Your job is only to extract evidence signals.

Return only valid JSON:

{
  "signals": [
    {
      "relation": "supports | contradicts | weakly_supports | weakly_contradicts | irrelevant",
      "strength": "strong | medium | weak",
      "snippet": "exact evidence text",
      "source": "facilities|room_facilities|policies|highlights|description|title|property_type|other",
      "path": "string|null",
      "explanation": "why this evidence has this relation to the constraint"
    }
  ],
  "has_direct_support": false,
  "has_direct_contradiction": false,
  "has_only_weak_or_indirect_evidence": false,
  "has_conflicting_evidence": false,
  "evidence_missing": true,
  "condition_or_extra_requirement_present": false,
  "confidence": 0.0,
  "reason": "short factual explanation of the evidence analysis"
}

Signal meanings:
- supports: evidence directly confirms the constraint.
- contradicts: evidence directly conflicts with the constraint.
- weakly_supports: evidence may support the constraint, but is indirect, vague, partial, or not fully reliable.
- weakly_contradicts: evidence may conflict with the constraint, but is indirect, vague, partial, or not fully reliable.
- irrelevant: evidence is unrelated to the constraint.

Strength meanings:
- strong: direct and explicit evidence.
- medium: reasonably relevant but not fully explicit.
- weak: vague, indirect, marketing-style, or incomplete evidence.

General rules:
- Use only the provided listing evidence.
- Do not invent facts.
- Do not infer from common sense if the listing does not say it.
- Missing evidence is not contradiction.
- Weak evidence is not direct support.
- Conditional evidence should be marked with condition_or_extra_requirement_present=true.
- Conflicting evidence should be marked with has_conflicting_evidence=true.
- If no relevant evidence exists, return an empty signals list and evidence_missing=true.

Contradiction principle:
A contradiction exists when the evidence says the requested property is unavailable, disallowed, paid when the user asked for free/included, shared when the user asked for private, off-site when the user asked for on-site, or otherwise incompatible with the constraint.

Do not make the final YES/NO/UNCERTAIN decision.
Only return evidence signals.
""".strip()

def is_constraint_fallback_eligible(
    constraint: UserConstraint,
    *,
    structured_value: Ternary | None,
    policy: FallbackPolicy,
) -> bool:
    if not policy.enabled:
        return False

    if policy.must_only and constraint.priority != ConstraintPriority.MUST:
        return False

    if (
        policy.run_for_unresolved
        and constraint.mapping_status == ConstraintMappingStatus.UNRESOLVED
    ):
        return True

    if (
        policy.run_for_structured_uncertain
        and structured_value == Ternary.UNCERTAIN
    ):
        return True

    return False

def build_resolution_request(
    *,
    listing: ListingRaw,
    constraint: UserConstraint,
    structured_value: Ternary | None,
) -> ConstraintResolutionRequest:
    normalized_strategy = _normalize_evidence_strategy_for_resolution(
        constraint.evidence_strategy.value
    )

    return ConstraintResolutionRequest(
        listing_id=getattr(listing, "id", None),
        listing_title=getattr(listing, "name", None),
        constraint_id=getattr(constraint, "id", None),
        raw_text=constraint.raw_text,
        normalized_text=constraint.normalized_text,
        priority=constraint.priority.value,
        category=constraint.category.value,
        mapping_status=constraint.mapping_status.value,
        evidence_strategy=normalized_strategy,
        mapped_fields=[f.value if hasattr(f, "value") else str(f) for f in (constraint.mapped_fields or [])],
        structured_value=structured_value.value if structured_value is not None else None,
        resolver_type="textual",
        listing_evidence=_prepare_listing_evidence(listing),
    )
from app.config.llm import get_gemini_model


async def resolve_constraint_via_textual_evidence(
    req: ConstraintResolutionRequest,
    *,
    model: str = get_gemini_model(),
    trace: RequestTrace | None = None,
) -> ConstraintResolutionResult:
    payload = {
        "constraint": {
            "raw_text": req.raw_text,
            "normalized_text": req.normalized_text,
            "priority": req.priority,
            "category": req.category,
            "mapping_status": req.mapping_status,
            "evidence_strategy": req.evidence_strategy,
            "mapped_fields": req.mapped_fields,
            "structured_value": req.structured_value,
        },
        "listing_evidence": req.listing_evidence,
    }

    system = _build_system_prompt()
    user_prompt = json.dumps(payload, ensure_ascii=False)

    def _call_sync() -> ConstraintResolutionResult:
        client = _gemini_client()
        genai_types = _genai_types()

        resp = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_prompt)],
                )
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
            ),
        )
        
        usage = getattr(resp, "usage_metadata", None)

        if trace is not None:
            trace.add_llm_call(
                LLMCallTrace(
                    step="constraint_textual_fallback",
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                    estimated_cost_usd=None,
                    success=True,
                )
            )

        raw_text = resp.text or ""
        raw_json = _extract_json(raw_text)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return _normalize_result(
                {
                    "signals": [],
                    "has_direct_support": False,
                    "has_direct_contradiction": False,
                    "has_only_weak_or_indirect_evidence": False,
                    "has_conflicting_evidence": False,
                    "evidence_missing": True,
                    "condition_or_extra_requirement_present": False,
                    "confidence": 0.0,
                    "reason": (
                        "LLM fallback returned invalid JSON. "
                        f"Raw response: {raw_text[:300]}"
                    ),
                },
                req,
            )

        return _normalize_result(data, req)

    return await asyncio.to_thread(_call_sync)


async def resolve_listing_constraints_with_fallback(
    *,
    listing: ListingRaw,
    constraints: list[UserConstraint],
    structured_matches_by_field: dict[CanonicalField, Any],
    policy: FallbackPolicy,
    trace: RequestTrace | None = None,
) -> list[ConstraintResolutionResult]:
    if not policy.enabled:
        return []

    results: list[ConstraintResolutionResult] = []
    max_constraints = policy.normalized_max_constraints_per_listing()

    for constraint in constraints or []:
        if len(results) >= max_constraints:
            break

        structured_value: Ternary | None = None

        if constraint.mapping_status == ConstraintMappingStatus.KNOWN and constraint.mapped_fields:
            field = constraint.mapped_fields[0]
            fm = structured_matches_by_field.get(field)
            structured_value = fm.value if fm is not None else None

        if not is_constraint_fallback_eligible(
            constraint,
            structured_value=structured_value,
            policy=policy,
        ):
            continue

        req = build_resolution_request(
            listing=listing,
            constraint=constraint,
            structured_value=structured_value,
        )
        result = await resolve_constraint_via_textual_evidence(
            req,
            model=policy.model,
            trace=trace,
        )
        results.append(result)

    return results