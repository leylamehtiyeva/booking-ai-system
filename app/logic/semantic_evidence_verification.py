"""
Gemini semantic evidence verifier.

Given one atomic hypothesis and one already-retrieved free-text evidence
snippet, classifies the semantic relation between them:

    SUPPORT | CONTRADICT | NOT_ENOUGH_EVIDENCE

This module does NOT see the whole property context, does not perform
retrieval, does not perform ranking, and does not make the final hotel
recommendation - it only evaluates one (evidence, hypothesis) pair at a
time. It also does not know about retrieval metadata (source_type,
source_path, retrieval_score) - the caller combines
SemanticVerificationResult with that metadata to build an EvidenceItem;
see app.schemas.soft_evidence.

Prompt/response_schema/temperature=0 pattern ported from the tested
evaluation/experiments/verifier_comparison/verifier_c_llm.py, which was
validated against the frozen held-out benchmark - not re-derived here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from pydantic import BaseModel, ConfigDict

from app.observability.llm_usage import (
    record_llm_call_failed,
    record_llm_call_from_response,
)
from app.observability.trace import RequestTrace
from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
from app.schemas.soft_evidence import EvidenceRelation, EvidenceResolutionStatus

STEP_NAME = "semantic_evidence_verifier"

VALID_RELATIONS = {relation.value for relation in EvidenceRelation}

SYSTEM_PROMPT = """You classify the semantic relation between one evidence snippet and one atomic hypothesis about a hotel property.

Return exactly one relation:
- SUPPORT: the evidence text itself provides sufficient semantic support for the hypothesis.
- CONTRADICT: the evidence text explicitly states a fact incompatible with the hypothesis.
- NOT_ENOUGH_EVIDENCE: the evidence is related to the hypothesis's topic, but neither supports nor contradicts it.

Judge only the literal semantic content of the evidence text as given. Do not use outside knowledge about hotels in general. Do not discount the evidence because it reads like marketing copy or a template - judge only what the text explicitly says, not who wrote it or why.

You do not have access to any other information about this property beyond the evidence text given to you. Do not assume anything not stated in the evidence.

Provide a short, one-sentence reason for your classification.""".strip()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_EVIDENCE"]},
        "reason": {"type": "string"},
    },
    "required": ["relation", "reason"],
}


class SemanticVerificationResult(BaseModel):
    """
    Narrow return type of verify_evidence_relation(). Still owns
    nothing about retrieval (no source_type/source_path/
    retrieval_score/evidence_text) - the orchestrator combines this
    with the retrieval-side metadata it already holds to build an
    EvidenceItem.

    Does own this SPECIFIC call's own telemetry (latency/tokens/cost/
    parse_failure), mirroring the LLMCallTrace entry
    record_llm_call_from_response/record_llm_call_failed already
    constructed for it - added so a caller can aggregate a call's usage
    directly from the awaited result, never by peeking at
    trace.llm_calls[-1] (unsafe once verifier calls run concurrently,
    since the last-appended trace entry is not necessarily this call's).

    verify_evidence_relation() always returns one of these - never None,
    never raises - so a caller can never accidentally drop a verification
    attempt. status here is only ever RESOLVED or VERIFICATION_FAILED:
    SKIPPED_CALL_LIMIT is assigned directly by the orchestrator BEFORE
    calling this function at all (a skipped call never reaches Gemini,
    so this module never produces that status itself).
    """

    model_config = ConfigDict(extra="forbid")

    relation: EvidenceRelation | None
    reason: str | None = None
    status: EvidenceResolutionStatus
    error: str | None = None

    parse_failure: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


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


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


async def verify_evidence_relation(
    evidence_text: str,
    hypothesis: str,
    *,
    policy: SemanticVerifierPolicy,
    trace: RequestTrace | None = None,
) -> SemanticVerificationResult:
    user_prompt = json.dumps(
        {"evidence": evidence_text, "hypothesis": hypothesis},
        ensure_ascii=False,
    )

    def _call_sync() -> SemanticVerificationResult:
        # Client/type-module construction is NOT wrapped in the broad
        # except below: a missing GOOGLE_API_KEY or an uninstalled
        # google-genai is a setup/config bug, not a transient operational
        # failure, and must propagate loudly rather than being silently
        # reported as a per-evidence-item VERIFICATION_FAILED (which, on
        # every subsequent call, would just repeat the same failure as
        # confusing noise instead of one clear startup error).
        client = _gemini_client()
        genai_types = _genai_types()

        started = time.perf_counter()

        # ---- 1. API / network / timeout / generate_content exception ----
        # No response object exists at all in this branch. Broad except is
        # deliberate and scoped ONLY to the actual external call: the
        # google-genai SDK does not guarantee a single well-known
        # exception hierarchy for network/timeout/HTTP/auth/rate-limit
        # failures, so narrowing further here would risk missing real
        # operational failure modes. Matches the same scope already used
        # in evaluation/experiments/verifier_comparison/verifier_c_llm.py
        # (client built once outside the try there too).
        try:
            resp = client.models.generate_content(
                model=policy.model,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=user_prompt)],
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=policy.temperature,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            error = f"{type(e).__name__}: {e}"

            call = record_llm_call_failed(
                trace=trace,
                step=STEP_NAME,
                model=policy.model,
                error=error,
                latency_ms=latency_ms,
                parse_failure=False,
            )

            return SemanticVerificationResult(
                relation=None,
                reason=None,
                status=EvidenceResolutionStatus.VERIFICATION_FAILED,
                error=error,
                parse_failure=call.parse_failure,
                latency_ms=call.latency_ms or 0.0,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                total_tokens=call.total_tokens,
                estimated_cost_usd=call.estimated_cost_usd,
            )

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        # ---- 2. Response received, but structured output / JSON / enum parsing fails ----
        raw_text = resp.text or ""
        try:
            data = json.loads(_extract_json(raw_text))
            relation_raw = data.get("relation")
            reason = data.get("reason")

            if relation_raw not in VALID_RELATIONS:
                raise ValueError(f"invalid relation value: {relation_raw!r}")

            relation = EvidenceRelation(relation_raw)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            error = f"{type(e).__name__}: {e}"

            call = record_llm_call_from_response(
                trace=trace,
                step=STEP_NAME,
                model=policy.model,
                response=resp,
                success=False,
                error=error,
                latency_ms=latency_ms,
                parse_failure=True,
            )

            return SemanticVerificationResult(
                relation=None,
                reason=None,
                status=EvidenceResolutionStatus.VERIFICATION_FAILED,
                error=error,
                parse_failure=call.parse_failure,
                latency_ms=call.latency_ms or 0.0,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                total_tokens=call.total_tokens,
                estimated_cost_usd=call.estimated_cost_usd,
            )

        # ---- 3. Valid structured response ----
        call = record_llm_call_from_response(
            trace=trace,
            step=STEP_NAME,
            model=policy.model,
            response=resp,
            success=True,
            latency_ms=latency_ms,
            parse_failure=False,
        )

        return SemanticVerificationResult(
            relation=relation,
            reason=reason,
            status=EvidenceResolutionStatus.RESOLVED,
            error=None,
            parse_failure=call.parse_failure,
            latency_ms=call.latency_ms or 0.0,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            total_tokens=call.total_tokens,
            estimated_cost_usd=call.estimated_cost_usd,
        )

    return await asyncio.to_thread(_call_sync)
