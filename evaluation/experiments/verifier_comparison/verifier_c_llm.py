from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.0

VALID_RELATIONS = {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_EVIDENCE"}

# Fixed before the benchmark run - no few-shot examples from the
# held-out set, no per-example tuning.
SYSTEM_PROMPT = """You classify the semantic relation between one evidence snippet and one atomic hypothesis about a hotel property.

Return exactly one relation:
- SUPPORT: the evidence text itself provides sufficient semantic support for the hypothesis.
- CONTRADICT: the evidence text explicitly states a fact incompatible with the hypothesis.
- NOT_ENOUGH_EVIDENCE: the evidence is related to the hypothesis's topic, but neither supports nor contradicts it.

Judge only the literal semantic content of the evidence text as given. Do not use outside knowledge about hotels in general. Do not discount the evidence because it reads like marketing copy or a template - judge only what the text explicitly says, not who wrote it or why.

You do not have access to any other information about this property beyond the evidence text given to you. Do not assume anything not stated in the evidence.

Provide a short, one-sentence reason for your classification."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_EVIDENCE"]},
        "reason": {"type": "string"},
    },
    "required": ["relation", "reason"],
}


@dataclass
class LlmResult:
    predicted_relation: str | None
    reason: str | None
    raw_response_text: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    parse_failure: bool
    error: str | None = None


def _client():
    from google.genai import Client

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    return Client(api_key=api_key)


def _estimate_cost(prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    # Local copy of app.observability.pricing's gemini-2.5-flash rate,
    # to avoid importing the pydantic-dependent app package tree into
    # this isolated venv for a one-line lookup. Same numbers.
    INPUT_PER_1M = 0.30
    OUTPUT_PER_1M = 2.50
    if prompt_tokens is None:
        return None
    completion_tokens = completion_tokens or 0
    return prompt_tokens / 1_000_000 * INPUT_PER_1M + completion_tokens / 1_000_000 * OUTPUT_PER_1M


class GeminiVerifier:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.client = _client()
        from google.genai import types

        self._types = types

    def predict(self, evidence_text: str, hypothesis: str) -> LlmResult:
        types = self._types
        user_prompt = json.dumps({"evidence": evidence_text, "hypothesis": hypothesis}, ensure_ascii=False)

        started = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000
            return LlmResult(
                predicted_relation=None,
                reason=None,
                raw_response_text="",
                latency_ms=round(latency_ms, 2),
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                estimated_cost_usd=None,
                parse_failure=True,
                error=f"{type(e).__name__}: {e}",
            )

        latency_ms = (time.perf_counter() - started) * 1000

        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)

        raw_text = resp.text or ""
        try:
            data = json.loads(raw_text)
            relation = data.get("relation")
            reason = data.get("reason")
            parse_failure = relation not in VALID_RELATIONS
        except (json.JSONDecodeError, AttributeError):
            relation = None
            reason = None
            parse_failure = True

        return LlmResult(
            predicted_relation=relation if not parse_failure else None,
            reason=reason,
            raw_response_text=raw_text,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=_estimate_cost(prompt_tokens, completion_tokens),
            parse_failure=parse_failure,
        )
