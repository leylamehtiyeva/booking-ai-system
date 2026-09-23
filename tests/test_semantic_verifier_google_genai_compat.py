"""
Compatibility check for the production google-genai version (per
pyproject.toml, resolved via google-adk's own dependency - not upgraded
for this migration; see app/logic/semantic_evidence_verification.py).

Two levels:
1. Structural (always runs, no network): confirms GenerateContentConfig
   actually exposes the fields the verifier module depends on.
2. Live (skipped unless GOOGLE_API_KEY is set): confirms the installed
   google-genai version's structured-output behavior actually matches
   what evaluation/experiments/verifier_comparison/verifier_c_llm.py
   validated - field presence alone doesn't prove runtime behavior.
"""

from __future__ import annotations

import os

import pytest


def test_generate_content_config_supports_structured_output_fields():
    from google.genai import types

    fields = types.GenerateContentConfig.model_fields
    for required_field in (
        "system_instruction",
        "temperature",
        "response_mime_type",
        "response_schema",
    ):
        assert required_field in fields, (
            f"google-genai's GenerateContentConfig no longer exposes "
            f"'{required_field}' - the semantic verifier's structured "
            f"output call needs to be re-verified before use."
        )


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - live google-genai compatibility check skipped",
)
async def test_live_structured_output_call_returns_valid_json(monkeypatch):
    from app.logic.semantic_evidence_verification import verify_evidence_relation
    from app.schemas.semantic_verifier_policy import SemanticVerifierPolicy
    from app.schemas.soft_evidence import EvidenceResolutionStatus

    result = await verify_evidence_relation(
        "Free WiFi is available throughout the property.",
        "Wi-Fi / internet access is available.",
        policy=SemanticVerifierPolicy(),
        trace=None,
    )

    assert result.status == EvidenceResolutionStatus.RESOLVED, result.error
    assert result.relation is not None
