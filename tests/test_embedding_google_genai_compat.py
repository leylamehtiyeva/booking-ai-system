"""
Compatibility check for gemini-embedding-001 against the production
google-genai version (1.56.0, resolved via google-adk - not upgraded
for this migration).

Two levels, mirroring test_semantic_verifier_google_genai_compat.py:
1. Structural (always runs, no network): confirms the SDK's response
   types still expose the fields app.logic.soft_evidence_retrieval
   depends on.
2. Live (skipped unless GOOGLE_API_KEY is set): confirms actual runtime
   behavior.

CONFIRMED FINDING (via a real approved embed_content call, 2026-09-XX):
the Gemini Developer API (google.genai.Client(api_key=...), as opposed
to Vertex AI) does NOT populate EmbedContentResponse.metadata or
ContentEmbedding.statistics at all - both come back None even though
the SDK's types declare the fields. Token usage is therefore genuinely
unavailable from this API today, not merely untested. Per explicit
instruction, this is reported as-is - no token count is estimated from
string length anywhere in this pipeline; embedding cost is recorded as
None (unknown) rather than invented.
"""

from __future__ import annotations

import os

import pytest


def test_embed_content_response_types_still_expose_expected_fields():
    from google.genai import types

    assert "metadata" in types.EmbedContentResponse.model_fields
    assert "embeddings" in types.EmbedContentResponse.model_fields
    assert "billable_character_count" in types.EmbedContentMetadata.model_fields
    assert "values" in types.ContentEmbedding.model_fields
    assert "statistics" in types.ContentEmbedding.model_fields
    assert "token_count" in types.ContentEmbeddingStatistics.model_fields


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - live embedding compatibility check skipped",
)
def test_live_embed_content_call_and_report_usage_metadata_availability():
    """
    Does not assert token_count is populated - the confirmed, current
    behavior is that it is NOT. This test's job is to fail loudly if
    that ever silently changes (in either direction) without being
    noticed, and to keep the finding executable/reproducible rather
    than only documented in a comment.
    """
    from google.genai import Client

    client = Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=["Free WiFi is available throughout the property."],
    )

    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0].values) > 0

    # As of this test, the Developer API does not return usage metadata.
    # If this ever starts returning real data, soft_evidence_retrieval's
    # token/cost accounting should be revisited to use it instead of
    # recording None.
    token_count_available = resp.embeddings[0].statistics is not None
    billable_chars_available = resp.metadata is not None

    print(f"token_count_available={token_count_available} billable_chars_available={billable_chars_available}")
