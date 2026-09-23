from __future__ import annotations

import asyncio
import os

# Default Gemini embedding model. This is the same google-genai client
# already used by app.logic.constraint_evidence_resolution for the LLM
# fallback (same GOOGLE_API_KEY, no new dependency) - just a different
# API call (embed_content instead of generate_content).
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# The API rejects more than 100 texts in a single embed_content call
# (BatchEmbedContentsRequest: at most 100 requests per batch).
_MAX_BATCH_SIZE = 100


def _gemini_client():
    """
    Local copy of the client-construction pattern used in
    app.logic.constraint_evidence_resolution._gemini_client().

    Not imported from there on purpose: that helper is private to the
    production fallback module, and this prototype is meant to stay
    isolated from it.
    """
    try:
        from google.genai import Client
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    return Client(api_key=api_key)


def embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """
    Embed a batch of texts via the Gemini embeddings API.

    Blocking call. From async code, use embed_texts_async instead.
    Makes a real, billed API call - do not call from automated code
    paths without confirming the run first.
    """
    if not texts:
        return []

    client = _gemini_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _MAX_BATCH_SIZE):
        batch = texts[start : start + _MAX_BATCH_SIZE]
        response = client.models.embed_content(
            model=model,
            contents=batch,
        )
        vectors.extend(list(embedding.values) for embedding in response.embeddings)

    return vectors


async def embed_texts_async(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    return await asyncio.to_thread(embed_texts, texts, model=model)
