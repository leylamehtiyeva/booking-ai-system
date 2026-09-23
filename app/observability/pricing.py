from __future__ import annotations

import os


MODEL_PRICES_USD_PER_1M = {
    "gemini-2.0-flash": {
        "input": float(
            os.getenv(
                "PRICE_GEMINI_2_0_FLASH_INPUT_PER_1M",
                "0.10",
            )
        ),
        "output": float(
            os.getenv(
                "PRICE_GEMINI_2_0_FLASH_OUTPUT_PER_1M",
                "0.40",
            )
        ),
    },
    "gemini-2.5-flash": {
        "input": float(
            os.getenv(
                "PRICE_GEMINI_2_5_FLASH_INPUT_PER_1M",
                "0.30",
            )
        ),
        "output": float(
            os.getenv(
                "PRICE_GEMINI_2_5_FLASH_OUTPUT_PER_1M",
                "2.50",
            )
        ),
    },
    "gemini-2.5-flash-lite": {
        "input": float(
            os.getenv(
                "PRICE_GEMINI_2_5_FLASH_LITE_INPUT_PER_1M",
                "0.10",
            )
        ),
        "output": float(
            os.getenv(
                "PRICE_GEMINI_2_5_FLASH_LITE_OUTPUT_PER_1M",
                "0.40",
            )
        ),
    },
    "groq/openai/gpt-oss-20b": {
        "input": float(
            os.getenv(
                "PRICE_GROQ_GPT_OSS_20B_INPUT_PER_1M",
                "0.075",
            )
        ),
        "output": float(
            os.getenv(
                "PRICE_GROQ_GPT_OSS_20B_OUTPUT_PER_1M",
                "0.30",
            )
        ),
    },
    # Verified against ai.google.dev/gemini-api/docs/pricing. No output/
    # completion tokens exist for an embedding call - "output" is 0 so
    # estimate_llm_cost_usd(model=..., prompt_tokens=..., completion_tokens=0)
    # can be reused unchanged rather than adding a separate cost function.
    # As of this pricing entry being added, the Developer API's
    # embed_content does not actually return token usage (confirmed via
    # a live call - see tests/test_embedding_google_genai_compat.py), so
    # this price is currently unused until/unless that changes; it is
    # not applied to any estimated string-length-based token count.
    "gemini-embedding-001": {
        "input": float(
            os.getenv(
                "PRICE_GEMINI_EMBEDDING_001_INPUT_PER_1M",
                "0.15",
            )
        ),
        "output": 0.0,
    },
}


def estimate_tokens_from_text(text: str | None) -> int:
    if text is None:
        return 0

    text = str(text)

    if not text.strip():
        return 0

    return max(1, round(len(text) / 4))


def _resolve_model_pricing(model: str) -> dict | None:
    if model in MODEL_PRICES_USD_PER_1M:
        return MODEL_PRICES_USD_PER_1M[model]

    # fallback 
    for key, value in MODEL_PRICES_USD_PER_1M.items():
        if model.startswith(key):
            return value

    return None


def estimate_llm_cost_usd(
    *,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    prices = _resolve_model_pricing(model)

    if prices is None:
        return None

    if prompt_tokens is None:
        return None

    if completion_tokens is None:
        completion_tokens = 0

    return (
        prompt_tokens / 1_000_000 * prices["input"]
        + completion_tokens / 1_000_000 * prices["output"]
    )


def estimate_apify_cost_usd(*, run_count: int = 1) -> float:
    value = os.getenv("APIFY_BOOKING_COST_PER_RUN_USD")
    return run_count * float(value or "0.0")