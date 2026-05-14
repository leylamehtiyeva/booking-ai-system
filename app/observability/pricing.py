from __future__ import annotations

import os


MODEL_PRICES_USD_PER_1M = {
    "gemini-2.0-flash": {
        "input": float(os.getenv("PRICE_GEMINI_2_0_FLASH_INPUT_PER_1M", "0.10")),
        "output": float(os.getenv("PRICE_GEMINI_2_0_FLASH_OUTPUT_PER_1M", "0.40")),
    },
    "gemini-2.5-flash": {
        "input": float(os.getenv("PRICE_GEMINI_2_5_FLASH_INPUT_PER_1M", "0.30")),
        "output": float(os.getenv("PRICE_GEMINI_2_5_FLASH_OUTPUT_PER_1M", "2.50")),
    },
    "gemini-2.5-flash-lite": {
        "input": float(os.getenv("PRICE_GEMINI_2_5_FLASH_LITE_INPUT_PER_1M", "0.10")),
        "output": float(os.getenv("PRICE_GEMINI_2_5_FLASH_LITE_OUTPUT_PER_1M", "0.40")),
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