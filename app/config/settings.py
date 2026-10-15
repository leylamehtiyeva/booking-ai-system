import os
from app.config.llm import (
    DEFAULT_LLM_PROFILE_NAME,
    GEMINI_2_5_FLASH_LITE_PROFILE_NAME,
)

MAX_ITEMS_HARD_CAP = 40
TOP_N_DEFAULT = 2  # how many results to return for each search request (default: 2)
FALLBACK_TOP_K_DEFAULT = 5
SOURCE_NAME = "fixtures"
#apify/fixtures

FX_BASE_CURRENCY = os.getenv("FX_BASE_CURRENCY", "USD")
FX_CACHE_TTL_DAYS = int(os.getenv("FX_CACHE_TTL_DAYS", "10"))
FX_CACHE_PATH = os.getenv("FX_CACHE_PATH", "app/resources/fx_rates_usd.json")
FX_API_URL = os.getenv("FX_API_URL", "https://api.frankfurter.dev/v2/rates?base=USD")

CONVERSATION_RESPONSE_LLM_PROFILE = os.getenv(
    "CONVERSATION_RESPONSE_LLM_PROFILE",
    GEMINI_2_5_FLASH_LITE_PROFILE_NAME,
)


DEFAULT_LLM_PROFILE = os.getenv(
    "DEFAULT_LLM_PROFILE",
    DEFAULT_LLM_PROFILE_NAME,
)

ROUTER_LLM_PROFILE = os.getenv(
    "ROUTER_LLM_PROFILE",
    DEFAULT_LLM_PROFILE,
)


CONVERSATION_ROUTER_TIMEOUT_SECONDS = float(
    os.getenv("CONVERSATION_ROUTER_TIMEOUT_SECONDS", "15")
)


CONVERSATION_ROUTER_MAX_ATTEMPTS = max(
    1,
    int(
        os.getenv(
            "CONVERSATION_ROUTER_MAX_ATTEMPTS",
            "2",
        )
    ),
)

CONVERSATION_ROUTER_RETRY_DELAY_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "CONVERSATION_ROUTER_RETRY_DELAY_SECONDS",
            "0.5",
        )
    ),
)


# Failure policy for the non-router LLM stages (intent extraction, intent
# update, factual extraction, textual evidence fallback) - see
# app/llm/call_policy.py. Only rate limits / provider unavailability are
# retried, with exponential backoff from the base delay.
LLM_STAGE_TIMEOUT_SECONDS = float(
    os.getenv("LLM_STAGE_TIMEOUT_SECONDS", "20")
)

LLM_STAGE_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("LLM_STAGE_MAX_ATTEMPTS", "3")),
)

LLM_STAGE_RETRY_BASE_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("LLM_STAGE_RETRY_BASE_DELAY_SECONDS", "1.0")),
)
