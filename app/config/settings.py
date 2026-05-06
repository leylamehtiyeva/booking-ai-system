import os

MAX_ITEMS_HARD_CAP = 400
TOP_N_DEFAULT = 2
FALLBACK_TOP_K_DEFAULT = 5
SOURCE_NAME = "fixtures"

FX_BASE_CURRENCY = os.getenv("FX_BASE_CURRENCY", "USD")
FX_CACHE_TTL_DAYS = int(os.getenv("FX_CACHE_TTL_DAYS", "10"))
FX_CACHE_PATH = os.getenv("FX_CACHE_PATH", "app/resources/fx_rates_usd.json")
FX_API_URL = os.getenv("FX_API_URL", "https://api.frankfurter.dev/v2/rates?base=USD")