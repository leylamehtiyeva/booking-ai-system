from __future__ import annotations

import os
from typing import List


# Основная модель (fallback если .env не задан)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_model() -> str:
    """
    Returns the primary Gemini model.

    Priority:
    1. GEMINI_MODEL from .env
    2. DEFAULT_GEMINI_MODEL
    """
    model = os.getenv("GEMINI_MODEL")

    if model:
        return model.strip().strip('"')

    return DEFAULT_GEMINI_MODEL


def get_gemini_fallback_models() -> List[str]:
    """
    Optional fallback models list (for production resilience).

    You can define in .env:
    GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-1.5-flash
    """
    raw = os.getenv("GEMINI_FALLBACK_MODELS")

    if not raw:
        return []

    return [
        m.strip()
        for m in raw.split(",")
        if m.strip()
    ]
