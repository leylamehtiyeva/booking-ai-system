from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_PROFILE_NAME = "gemini_default"


class LlmProvider(str, Enum):
    """
    Remote service that ultimately executes an LLM request.
    """

    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"


class AdkModelAdapter(str, Enum):
    """
    ADK integration used to connect to an LLM provider.
    """

    GEMINI = "gemini"
    LITELLM = "litellm"


class LlmModelConfig(BaseModel):
    """
    Provider-agnostic configuration needed to build
    an ADK-compatible model.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    provider: LlmProvider
    adapter: AdkModelAdapter
    model: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    api_base: str | None = None


def get_gemini_model() -> str:
    """
    Return the configured Gemini model name.
    """
    model = os.getenv("GEMINI_MODEL")

    if model:
        return model.strip().strip('"')

    return DEFAULT_GEMINI_MODEL


def get_gemini_fallback_models() -> list[str]:
    """
    Return optional Gemini fallback model names.
    """
    raw = os.getenv("GEMINI_FALLBACK_MODELS")

    if not raw:
        return []

    return [model.strip() for model in raw.split(",") if model.strip()]


def get_llm_profile(
    profile_name: str = DEFAULT_LLM_PROFILE_NAME,
) -> LlmModelConfig:
    """
    Return one named LLM configuration profile.
    """
    profiles = {
        DEFAULT_LLM_PROFILE_NAME: LlmModelConfig(
            provider=LlmProvider.GOOGLE,
            adapter=AdkModelAdapter.GEMINI,
            model=get_gemini_model(),
            api_key_env="GOOGLE_API_KEY",
        ),
    }

    try:
        return profiles[profile_name]
    except KeyError as exc:
        available_profiles = ", ".join(sorted(profiles))

        raise ValueError(
            f"Unknown LLM profile: {profile_name}. "
            f"Available profiles: {available_profiles}"
        ) from exc
