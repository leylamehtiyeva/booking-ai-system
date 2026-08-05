from __future__ import annotations

import os

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm

from app.config.llm import (
    AdkModelAdapter,
    LlmModelConfig,
    LlmProvider,
)


class LlmModelConfigurationError(ValueError):
    """
    Raised when an LLM profile cannot build
    a valid ADK model.
    """


def _require_api_key(
    config: LlmModelConfig,
) -> str:
    api_key = os.getenv(config.api_key_env)

    if not api_key:
        raise LlmModelConfigurationError(
            f"Missing API key environment variable: {config.api_key_env}"
        )

    return api_key


def _build_gemini_model(
    config: LlmModelConfig,
) -> BaseLlm:
    if config.provider != LlmProvider.GOOGLE:
        raise LlmModelConfigurationError(
            "The Gemini ADK adapter requires "
            "provider='google'."
        )

    if config.api_key_env not in {
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    }:
        raise LlmModelConfigurationError(
            "google-adk 1.21.0 reads Gemini credentials "
            "only from GOOGLE_API_KEY or GEMINI_API_KEY."
        )

    if config.api_base is not None:
        raise LlmModelConfigurationError(
            "api_base is not supported by "
            "the Gemini ADK adapter."
        )

    _require_api_key(config)

    return Gemini(
        model=config.model,
    )


def _build_litellm_model(
    config: LlmModelConfig,
) -> BaseLlm:
    api_key = _require_api_key(config)

    model_kwargs: dict[str, str] = {
        "model": config.model,
        "api_key": api_key,
    }

    if config.api_base is not None:
        model_kwargs["api_base"] = config.api_base

    return LiteLlm(
        **model_kwargs,
    )


def build_adk_model(
    config: LlmModelConfig,
) -> BaseLlm:
    if config.adapter == AdkModelAdapter.GEMINI:
        return _build_gemini_model(config)

    if config.adapter == AdkModelAdapter.LITELLM:
        return _build_litellm_model(config)

    raise LlmModelConfigurationError(
        f"Unsupported ADK model adapter: {config.adapter.value}"
    )
