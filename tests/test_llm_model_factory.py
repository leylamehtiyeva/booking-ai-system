from unittest.mock import Mock

import pytest

from app.config.llm import (
    AdkModelAdapter,
    LlmModelConfig,
    LlmProvider,
    get_llm_profile,
)
from app.llm import model_factory
from app.llm.model_factory import (
    LlmModelConfigurationError,
    build_adk_model,
)


def test_get_llm_profile_uses_configured_gemini_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "GEMINI_MODEL",
        "gemini-test-model",
    )

    profile = get_llm_profile()

    assert profile.provider == LlmProvider.GOOGLE
    assert profile.adapter == AdkModelAdapter.GEMINI
    assert profile.model == "gemini-test-model"
    assert profile.api_key_env == "GOOGLE_API_KEY"
    assert profile.api_base is None


def test_get_llm_profile_rejects_unknown_name():
    with pytest.raises(
        ValueError,
        match="Unknown LLM profile: missing-profile",
    ):
        get_llm_profile("missing-profile")


def test_build_adk_model_creates_gemini_adapter(
    monkeypatch,
):
    monkeypatch.setenv(
        "GOOGLE_API_KEY",
        "test-key",
    )

    gemini_model = object()
    gemini_constructor = Mock(
        return_value=gemini_model,
    )

    monkeypatch.setattr(
        model_factory,
        "Gemini",
        gemini_constructor,
    )

    config = LlmModelConfig(
        provider=LlmProvider.GOOGLE,
        adapter=AdkModelAdapter.GEMINI,
        model="gemini-test-model",
        api_key_env="GOOGLE_API_KEY",
    )

    result = build_adk_model(config)

    assert result is gemini_model

    gemini_constructor.assert_called_once_with(
        model="gemini-test-model",
    )


def test_build_adk_model_rejects_missing_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "GOOGLE_API_KEY",
        raising=False,
    )

    config = LlmModelConfig(
        provider=LlmProvider.GOOGLE,
        adapter=AdkModelAdapter.GEMINI,
        model="gemini-test-model",
        api_key_env="GOOGLE_API_KEY",
    )

    with pytest.raises(
        LlmModelConfigurationError,
        match="GOOGLE_API_KEY",
    ):
        build_adk_model(config)


def test_build_adk_model_rejects_unsupported_adapter(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    config = LlmModelConfig(
        provider=LlmProvider.OPENROUTER,
        adapter=AdkModelAdapter.LITELLM,
        model="openrouter/example-model:free",
        api_key_env="OPENROUTER_API_KEY",
        api_base="https://openrouter.ai/api/v1",
    )

    with pytest.raises(
        LlmModelConfigurationError,
        match="Unsupported ADK model adapter: litellm",
    ):
        build_adk_model(config)


def test_build_adk_model_rejects_custom_gemini_key_name(
    monkeypatch,
):
    monkeypatch.setenv(
        "CUSTOM_GEMINI_KEY",
        "test-key",
    )

    config = LlmModelConfig(
        provider=LlmProvider.GOOGLE,
        adapter=AdkModelAdapter.GEMINI,
        model="gemini-test-model",
        api_key_env="CUSTOM_GEMINI_KEY",
    )

    with pytest.raises(
        LlmModelConfigurationError,
        match="GOOGLE_API_KEY or GEMINI_API_KEY",
    ):
        build_adk_model(config)
