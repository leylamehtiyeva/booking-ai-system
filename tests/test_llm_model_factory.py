from unittest.mock import Mock

import pytest

from app.config.llm import (
    AdkModelAdapter,
    LlmModelConfig,
    LlmProvider,
    get_llm_profile,
    GROQ_GPT_OSS_20B_PROFILE_NAME,
    GEMINI_2_5_FLASH_LITE_PROFILE_NAME,
)
from app.llm import model_factory
from app.llm.model_factory import (
    LlmModelConfigurationError,
    build_adk_model,
)


def test_get_llm_profile_returns_gemini_flash_lite_profile():
    profile = get_llm_profile(
        GEMINI_2_5_FLASH_LITE_PROFILE_NAME
    )

    assert profile.provider == LlmProvider.GOOGLE
    assert profile.adapter == AdkModelAdapter.GEMINI
    assert profile.model == "gemini-2.5-flash-lite"
    assert profile.api_key_env == "GOOGLE_API_KEY"
    assert profile.api_base is None

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


def test_build_adk_model_creates_litellm_adapter(
    monkeypatch,
):
    monkeypatch.setenv(
        "GROQ_API_KEY",
        "test-groq-key",
    )

    litellm_model = object()
    litellm_constructor = Mock(
        return_value=litellm_model,
    )

    monkeypatch.setattr(
        model_factory,
        "LiteLlm",
        litellm_constructor,
    )

    config = LlmModelConfig(
        provider=LlmProvider.GROQ,
        adapter=AdkModelAdapter.LITELLM,
        model="groq/openai/gpt-oss-20b",
        api_key_env="GROQ_API_KEY",
    )

    result = build_adk_model(config)

    assert result is litellm_model

    litellm_constructor.assert_called_once_with(
        model="groq/openai/gpt-oss-20b",
        api_key="test-groq-key",
    )


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


def test_get_llm_profile_returns_groq_profile():
    profile = get_llm_profile(GROQ_GPT_OSS_20B_PROFILE_NAME)

    assert profile.provider == LlmProvider.GROQ
    assert profile.adapter == AdkModelAdapter.LITELLM
    assert profile.model == ("groq/openai/gpt-oss-20b")
    assert profile.api_key_env == "GROQ_API_KEY"
    assert profile.api_base is None
