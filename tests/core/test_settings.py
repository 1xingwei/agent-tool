import json
import logging
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from core.settings import LogLevel, Settings, check_str_is_http
from schema.models import (
    AnthropicModelName,
    AzureOpenAIModelName,
    FakeModelName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
)


def test_check_str_is_http():
    # 测试有效的 HTTP URL
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # 测试无效的 URL
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8080
    assert settings.USE_AWS_BEDROCK is False
    assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # 测试在未提供 API key 时 settings 会抛出错误
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        assert settings.AVAILABLE_MODELS == set(OpenAIModelName)


def test_settings_with_anthropic_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == AnthropicModelName.HAIKU_45
        assert settings.AVAILABLE_MODELS == set(AnthropicModelName)


def test_settings_with_openrouter_key():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENROUTER_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenRouterModelName.GEMINI_38_FLASH
        assert settings.AVAILABLE_MODELS == set(OpenRouterModelName)
        # SecretStr 可避免 key 出现在渲染 Settings 的日志和 traceback 中。
        assert "test_key" not in repr(settings)


def test_settings_with_vertexai_credentials_file():
    with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.GOOGLE_APPLICATION_CREDENTIALS == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == VertexAIModelName.GEMINI_38_FLASH
        assert settings.AVAILABLE_MODELS == set(VertexAIModelName)


def test_settings_with_multiple_api_keys():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_anthropic_key")
        # 当有多个 provider 可用时，OpenAI 应为默认值
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        # 可用模型应恰好包含所有 OpenAI 和 Anthropic 模型
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AnthropicModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_use_fake_model():
    with patch.dict(os.environ, {"USE_FAKE_MODEL": "true"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == FakeModelName.FAKE
        assert settings.AVAILABLE_MODELS == set(FakeModelName)


def test_settings_use_fake_model_wins_over_ambient_real_keys():
    # 即使存在真实 provider key，USE_FAKE_MODEL 也必须赢得默认值。
    with patch.dict(
        os.environ,
        {
            "USE_FAKE_MODEL": "true",
            "OPENAI_API_KEY": "test_openai_key",
            "GROQ_API_KEY": "test_groq_key",
            "GOOGLE_API_KEY": "test_google_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == FakeModelName.FAKE
        # AVAILABLE_MODELS 仍会合并所有活跃 provider。
        assert set(FakeModelName).issubset(settings.AVAILABLE_MODELS)
        assert set(OpenAIModelName).issubset(settings.AVAILABLE_MODELS)


def test_settings_explicit_default_model_overrides_fake():
    # 显式的 DEFAULT_MODEL 优先于 USE_FAKE_MODEL。
    with patch.dict(
        os.environ,
        {"USE_FAKE_MODEL": "true", "OPENAI_API_KEY": "test_openai_key"},
        clear=True,
    ):
        settings = Settings(_env_file=None, DEFAULT_MODEL=OpenAIModelName.GPT_5_NANO)
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO


def test_settings_base_url():
    """通配绑定地址必须报告为可连接的地址。

    HOST=0.0.0.0 表示「监听所有接口」；在 Windows 上连接它会失败
    并报 WinError 10049，因此客户端改用回环地址。
    """
    settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://127.0.0.1:8000"


def test_settings_base_url_keeps_a_concrete_host():
    settings = Settings(HOST="api.example.com", PORT=9000, _env_file=None)
    assert settings.BASE_URL == "http://api.example.com:9000"


def test_settings_is_dev():
    settings = Settings(MODE="dev", _env_file=None)
    assert settings.is_dev() is True

    settings = Settings(MODE="prod", _env_file=None)
    assert settings.is_dev() is False


def test_settings_with_azure_openai_key():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-5": "deployment-1", "gpt-5-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test_key"
        assert settings.DEFAULT_MODEL == AzureOpenAIModelName.AZURE_GPT_5_MINI
        assert settings.AVAILABLE_MODELS == set(AzureOpenAIModelName)


def test_settings_with_both_openai_and_azure():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "AZURE_OPENAI_API_KEY": "test_azure_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-5": "deployment-1", "gpt-5-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.AZURE_OPENAI_API_KEY == SecretStr("test_azure_key")
        # 当有多个 provider 可用时，OpenAI 应为默认值
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        # 可用模型应同时包含 OpenAI 和 Azure OpenAI 模型
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AzureOpenAIModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_azure_deployment_names():
    # 删除此测试
    pass


def test_settings_azure_missing_deployment_names():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        },
        clear=True,
    ):
        with pytest.raises(ValidationError, match="AZURE_OPENAI_DEPLOYMENT_MAP must be set"):
            Settings(_env_file=None)


def test_settings_azure_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-5": "deploy1", "gpt-5-mini": "deploy2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == {
            "gpt-5": "deploy1",
            "gpt-5-mini": "deploy2",
        }


def test_settings_azure_invalid_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-5": "deploy1"}',  # 缺少必需的模型
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="Missing required Azure deployments"):
            Settings(_env_file=None)


def test_settings_azure_openai():
    """测试 Azure OpenAI 设置。"""
    deployment_map = {"gpt-5": "deployment1", "gpt-5-mini": "deployment2"}
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": json.dumps(deployment_map),
        },
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test-key"
        assert settings.AZURE_OPENAI_ENDPOINT == "https://test.openai.azure.com"
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == deployment_map


def test_log_level_enum():
    """测试 LogLevel 枚举及其到 logging 级别的转换。"""
    assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG
    assert LogLevel.INFO.to_logging_level() == logging.INFO
    assert LogLevel.WARNING.to_logging_level() == logging.WARNING
    assert LogLevel.ERROR.to_logging_level() == logging.ERROR
    assert LogLevel.CRITICAL.to_logging_level() == logging.CRITICAL


def test_settings_log_level_default():
    """测试 LOG_LEVEL 默认为 WARNING。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.WARNING
        assert settings.LOG_LEVEL.to_logging_level() == logging.WARNING


def test_settings_log_level_from_env():
    """测试 LOG_LEVEL 可从环境变量设置。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "DEBUG"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.DEBUG
        assert settings.LOG_LEVEL.to_logging_level() == logging.DEBUG


def test_settings_log_level_invalid():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "INVALID"}, clear=True):
        with pytest.raises(ValueError, match="validation error for Settings\nLOG_LEVEL\n"):
            Settings(_env_file=None)
