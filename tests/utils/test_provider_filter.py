import pytest

from mlflow.utils.provider_filter import (
    _parse_provider_list,
    filter_providers,
    is_provider_allowed,
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (None, frozenset()),
        ("", frozenset()),
        ("openai", frozenset({"openai"})),
        ("openai,anthropic", frozenset({"openai", "anthropic"})),
        ("openai, anthropic, gemini", frozenset({"openai", "anthropic", "gemini"})),
        (" openai , anthropic ", frozenset({"openai", "anthropic"})),
        ("OpenAI,Anthropic", frozenset({"openai", "anthropic"})),
        ("openai,,anthropic", frozenset({"openai", "anthropic"})),
        ("  ,  ,  ", frozenset()),
        ("amazon-bedrock", frozenset({"bedrock"})),
        ("amazon-bedrock,openai", frozenset({"bedrock", "openai"})),
    ],
)
def test_parse_provider_list(input_value, expected):
    assert _parse_provider_list(input_value) == expected


def test_filter_providers_no_filter():
    providers = ["openai", "anthropic", "gemini"]
    assert filter_providers(providers) == providers


def test_filter_providers_with_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai,anthropic")
    result = filter_providers(["openai", "anthropic", "gemini", "bedrock"])
    assert result == ["openai", "anthropic"]


def test_filter_providers_case_insensitive(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "OpenAI,ANTHROPIC")
    result = filter_providers(["openai", "anthropic", "gemini"])
    assert result == ["openai", "anthropic"]


def test_is_provider_allowed_no_filter():
    assert is_provider_allowed("openai") is True
    assert is_provider_allowed("litellm") is True


def test_is_provider_allowed_with_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai,anthropic")
    assert is_provider_allowed("openai") is True
    assert is_provider_allowed("anthropic") is True
    assert is_provider_allowed("gemini") is False
    assert is_provider_allowed("litellm") is False


def test_is_provider_allowed_case_insensitive_with_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    assert is_provider_allowed("OpenAI") is True
    assert is_provider_allowed("OPENAI") is True
    assert is_provider_allowed("openai") is True


def test_is_provider_allowed_bedrock_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "bedrock")
    assert is_provider_allowed("bedrock") is True
    assert is_provider_allowed("amazon-bedrock") is True
    assert is_provider_allowed("openai") is False


def test_filter_providers_normalizes_bedrock_alias(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "bedrock,openai")
    result = filter_providers(["openai", "bedrock", "amazon-bedrock", "anthropic"])
    assert result == ["openai", "bedrock", "amazon-bedrock"]
