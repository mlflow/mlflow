import pytest

from mlflow.assistant.providers import (
    ClaudeCodeProvider,
    CodexProvider,
    MlflowGatewayProvider,
    OllamaProvider,
    resolve_default_provider,
)

_PROVIDERS = {
    "claude_code": ClaudeCodeProvider,
    "codex": CodexProvider,
    "mlflow_gateway": MlflowGatewayProvider,
    "ollama": OllamaProvider,
}


@pytest.fixture
def availability(monkeypatch):
    def set_available(*available: str):
        for name, provider in _PROVIDERS.items():
            monkeypatch.setattr(provider, "is_available", lambda self, n=name: n in available)

    return set_available


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (("claude_code", "codex", "mlflow_gateway"), "claude_code"),
        (("codex", "mlflow_gateway"), "codex"),
        (("mlflow_gateway",), "mlflow_gateway"),
        (("ollama",), None),
        ((), None),
    ],
)
def test_default_provider_precedence(availability, available, expected):
    availability(*available)

    provider = resolve_default_provider()

    assert (provider.name if provider else None) == expected


def test_default_provider_skips_probe_failures(availability, monkeypatch):
    availability("claude_code", "codex")

    def raise_probe_error(self):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(ClaudeCodeProvider, "is_available", raise_probe_error)

    assert resolve_default_provider().name == "codex"


def test_remote_default_provider_uses_remote_safe_gateway(availability):
    availability("claude_code", "codex", "mlflow_gateway")

    assert resolve_default_provider(remote=True).name == "mlflow_gateway"


def test_remote_default_provider_returns_none_without_gateway(availability):
    availability("claude_code", "codex")

    assert resolve_default_provider(remote=True) is None
