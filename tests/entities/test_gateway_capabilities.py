from types import SimpleNamespace

import pytest

from mlflow.entities.gateway_capabilities import (
    SYSTEM_ONE_ACTION,
    endpoint_supported_actions,
    model_supports_system_one,
)


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("typesafe", "jev-latest", True),
        ("typesafe", "jev-1.13.0", True),
        ("openrouter", "typesafe/jev-1.13", True),
        ("openrouter", "~typesafe/jev-latest", True),
        ("openrouter", "typesafe/jev-router", False),
        ("openrouter", "openai/gpt-4o", False),
        ("openai", "typesafe/jev-1.13", False),
    ],
)
def test_model_supports_system_one(provider, model_name, expected):
    assert model_supports_system_one(provider, model_name) is expected


def test_endpoint_supported_actions_requires_all_models_to_support_system_one():
    typesafe = SimpleNamespace(provider="typesafe", model_name="jev-latest")
    openrouter_jev = SimpleNamespace(provider="openrouter", model_name="~typesafe/jev-latest")
    chat = SimpleNamespace(provider="openai", model_name="gpt-4o")

    assert endpoint_supported_actions([typesafe, openrouter_jev]) == [SYSTEM_ONE_ACTION]
    assert endpoint_supported_actions([typesafe, chat]) == []
    assert endpoint_supported_actions([chat]) == []
