import asyncio
from unittest.mock import patch

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, Model

from mlflow.genai.scorers.inspect_ai.models import (
    _PROVIDER_STATE,
    MLFLOW_PROVIDER_NAME,
    _register_mlflow_provider,
    create_inspect_model,
)

_COMPLETE = "mlflow.genai.scorers.llm_backend.ScorerLLMClient.complete"


def test_create_inspect_model_registers_the_mlflow_provider():
    model = create_inspect_model("openai:/gpt-4o-mini")

    assert isinstance(model, Model)
    assert str(model) == f"{MLFLOW_PROVIDER_NAME}/openai:/gpt-4o-mini"
    assert type(model.api).__name__ == "MlflowModelAPI"
    assert model.api._backend.model_name == "openai/gpt-4o-mini"
    assert _PROVIDER_STATE["registered"] is True


def test_register_is_idempotent():
    _register_mlflow_provider()
    _register_mlflow_provider()

    assert str(create_inspect_model("databricks")) == f"{MLFLOW_PROVIDER_NAME}/databricks"


@pytest.mark.parametrize(
    ("model_kwargs", "config", "expected_kwargs"),
    [
        (None, GenerateConfig(), {}),
        ({"temperature": 0.2}, GenerateConfig(), {"temperature": 0.2}),
        (None, GenerateConfig(max_tokens=5, top_p=0.9), {"max_tokens": 5, "top_p": 0.9}),
        ({"temperature": 0.2}, GenerateConfig(temperature=0.7), {"temperature": 0.2}),
    ],
)
def test_generate_forwards_messages_and_config_to_the_backend(
    model_kwargs, config, expected_kwargs
):
    model = create_inspect_model("openai:/gpt-4o-mini", model_kwargs=model_kwargs)
    messages = [ChatMessageSystem(content="Be brief."), ChatMessageUser(content="Hi")]

    with patch(_COMPLETE, return_value="Hello") as mock_complete:
        output = asyncio.run(model.generate(messages, config=config))

    mock_complete.assert_called_once_with(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        **expected_kwargs,
    )
    assert output.completion == "Hello"
    assert output.model == "openai:/gpt-4o-mini"
