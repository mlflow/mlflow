from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow.entities.model_registry import Prompt
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize import optimize_prompt
from mlflow.genai.optimize.types import LLMParam, OptimizerConfig
from mlflow.genai.scorers import scorer
from mlflow.tracking._model_registry.fluent import register_prompt


@pytest.fixture
def test_prompt():
    return register_prompt(
        name="test_translation_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "inputs": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
            ],
            "expectations": [{"translation": "Hola"}, {"translation": "Monde"}],
        }
    )


@scorer
def sample_scorer(inputs, outputs, expectations):
    return 1.0


def test_optimize_prompt_basic(test_prompt, sample_data):
    with patch(
        "mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize", return_value="optimized"
    ) as mock_optimizer:
        result = optimize_prompt(
            agent_llm=LLMParam(model_name="test/model"),
            prompt=f"prompts:/{test_prompt.name}/{test_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
        )

    assert isinstance(result.prompt, Prompt)
    assert result.prompt.name == test_prompt.name
    assert result.prompt.version == 2
    assert result.prompt.template == "optimized"
    assert mock_optimizer.call_count == 1


def test_optimize_prompt_unsupported_algorithm(test_prompt, sample_data):
    optimizer_params = OptimizerConfig(algorithm="UnsupportedAlgorithm")

    with pytest.raises(ValueError, match="Algorithm UnsupportedAlgorithm is not supported"):
        optimize_prompt(
            agent_llm=LLMParam(model_name="test/model"),
            prompt=f"prompts:/{test_prompt.name}/{test_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
            optimizer_params=optimizer_params,
        )


def test_optimize_prompt_with_invalid_scorer(test_prompt, sample_data):
    def invalid_scorer(inputs, outputs):
        return 1.0

    with pytest.raises(MlflowException, match="is not a valid scorer"):
        optimize_prompt(
            agent_llm=LLMParam(model_name="test/model"),
            prompt=f"prompts:/{test_prompt.name}/{test_prompt.version}",
            train_data=sample_data,
            scorers=[invalid_scorer],
        )


def test_optimize_prompt_with_trace_scorer(test_prompt, sample_data):
    @scorer
    def trace_scorer(inputs, outputs, expectations, trace):
        return 1.0

    with pytest.raises(MlflowException, match="Trace input is found in Scorer"):
        optimize_prompt(
            agent_llm=LLMParam(model_name="test/model"),
            prompt=f"prompts:/{test_prompt.name}/{test_prompt.version}",
            train_data=sample_data,
            scorers=[trace_scorer],
        )
