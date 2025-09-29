from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("dspy", minversion="2.6.0")

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize import optimize_prompt
from mlflow.genai.optimize.optimizers import BasePromptOptimizer
from mlflow.genai.optimize.types import (
    LLMParams,
    OptimizerConfig,
    OptimizerOutput,
    PromptOptimizationResult,
)
from mlflow.genai.scorers import scorer
from mlflow.tracking import MlflowClient
from mlflow.tracking._model_registry.fluent import register_prompt


@pytest.fixture
def sample_prompt():
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


def test_optimize_prompt_basic(sample_prompt, sample_data):
    with patch(
        "mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize",
        return_value=OptimizerOutput(
            final_eval_score=1.0,
            initial_eval_score=0.5,
            optimizer_name="DSPy/MIPROv2",
            optimized_prompt="optimized",
        ),
    ) as mock_optimizer:
        result = optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
        )

    assert isinstance(result, PromptOptimizationResult)
    assert result.prompt.name == sample_prompt.name
    assert result.prompt.version == sample_prompt.version + 1
    assert result.prompt.template == "optimized"
    assert result.prompt.tags["overall_eval_score"] == "1.0"
    assert result.initial_prompt.name == sample_prompt.name
    assert result.initial_prompt.version == sample_prompt.version
    assert result.initial_prompt.template == sample_prompt.template
    assert result.optimizer_name == "DSPy/MIPROv2"
    assert result.final_eval_score == 1.0
    assert result.initial_eval_score == 0.5
    assert mock_optimizer.call_count == 1

    # Verify that default autolog=True behavior includes MLflow run and logging
    run = mlflow.last_active_run()
    assert run is not None
    assert run.data.metrics["final_eval_score"] == 1.0
    assert (
        run.data.params["optimized_prompt_uri"]
        == f"prompts:/{sample_prompt.name}/{sample_prompt.version + 1}"
    )


def test_optimize_prompt_custom_optimizer(sample_prompt, sample_data):
    class _CustomOptimizer(BasePromptOptimizer):
        def optimize(self, prompt, target_llm_params, train_data, scorers, objective, eval_data):
            return OptimizerOutput(
                final_eval_score=1.0,
                initial_eval_score=0.5,
                optimizer_name="CustomOptimizer",
                optimized_prompt="optimized",
            )

    result = optimize_prompt(
        target_llm_params=LLMParams(model_name="test/model"),
        prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
        train_data=sample_data,
        scorers=[sample_scorer],
        optimizer_config=OptimizerConfig(algorithm=_CustomOptimizer),
    )
    assert isinstance(result, PromptOptimizationResult)
    assert result.prompt.name == sample_prompt.name
    assert result.prompt.version == sample_prompt.version + 1
    assert result.prompt.template == "optimized"
    assert result.prompt.tags["overall_eval_score"] == "1.0"
    assert result.initial_prompt.name == sample_prompt.name
    assert result.initial_prompt.version == sample_prompt.version


def test_optimize_prompt_unsupported_algorithm(sample_prompt, sample_data):
    optimizer_config = OptimizerConfig(algorithm="UnsupportedAlgorithm")

    with pytest.raises(ValueError, match="Unsupported algorithm: 'UnsupportedAlgorithm'"):
        optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
            optimizer_config=optimizer_config,
        )


def test_optimize_prompt_with_invalid_scorer(sample_prompt, sample_data):
    def invalid_scorer(inputs, outputs):
        return 1.0

    with pytest.raises(MlflowException, match="Invalid scorer:"):
        optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[invalid_scorer],
        )


def test_optimize_prompt_with_trace_scorer(sample_prompt, sample_data):
    @scorer
    def trace_scorer(inputs, outputs, expectations, trace):
        return 1.0

    with pytest.raises(
        MlflowException, match="Invalid scorer parameter:.*contains 'trace' parameter"
    ):
        optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[trace_scorer],
        )


def test_optimize_autolog(sample_prompt, sample_data):
    with patch(
        "mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize",
        return_value=OptimizerOutput(
            final_eval_score=1.0,
            initial_eval_score=0.5,
            optimizer_name="DSPy/MIPROv2",
            optimized_prompt="optimized",
        ),
    ):
        optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            eval_data=sample_data,
            scorers=[sample_scorer],
            optimizer_config=OptimizerConfig(autolog=True),
        )

    run = mlflow.last_active_run()
    client = MlflowClient()
    assert run is not None
    assert run.data.metrics["final_eval_score"] == 1.0
    assert (
        run.data.params["optimized_prompt_uri"]
        == f"prompts:/{sample_prompt.name}/{sample_prompt.version + 1}"
    )
    expected_params = {
        "optimizer_config.algorithm": "DSPy/MIPROv2",
        "optimizer_config.autolog": "True",
        "optimizer_config.max_few_show_examples": "6",
        "optimizer_config.num_instruction_candidates": "6",
        "optimizer_config.optimizer_llm": "None",
        "optimizer_config.verbose": "False",
        "prompt_uri": "prompts:/test_translation_prompt/1",
        "target_llm_params.model_name": "test/model",
    }
    for key, expected_value in expected_params.items():
        assert run.data.params[key] == expected_value
    artifacts = [x.path for x in client.list_artifacts(run.info.run_id)]
    assert "train_data.json" in artifacts
    assert "eval_data.json" in artifacts


def test_optimize_prompt_no_autolog(sample_prompt, sample_data):
    with patch(
        "mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize",
        return_value=OptimizerOutput(
            final_eval_score=1.0,
            initial_eval_score=0.5,
            optimizer_name="DSPy/MIPROv2",
            optimized_prompt="Optimized: Translate {{input_text}} to {{language}} accurately.",
        ),
    ):
        result = optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
            optimizer_config=OptimizerConfig(autolog=False),
        )

    assert isinstance(result, PromptOptimizationResult)
    assert isinstance(result.prompt, str)
    assert result.prompt == "Optimized: Translate {{input_text}} to {{language}} accurately."
    assert result.initial_prompt.name == sample_prompt.name
    assert result.optimizer_name == "DSPy/MIPROv2"
    assert result.final_eval_score == 1.0

    client = MlflowClient()
    with pytest.raises(MlflowException, match="not found"):
        client.get_prompt_version(sample_prompt.name, sample_prompt.version + 1)
