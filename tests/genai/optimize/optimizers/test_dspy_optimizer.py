from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow import register_prompt
from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.dspy_optimizer import DSPyPromptOptimizer
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig, OptimizerOutput
from mlflow.genai.scorers import scorer


class _TestDSPyPromptOptimizer(DSPyPromptOptimizer):
    """Test implementation of DSPyPromptOptimizer for testing the base class functionality."""

    def run_optimization(
        self,
        prompt,
        program,
        metric,
        train_data,
        eval_data,
    ) -> OptimizerOutput:
        optimized_program = program
        optimized_program.demos = [train_data[0]] if train_data else []

        # Create optimized prompt template
        optimized_template = f"Optimized: {prompt.template}"

        return OptimizerOutput(
            optimized_prompt=optimized_template,
            final_eval_score=0.85,
            initial_eval_score=0.5,
            optimizer_name="TestDSPyOptimizer",
        )


@pytest.fixture(
    params=[
        pytest.param(True, id="extract_instructions_true"),
        pytest.param(False, id="extract_instructions_false"),
    ],
    name="optimizer_config",
)
def optimizer_config_fixture(request):
    return OptimizerConfig(
        optimizer_llm=LLMParams(
            model_name="openai:/gpt-3.5-turbo",
        ),
        verbose=True,
        extract_instructions=request.param,
    )


@pytest.fixture
def target_llm_params():
    return LLMParams(
        model_name="openai:/gpt-4", temperature=0.2, base_uri="https://api.openai.com/v1"
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "inputs": [
                {"question": "What is 2+2?", "context": "Math"},
                {"question": "What is 3+3?", "context": "Math"},
            ],
            "expectations": [
                {"answer": "4"},
                {"answer": "6"},
            ],
        }
    )


@pytest.fixture
def sample_prompt():
    return register_prompt(
        name="test_prompt",
        template="Answer the question: {question} using context: {context}",
    )


@pytest.fixture
def mock_scorer():
    @scorer
    def accuracy(inputs, outputs, expectations):
        return 1.0 if outputs.get("answer") == expectations.get("answer") else 0.0

    return accuracy


@pytest.fixture
def mock_extractor():
    with patch(
        "mlflow.genai.optimize.optimizers.dspy_optimizer.DSPyPromptOptimizer._extract_instructions",
        return_value="Answer questions accurately",
    ) as mock:
        yield mock


def test_optimize_basic_functionality(
    target_llm_params, sample_data, sample_prompt, mock_scorer, mock_extractor, optimizer_config
):
    optimizer = _TestDSPyPromptOptimizer(optimizer_config)

    result = optimizer.optimize(
        prompt=sample_prompt,
        target_llm_params=target_llm_params,
        train_data=sample_data,
        scorers=[mock_scorer],
        eval_data=None,
    )

    # Verify result
    assert isinstance(result, OptimizerOutput)
    assert result.optimized_prompt.startswith("Optimized:")
    assert result.final_eval_score == 0.85
    assert result.initial_eval_score == 0.5
    assert result.optimizer_name == "TestDSPyOptimizer"
    if optimizer_config.extract_instructions:
        mock_extractor.assert_called_once()
    else:
        mock_extractor.assert_not_called()


def test_convert_to_dspy_metric(mock_scorer):
    import dspy

    optimizer = _TestDSPyPromptOptimizer(OptimizerConfig())

    def objective(scores):
        return 2 * scores["accuracy"]

    metric = optimizer._convert_to_dspy_metric(
        input_fields={"input_text": str, "language": str},
        output_fields={"translation": str},
        scorers=[mock_scorer],
        objective=objective,
    )

    pred = dspy.Example(translation="Hola")
    gold = dspy.Example(translation="Hola")
    state = None

    assert metric(pred, gold, state) == 2.0


def test_convert_to_dspy_metric_raises_on_non_numeric_score():
    import dspy

    optimizer = _TestDSPyPromptOptimizer(OptimizerConfig())

    @scorer
    def non_numeric_scorer(inputs, outputs, expectations):
        return "good"

    metric = optimizer._convert_to_dspy_metric(
        input_fields={"input_text": str, "language": str},
        output_fields={"translation": str},
        scorers=[non_numeric_scorer],
        objective=None,
    )

    with pytest.raises(
        MlflowException,
        match=r"Scorer \[non_numeric_scorer\] return a string, Assessment or a list of Assessment.",
    ):
        metric(
            dspy.Example(input_text="Hello", language="Spanish"),
            dspy.Example(translation="Hola"),
            None,
        )


def test_optimize_prompt_with_old_dspy_version():
    with patch("importlib.metadata.version", return_value="2.5.0"):
        with pytest.raises(MlflowException, match="Current dspy version 2.5.0 is unsupported"):
            _TestDSPyPromptOptimizer(OptimizerConfig())


def test_validate_input_fields_with_missing_variables():
    optimizer = _TestDSPyPromptOptimizer(OptimizerConfig())
    prompt = PromptVersion(
        name="test_prompt",
        template="Translate {{text}} to {{language}} and explain in {{style}}",
        version=1,
    )
    input_fields = {"text": str, "language": str}  # Missing 'style' variable

    with pytest.raises(
        MlflowException,
        match=r"Validation failed. Missing prompt variables in dataset: {'style'}",
    ):
        optimizer._validate_input_fields(input_fields, prompt)


def test_extract_instructions():
    import dspy

    optimizer = _TestDSPyPromptOptimizer(OptimizerConfig())
    mock_lm = MagicMock(spec=dspy.LM)
    template = "Translate {{text}} to {{language}}"

    with patch(
        "dspy.Predict.forward", return_value=dspy.Prediction(instruction="extracted system message")
    ) as mock_forward:
        result = optimizer._extract_instructions(template, mock_lm)

    mock_forward.assert_called_once_with(prompt=template)

    assert result == "extracted system message"


def test_parse_model_name():
    optimizer = _TestDSPyPromptOptimizer(OptimizerConfig())

    assert optimizer._parse_model_name("openai:/gpt-4") == "openai/gpt-4"
    assert optimizer._parse_model_name("anthropic:/claude-3") == "anthropic/claude-3"
    assert optimizer._parse_model_name("mistral:/mistral-7b") == "mistral/mistral-7b"

    # Test that already formatted names are unchanged
    assert optimizer._parse_model_name("openai/gpt-4") == "openai/gpt-4"
    assert optimizer._parse_model_name("anthropic/claude-3") == "anthropic/claude-3"

    # Test invalid formats raise errors
    with pytest.raises(MlflowException, match="Invalid model name format"):
        optimizer._parse_model_name("invalid-model-name")

    with pytest.raises(MlflowException, match="Model name cannot be empty"):
        optimizer._parse_model_name("")

    with pytest.raises(MlflowException, match="Invalid model name format"):
        optimizer._parse_model_name("openai:")

    with pytest.raises(MlflowException, match="Invalid model name format"):
        optimizer._parse_model_name("openai/")
