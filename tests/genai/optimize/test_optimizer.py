from unittest.mock import patch

import pandas as pd
import pytest

from mlflow.exceptions import MlflowException

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow.entities.model_registry import Prompt
from mlflow.genai.optimize.optimizer import _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import LLMParam, OptimizerParam
from mlflow.genai.scorers import scorer


@pytest.fixture
def mock_mipro():
    with patch("dspy.MIPROv2") as mock:
        yield mock


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "request": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
            ],
            "expectations": [{"translation": "Hola"}, {"translation": "Monde"}],
        }
    )


@pytest.fixture
def sample_prompt():
    return Prompt(
        name="test_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
        version=1,
    )


@scorer
def sample_scorer(inputs, outputs, expectations):
    return 1.0


@pytest.mark.parametrize(
    ("num_candidates", "expected_trials"),
    [
        (4, 6),  # max(2 * log2(4), 1.5 * 4) = max(4, 6) = 6
        (8, 12),  # max(2 * log2(8), 1.5 * 8) = max(6, 12) = 12
        (16, 24),  # max(2 * log2(16), 1.5 * 16) = max(8, 24) = 24
    ],
)
def test_get_num_trials(num_candidates, expected_trials):
    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam(num_instruction_candidates=num_candidates))
    assert optimizer._get_num_trials(num_candidates) == expected_trials


@pytest.mark.parametrize(
    ("train_size", "eval_size", "expected_batch_size"),
    [
        (100, 50, 25),  # min(35, 50/2)
        (100, None, 35),  # min(35, 100/2)
        (40, 20, 10),  # min(35, 20/2)
        (20, None, 10),  # min(35, 20/2)
    ],
)
def test_get_minibatch_size(train_size, eval_size, expected_batch_size):
    train_data = pd.DataFrame({"col": range(train_size)})
    eval_data = pd.DataFrame({"col": range(eval_size)}) if eval_size else None

    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam())
    assert optimizer._get_minibatch_size(train_data, eval_data) == expected_batch_size


def test_format_optimized_prompt():
    import dspy

    mock_program = dspy.Predict("input_text, language -> translation")
    input_fields = {"input_text": str, "language": str}
    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam())

    with patch("dspy.JSONAdapter.format") as mock_format:
        mock_format.return_value = [
            {"role": "system", "content": "You are a translator"},
            {"role": "user", "content": "Input Text: {{input_text}}, Language: {{language}}"},
        ]
        result = optimizer._format_optimized_prompt(dspy.JSONAdapter(), mock_program, input_fields)

    expected = (
        "<system>\nYou are a translator\n</system>\n\n"
        "<user>\nInput Text: {{input_text}}, Language: {{language}}\n</user>"
    )

    assert result == expected
    mock_format.assert_called_once()


def test_optimize_with_teacher_llm(mock_mipro, sample_data, sample_prompt):
    import dspy

    teacher_llm = LLMParam(model_name="test/model")
    optimizer_params = OptimizerParam(
        num_instruction_candidates=4,
        max_few_show_examples=2,
        num_threads=2,
        optimizer_llm=teacher_llm,
    )

    optimizer = _DSPyMIPROv2Optimizer(optimizer_params)

    optimized_program = dspy.Predict("input_text, language -> translation")
    mock_mipro.return_value.compile.return_value = optimized_program

    result = optimizer.optimize(
        prompt=sample_prompt,
        agent_lm=LLMParam(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
        eval_data=None,
    )

    mock_mipro.assert_called_once()
    kwargs = mock_mipro.call_args[1]
    assert "teacher_settings" in kwargs
    assert "lm" in kwargs["teacher_settings"]
    assert isinstance(result, str)


def test_optimize_without_teacher_llm(mock_mipro, sample_data, sample_prompt):
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam(num_instruction_candidates=4))

    optimized_program = dspy.Predict("input_text, language -> translation")
    mock_mipro.return_value.compile.return_value = optimized_program

    result = optimizer.optimize(
        prompt=sample_prompt,
        agent_lm=LLMParam(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
        eval_data=None,
    )

    mock_mipro.assert_called_once()
    kwargs = mock_mipro.call_args[1]
    assert "teacher_settings" in kwargs
    assert kwargs["teacher_settings"] == {}
    assert isinstance(result, str)


def test_optimize_with_eval_data(mock_mipro, sample_data, sample_prompt):
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam())
    eval_data = sample_data.copy()

    optimized_program = dspy.Predict("input_text, language -> translation")
    mock_mipro.return_value.compile.return_value = optimized_program

    optimizer.optimize(
        prompt=sample_prompt,
        agent_lm=LLMParam(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
        eval_data=eval_data,
    )

    compile_args = mock_mipro.return_value.compile.call_args[1]
    assert "trainset" in compile_args
    assert "valset" in compile_args
    assert compile_args["valset"] is not None


def test_convert_to_dspy_metric():
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam())

    def objective(scores):
        return 2 * scores["sample_scorer"]

    metric = optimizer._convert_to_dspy_metric(
        input_fields={"input_text": str, "language": str},
        output_fields={"translation": str},
        scorers=[sample_scorer],
        objective=objective,
    )

    pred = dspy.Example(translation="Hola")
    gold = dspy.Example(translation="Hola")
    state = None

    assert metric(pred, gold, state) == 2.0


def test_convert_to_dspy_metric_raises_on_non_numeric_score():
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerParam())

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
        ValueError,
        match="Non numerical score value found.",
    ):
        metric(
            dspy.Example(input_text="Hello", language="Spanish"),
            dspy.Example(translation="Hola"),
            None,
        )


def test_optimize_prompt_with_old_dspy_version():
    with patch("importlib.metadata.version", return_value="2.5.0"):
        with pytest.raises(MlflowException, match="dspy version is too old"):
            _DSPyMIPROv2Optimizer(OptimizerParam())
