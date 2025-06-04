import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow import register_prompt
from mlflow.exceptions import MlflowException

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow.entities.model_registry import Prompt
from mlflow.genai.optimize.optimizers import _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig
from mlflow.genai.scorers import scorer


@pytest.fixture
def mock_mipro():
    with patch("dspy.MIPROv2") as mock:
        yield mock


@pytest.fixture
def mock_extractor():
    with patch(
        "mlflow.genai.optimize.optimizers._DSPyMIPROv2Optimizer._extract_instructions",
        new=lambda self, template, lm: template,
    ) as mock:
        yield mock


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


@pytest.fixture
def sample_prompt():
    return register_prompt(
        name="test_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
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
    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig(num_instruction_candidates=num_candidates))
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

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())
    assert optimizer._get_minibatch_size(train_data, eval_data) == expected_batch_size


def test_format_optimized_prompt():
    import dspy

    mock_program = dspy.Predict("input_text, language -> translation")
    input_fields = {"input_text": str, "language": str}
    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())

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


@pytest.mark.parametrize(
    ("optimizer_config", "use_eval_data", "expected_teacher_settings"),
    [
        (
            OptimizerConfig(
                num_instruction_candidates=4,
                max_few_show_examples=2,
                num_threads=2,
                optimizer_llm=LLMParams(model_name="test/model"),
            ),
            False,
            {"lm": True},
        ),
        (
            OptimizerConfig(num_instruction_candidates=4),
            False,
            {},
        ),
        (
            OptimizerConfig(),
            True,
            {},
        ),
    ],
)
def test_optimize_scenarios(
    mock_mipro,
    sample_data,
    sample_prompt,
    mock_extractor,
    capsys,
    optimizer_config,
    use_eval_data,
    expected_teacher_settings,
):
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(optimizer_config)

    optimized_program = dspy.Predict("input_text, language -> translation")
    optimized_program.score = 1.0
    optimized_program.trial_logs = {
        1: {"full_eval_score": 0.0},
    }
    mock_mipro.return_value.compile.return_value = optimized_program

    # Prepare eval_data if needed
    eval_data = sample_data.copy() if use_eval_data else None

    result = optimizer.optimize(
        prompt=sample_prompt,
        target_llm_params=LLMParams(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
        eval_data=eval_data,
    )

    # Verify teacher LLM settings
    mock_mipro.assert_called_once()
    kwargs = mock_mipro.call_args[1]
    assert "teacher_settings" in kwargs
    if expected_teacher_settings:
        assert "lm" in kwargs["teacher_settings"]
    else:
        assert kwargs["teacher_settings"] == {}

    # Verify optimization result
    assert isinstance(result, Prompt)
    assert result.version == 2
    assert result.version_metadata["overall_eval_score"] == "1.0"

    # Verify eval data handling
    compile_args = mock_mipro.return_value.compile.call_args[1]
    assert "trainset" in compile_args
    assert "valset" in compile_args
    if eval_data is not None:
        assert compile_args["valset"] is not None
    else:
        assert compile_args["valset"] is None

    # Verify logging
    captured = capsys.readouterr()
    assert "Started optimizing prompt" in captured.err
    assert "Please wait as this process typically takes several minutes" in captured.err
    assert (
        "Prompt optimization completed. Evaluation score changed from 0.0 to 1.0." in captured.err
    )


def test_convert_to_dspy_metric():
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())

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

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())

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
            _DSPyMIPROv2Optimizer(OptimizerConfig())


def test_validate_input_fields_with_missing_variables():
    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())
    prompt = Prompt(
        name="test_prompt",
        template="Translate {{text}} to {{language}} and explain in {{style}}",
        version=1,
    )
    input_fields = {"text": str, "language": str}  # Missing 'style' variable

    with pytest.raises(
        MlflowException,
        match=r"The following variables of the prompt are missing from the dataset: {'style'}",
    ):
        optimizer._validate_input_fields(input_fields, prompt)


def test_extract_instructions():
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig())
    mock_lm = MagicMock(spec=dspy.LM)
    template = "Translate {{text}} to {{language}}"

    with patch(
        "dspy.Predict.forward", return_value=dspy.Prediction(instruction="extracted system message")
    ) as mock_forward:
        result = optimizer._extract_instructions(template, mock_lm)

    mock_forward.assert_called_once_with(prompt=template)

    assert result == "extracted system message"


@pytest.mark.parametrize(
    "verbose",
    [
        False,
        True,
    ],
)
def test_optimize_with_verbose(
    mock_mipro, sample_data, sample_prompt, mock_extractor, verbose, capsys
):
    import dspy

    mock_mipro.return_value.compile.side_effect = lambda *args, **kwargs: (
        print("DSPy optimization progress")  # noqa: T201
        or print("DSPy debug info", file=sys.stderr)  # noqa: T201
        or dspy.Predict("input_text, language -> translation")
    )

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig(verbose=verbose))

    optimizer.optimize(
        prompt=sample_prompt,
        target_llm_params=LLMParams(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
    )

    captured = capsys.readouterr()
    if verbose:
        assert "DSPy optimization progress" in captured.out
        assert "DSPy debug info" in captured.err
    else:
        assert "DSPy optimization progress" not in captured.out
        assert "DSPy debug info" not in captured.err

    mock_mipro.assert_called_once()
