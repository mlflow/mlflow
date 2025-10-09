import sys
from contextlib import nullcontext
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("dspy", minversion="2.6.0")

import mlflow
from mlflow import register_prompt
from mlflow.genai.optimize.optimizers import _DSPyMIPROv2Optimizer
from mlflow.genai.optimize.optimizers.utils.dspy_mipro_callback import _DSPyMIPROv2Callback
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig, OptimizerOutput
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


@pytest.mark.parametrize(
    ("optimizer_config", "use_eval_data", "expected_teacher_settings", "trial_logs"),
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
            {1: {"full_eval_score": 0.5}},
        ),
        (
            OptimizerConfig(num_instruction_candidates=4),
            False,
            {},
            {1: {"full_eval_score": 0.5}},
        ),
        (
            OptimizerConfig(),
            True,
            {},
            {1: {"full_eval_score": 0.5}},
        ),
        (
            OptimizerConfig(),
            True,
            {},
            {-1: {"full_eval_score": 0.5}},
        ),
        (
            OptimizerConfig(),
            True,
            {},
            {1: {"full_eval_score": 1.0}},
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
    trial_logs,
):
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(optimizer_config)

    optimized_program = dspy.Predict("input_text, language -> translation")
    optimized_program.score = 1.0
    initial_score = trial_logs.get(1, {}).get("full_eval_score") or trial_logs.get(-1, {}).get(
        "full_eval_score"
    )
    optimized_program.trial_logs = trial_logs
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
    assert isinstance(result, OptimizerOutput)
    assert result.final_eval_score == optimized_program.score
    assert result.initial_eval_score == initial_score
    assert result.optimizer_name == "DSPy/MIPROv2"

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
    assert "Starting prompt optimization" in captured.err
    if initial_score == 1.0 == optimized_program.score:
        assert "Optimization complete! Score remained stable at: 1.0" in captured.err
    else:
        assert "Optimization complete! Initial score: 0.5. Final score: 1.0" in captured.err


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


@pytest.mark.parametrize(
    "autolog",
    [
        False,
        True,
    ],
)
def test_optimize_with_autolog(
    mock_mipro,
    sample_data,
    sample_prompt,
    mock_extractor,
    autolog,
):
    import dspy

    optimizer = _DSPyMIPROv2Optimizer(OptimizerConfig(autolog=autolog))

    callbacks = []

    optimized_program = dspy.Predict("input_text, language -> translation")
    optimized_program.score = 1.0

    def fn(*args, **kwargs):
        nonlocal callbacks
        callbacks = dspy.settings.callbacks
        return optimized_program

    mock_mipro.return_value.compile.side_effect = fn

    context = mlflow.start_run() if autolog else nullcontext()
    with context:
        optimizer.optimize(
            prompt=sample_prompt,
            target_llm_params=LLMParams(model_name="agent/model"),
            train_data=sample_data,
            scorers=[sample_scorer],
            eval_data=sample_data,
        )

    if autolog:
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], _DSPyMIPROv2Callback)
    else:
        assert len(callbacks) == 0
